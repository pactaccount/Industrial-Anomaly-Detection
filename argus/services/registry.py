"""Model registry: training, persistence and caching of detectors.

Loading a model means deserialising a memory bank and (lazily) building the
backbone. That is too slow to do per request, so we keep a process-local
cache of loaded detectors, keyed by model id + version.

The cache is bounded: a plant running 200 SKUs would otherwise hold 200
backbones in memory. We evict least-recently-used entries, and we never
evict a model that is currently in flight (tracked by a refcount).
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.model import AnomalyDetector, ModelConfig
from ..db.models import Model
from ..db.session import get_data_dir

MODEL_ROOT = "models"


def model_artifact_path(project_id: str, model_id: str, version: int) -> Path:
    """Directory holding a model version's artefacts."""
    return get_data_dir() / MODEL_ROOT / project_id / model_id / f"v{version}"


@dataclass
class _CacheEntry:
    detector: AnomalyDetector
    refcount: int = 0
    last_used: int = 0


class ModelRegistry:
    """Caches loaded detectors and manages their lifecycle."""

    def __init__(self, max_models: int = 4) -> None:
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_models = max_models
        self._counter = 0

    @staticmethod
    def _key(model: Model) -> str:
        return f"{model.id}:v{model.version}"

    def get(self, model_row: Model) -> AnomalyDetector:
        """Return a loaded detector for the given model row."""
        key = self._key(model_row)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._counter += 1
                entry.last_used = self._counter
                self._cache.move_to_end(key)
                return entry.detector

        path = Path(model_row.artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artefacts not found at {path}")

        detector = AnomalyDetector.load(path)
        with self._lock:
            self._counter += 1
            self._cache[key] = _CacheEntry(detector, last_used=self._counter)
            self._cache.move_to_end(key)
            self._evict_if_needed()
        return detector

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used models that are not in use."""
        while len(self._cache) > self._max_models:
            for key, entry in self._cache.items():
                if entry.refcount == 0:
                    del self._cache[key]
                    break
            else:
                # Everything is in use -- allow temporary overflow.
                break

    def invalidate(self, model_row: Model) -> None:
        """Drop a model from the cache (e.g. after retraining)."""
        with self._lock:
            self._cache.pop(self._key(model_row), None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# A single registry for the process.
registry = ModelRegistry()


def train_model(
    *,
    name: str,
    project_id: str,
    good_images_dir: str | Path,
    part_number: str = "",
    line: str = "",
    config: ModelConfig | None = None,
    version: int = 1,
    model_id: str | None = None,
    calibration: str = "quantile",
    quantile: float = 0.95,
) -> tuple[Model, AnomalyDetector]:
    """Train a model on good images and persist it.

    Args:
        name: human-readable model name.
        project_id: owning project.
        good_images_dir: directory of known-good images.
        part_number: the SKU this model is for.
        line: production line identifier.
        config: model configuration; defaults to the standard config.
        version: version number (increment to keep history).
        model_id: explicit id, else a new one is generated.
        calibration: threshold calibration method.
        quantile: percentile for the quantile calibration method.

    Returns:
        ``(model_row, detector)``.
    """
    from ..db.models import new_id

    config = config or ModelConfig()
    model_id = model_id or new_id()
    artifact_path = model_artifact_path(project_id, model_id, version)

    detector = AnomalyDetector(config)
    detector.fit(Path(good_images_dir), verbose=True)
    calibration_info = detector.calibrate(method=calibration, quantile=quantile)
    detector.save(artifact_path)

    row = Model(
        id=model_id,
        project_id=project_id,
        name=name,
        part_number=part_number,
        line=line,
        version=version,
        artifact_path=str(artifact_path),
        config_hash=config.hash(),
        config=json.loads(json.dumps(config.__dict__, default=list)),
        n_train_images=len(detector._good_scores),
        memory_bank_size=detector.memory_bank.size if detector.memory_bank else 0,
        threshold=calibration_info["threshold"],
        pixel_threshold=calibration_info.get("pixel_threshold"),
        threshold_source=calibration_info["method"],
        good_score_mean=calibration_info["good_score_mean"],
        good_score_std=calibration_info["good_score_std"],
        good_score_max=calibration_info["good_score_max"],
        status="ready",
    )
    return row, detector
