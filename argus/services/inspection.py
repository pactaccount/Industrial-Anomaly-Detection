"""Inspection pipeline: score an image, localise defects, persist the record.

The localisation step is what turns "this image scored 8.2" into "there is a
4.3mm crack at (412, 288) covering 0.4% of the part". An operator needs
coordinates and a size, because "anomaly detected" is not actionable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from ..core.model import DetectionResult
from ..db.models import Defect, Inspection, Model, new_id
from . import storage
from .registry import registry


@dataclass
class LocalisedDefect:
    """A defect region in original-image coordinates."""

    x: int
    y: int
    width: int
    height: int
    area_px: int
    area_ratio: float
    peak_score: float
    mean_score: float


def localise_defects(
    heatmap: np.ndarray,
    threshold: float,
    min_area_px: int = 16,
) -> list[LocalisedDefect]:
    """Find connected regions in the anomaly map above a threshold.

    Args:
        heatmap: float map in original image coordinates.
        threshold: score above which a pixel counts as defective.
        min_area_px: discard regions smaller than this (noise).

    Returns:
        A list of regions, largest first.
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # Close small gaps so a single physical defect split by noise is one blob.
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    total_px = heatmap.size
    results: list[LocalisedDefect] = []

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area_px:
            continue
        region = heatmap[labels == label]
        results.append(
            LocalisedDefect(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                area_px=int(area),
                area_ratio=float(area / total_px),
                peak_score=float(region.max()),
                mean_score=float(region.mean()),
            )
        )

    results.sort(key=lambda d: d.area_px, reverse=True)
    return results


def inspect_image(
    *,
    db: Session,
    model_row: Model,
    image: Image.Image,
    source: str = "api",
    save_artifacts: bool = True,
    image_format: str | None = None,
) -> Inspection:
    """Run a full inspection and persist the result.

    Args:
        db: database session.
        model_row: the model to inspect against.
        image: the image to inspect.
        source: where the image came from (api, upload, camera, ...).
        save_artifacts: persist the image, heatmap and overlay.
        image_format: format to store the original in.

    Returns:
        The persisted Inspection row.
    """
    detector = registry.get(model_row)
    result: DetectionResult = detector.score(image)

    # Use the model's calibrated pixel threshold when the detector was loaded
    # without one (e.g. loaded from disk before calibration was stored).
    pixel_threshold = model_row.pixel_threshold or detector.pixel_threshold
    if pixel_threshold and pixel_threshold > 0:
        defects = localise_defects(result.anomaly_map, pixel_threshold)
    else:
        defects = localise_defects(
            result.anomaly_map, float(np.quantile(result.anomaly_map, 0.995))
        )

    inspection_id = new_id()
    image_path: str | None = None
    heatmap_path: str | None = None
    overlay_path: str | None = None
    image_hash: str | None = None

    if save_artifacts:
        image_path, image_hash = storage.save_original(
            model_row.project_id, inspection_id, image, image_format
        )
        heatmap_path = storage.save_heatmap(
            model_row.project_id, inspection_id, result.anomaly_map
        )
        overlay_path = storage.save_overlay(
            model_row.project_id, inspection_id, image, result.anomaly_map
        )

    row = Inspection(
        id=inspection_id,
        project_id=model_row.project_id,
        model_id=model_row.id,
        model_version=model_row.version,
        config_hash=model_row.config_hash,
        source=source,
        image_path=image_path,
        image_hash=image_hash,
        is_anomaly=result.is_anomaly,
        image_score=result.image_score,
        threshold=result.threshold,
        confidence=result.confidence,
        defect_ratio=result.defect_ratio,
        max_score=result.max_score,
        latency_ms=result.latency_ms,
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
    )
    db.add(row)

    for defect in defects:
        db.add(
            Defect(
                inspection_id=inspection_id,
                x=defect.x,
                y=defect.y,
                width=defect.width,
                height=defect.height,
                area_px=defect.area_px,
                area_ratio=defect.area_ratio,
                peak_score=defect.peak_score,
                mean_score=defect.mean_score,
            )
        )

    db.commit()
    db.refresh(row)
    return row
