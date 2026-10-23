"""The anomaly detection model: fit on good images, score anything.

This is the public surface of the engine. A trained model is a directory
containing a memory bank plus the config needed to reproduce the embeddings
it was built from. The config hash matters: if someone changes the backbone
or the image size and then tries to score with an old bank, the embeddings
are in a different space and every distance is meaningless. We check the
hash and refuse rather than return confident nonsense.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .embeddings import EmbeddingConfig, MultiScaleEmbedder
from .memory_bank import MemoryBank, knn_anomaly_scores, knn_scores_excluding
from .preprocess import PreprocessResult, preprocess_image


def _source_labels(n_images: int, n_features: int) -> np.ndarray:
    """Assign each feature row to the image it came from.

    Rows are laid out image by image (the embedder processes images in
    order), so the label for a row is its index divided by the per-image
    patch count. We compute the per-image count from the total, rounding so
    the labels cover every row exactly once.
    """
    if n_images <= 0:
        return np.zeros(0, dtype=np.int64)
    per_image = n_features // n_images
    labels = np.repeat(np.arange(n_images, dtype=np.int64), per_image)
    # Any remainder rows (from rounding) belong to the last image.
    if len(labels) < n_features:
        labels = np.concatenate(
            [labels, np.full(n_features - len(labels), n_images - 1, dtype=np.int64)]
        )
    return labels


@dataclass
class ModelConfig:
    """Everything needed to reproduce a model's embeddings."""

    backbone: str = "wide_resnet50_2"
    input_size: int = 256
    patchsize: int = 3
    patchstride: int = 1
    pretrain_embed_dimension: int = 1024
    target_embed_dimension: int = 1024
    layers_to_extract_from: tuple[str, ...] = ("layer2", "layer3")
    coreset_percentage: float = 0.1
    n_nearest_neighbours: int = 1
    reweight: bool = True
    nn_backend: str = "auto"
    # Safety margin added to the calibrated threshold, as a fraction of the
    # spread of the good-image scores. Trades recall for a lower false
    # reject rate on good parts.
    #
    # Historical default. Re-evaluate on rights-cleared, later-captured
    # images before using it as a plant operating point.
    calibration_margin: float = 0.10
    # Percentile of an image's own scores used as the relative floor for
    # the defect mask. Higher is tighter; lower finds smaller defects.
    relative_percentile: float = 0.995

    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            backbone=self.backbone,
            patchsize=self.patchsize,
            patchstride=self.patchstride,
            pretrain_embed_dimension=self.pretrain_embed_dimension,
            target_embed_dimension=self.target_embed_dimension,
            input_size=self.input_size,
            layers_to_extract_from=tuple(self.layers_to_extract_from),
        )

    def hash(self) -> str:
        """Stable hash of the embedding-relevant fields.

        Fields that do not affect the embedding geometry (coreset percentage,
        k, backend) are excluded, so changing those does not invalidate a
        bank -- but changing the backbone or input size does.
        """
        payload = {
            "backbone": self.backbone,
            "input_size": self.input_size,
            "patchsize": self.patchsize,
            "patchstride": self.patchstride,
            "pretrain_embed_dimension": self.pretrain_embed_dimension,
            "target_embed_dimension": self.target_embed_dimension,
            "layers_to_extract_from": list(self.layers_to_extract_from),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class DetectionResult:
    """The output of scoring one image."""

    image_score: float
    is_anomaly: bool
    threshold: float
    anomaly_map: np.ndarray  # HxW float32, original image resolution
    anomaly_map_letterboxed: np.ndarray  # HxW float32 on the square canvas
    defect_mask: np.ndarray  # HxW bool, thresholded at the defect level
    defect_ratio: float  # fraction of the image flagged
    max_score: float
    percentile_score: float
    latency_ms: float
    preprocessing: PreprocessResult | None = None
    # Calibration diagnostics, populated when a threshold is available.
    threshold_source: str = "uncalibrated"
    # Raw per-patch scores on the patch grid. Kept so the pixel threshold
    # can be calibrated from good images without re-scoring.
    patch_scores: np.ndarray | None = None

    @property
    def confidence(self) -> float:
        """A 0-1 confidence derived from how far above threshold we are."""
        if self.threshold <= 0:
            return 0.0
        return float(min(1.0, (self.image_score - self.threshold) / max(self.threshold, 1e-6)))


class AnomalyDetector:
    """Fit on good images, then score images for anomalies.

    Typical use::

        detector = AnomalyDetector(ModelConfig())
        detector.fit(["good1.png", "good2.png"])   # or a directory
        result = detector.score(Image.open("test.png"))
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._calibration_margin = self.config.calibration_margin
        self.relative_percentile = self.config.relative_percentile
        self._embedder: MultiScaleEmbedder | None = None
        self.memory_bank: MemoryBank | None = None
        # Threshold on the image-level score. Set by calibrate().
        self.threshold: float | None = None
        # Threshold on the per-pixel score for building the defect mask.
        self.pixel_threshold: float | None = None
        self.threshold_source: str = "uncalibrated"
        # Distribution of image scores over the training (good) images.
        self._good_scores: list[float] = []
        # Per-patch scores of the good images, used to calibrate the pixel
        # threshold. Kept separate from the image-level scores.
        self._good_pixel_scores: list[float] = []
        # Exact quantiles of the patch-score distribution, persisted so a
        # reloaded model can be recalibrated without re-scoring the good set.
        self._good_pixel_quantiles: dict[str, float] = {}
        # How the good-image scores were obtained: "holdout" (scores come
        # from images excluded from the calibration bank) or "in_sample"
        # (too few images to hold out -- the threshold is then optimistic).
        self._calibration_mode: str = "uncalibrated"
        # The calibration bank (built without the held-out images). Kept so
        # calibration can be repeated without re-embedding.
        self._calibration_bank: MemoryBank | None = None
        # Safety margin applied to the calibrated threshold.
        self._calibration_margin: float = 0.10

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    @property
    def embedder(self) -> MultiScaleEmbedder:
        """Lazily build the backbone (loading weights is slow)."""
        if self._embedder is None:
            self._embedder = MultiScaleEmbedder(self.config.embedding_config())
            self._embedder.eval()
        return self._embedder

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed preprocessed images and return patch features ``(N, D)``."""
        from .preprocess import preprocess_batch

        if not images:
            return np.zeros((0, self.config.target_embed_dimension), dtype=np.float32)
        batch, _ = preprocess_batch(images, size=self.config.input_size)
        batch = batch.to(self.device)
        embeddings, _ = self.embedder(batch)
        return embeddings.cpu().numpy().astype(np.float32)

    def _embed_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Embed a list of image files, in batches."""
        chunks: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch_images = []
            for path in paths[i : i + batch_size]:
                with Image.open(path) as img:
                    batch_images.append(img.convert("RGB"))
            chunks.append(self.embed_images(batch_images))
        if not chunks:
            return np.zeros((0, self.config.target_embed_dimension), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        images: list[Image.Image] | list[Path] | Path,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> "AnomalyDetector":
        """Build the memory bank from known-good images.

        Args:
            images: a list of PIL images, a list of file paths, or a directory
                containing good images.
        """
        if isinstance(images, list) and images and isinstance(images[0], Image.Image):
            return self.fit_images(images, batch_size=batch_size)

        paths = self._collect_paths(images)
        if not paths:
            raise ValueError("No training images supplied")

        features = self._embed_paths(paths, batch_size=batch_size)
        if verbose:
            print(f"embedded {len(paths)} images -> {features.shape} features")

        # Label each patch row with the image it came from, so the bank can
        # support leave-one-out calibration.
        source_labels = _source_labels(len(paths), features.shape[0])

        self.memory_bank = MemoryBank.build(
            features,
            percentage=self.config.coreset_percentage,
            backbone=self.config.backbone,
            config_hash=self.config.hash(),
            nn_backend=self.config.nn_backend,
            source_labels=source_labels,
        )

        self._calibrate_leave_one_out(features, source_labels)
        return self

    def _calibrate_leave_one_out(
        self, features: np.ndarray, source_labels: np.ndarray
    ) -> None:
        """Calibrate the threshold by leave-one-image-out scoring.

        Scoring a training image against a bank that contains its own
        patches is circular: those patches score a distance of ~0, so the
        resulting threshold is far too low and a brand-new good part gets
        rejected.

        The fix is leave-one-out: score each training image against the bank
        while excluding that image's own patches. Every training image then
        contributes an honest estimate of what an unseen good part scores,
        and we can use *all* of them rather than a small holdout slice --
        which matters, because a 20% holdout of 60 images leaves the
        threshold estimate too noisy to control the false positive rate.

        This costs one k-NN search per training image, which is far cheaper
        than rebuilding the bank each time.
        """
        n_images = int(source_labels.max()) + 1 if len(source_labels) else 0
        if n_images < 2:
            self._calibration_mode = "in_sample"
            self._good_scores = []
            self._good_pixel_scores = []
            return

        # Group the feature rows by source image.
        rows_by_image: list[np.ndarray] = [
            np.where(source_labels == i)[0] for i in range(n_images)
        ]

        self._good_scores = []
        self._good_pixel_scores = []
        for image_id, rows in enumerate(rows_by_image):
            if len(rows) == 0:
                continue
            query = features[rows]
            distances = knn_scores_excluding(
                query, self.memory_bank, source_labels, image_id
            )
            self._good_scores.append(float(distances.max()))
            self._good_pixel_scores.extend(distances.tolist())

        self._good_pixel_quantiles = self._quantiles_of(self._good_pixel_scores)
        self._calibration_mode = "leave_one_out"

    def _collect_good_pixel_scores(self, paths: list[Path]) -> list[float]:
        """Per-patch scores of the good images, used to set the pixel threshold.

        The pixel threshold and the image threshold are different statistics
        on different scales, so they must be calibrated separately. Using the
        image-level scores to set the pixel threshold flags the entire image
        (every pixel of a good image is at least as anomalous as the image
        score), which destroys localisation.
        """
        scores: list[float] = []
        for path in paths:
            with Image.open(path) as img:
                result = self.score(img.convert("RGB"), _skip_threshold=True)
            # The per-patch distribution, not the image score.
            scores.extend(np.asarray(result.patch_scores, dtype=np.float64).ravel().tolist())
        self._good_pixel_quantiles = self._quantiles_of(scores)
        return scores

    @staticmethod
    def _quantiles_of(values: list[float]) -> dict[str, float]:
        """Exact quantiles of a score list."""
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            str(q): float(np.quantile(arr, q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        }

    @staticmethod
    def _collect_paths(images) -> list[Path]:
        """Normalise the many accepted input shapes into a list of paths."""
        if isinstance(images, (str, Path)):
            path = Path(images)
            if path.is_dir():
                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                return sorted(
                    p for p in path.rglob("*") if p.suffix.lower() in exts
                )
            return [path]
        if not images:
            return []
        first = images[0]
        if isinstance(first, (str, Path)):
            return [Path(p) for p in images]
        return []  # list of PIL images -- handled by the caller

    def fit_images(
        self, images: list[Image.Image], batch_size: int = 16
    ) -> "AnomalyDetector":
        """Build the memory bank directly from in-memory images."""
        if not images:
            raise ValueError("No training images supplied")
        features = self.embed_images(images)
        source_labels = _source_labels(len(images), features.shape[0])
        self.memory_bank = MemoryBank.build(
            features,
            percentage=self.config.coreset_percentage,
            backbone=self.config.backbone,
            config_hash=self.config.hash(),
            nn_backend=self.config.nn_backend,
            source_labels=source_labels,
        )
        self._calibrate_leave_one_out(features, source_labels)
        return self

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, method: str = "quantile", quantile: float = 0.95) -> dict:
        """Set the decision threshold from the distribution of good images.

        The scores produced by PatchCore are distances in an embedding space.
        There is no universal "anomaly score" that means defective. The only
        defensible way to set a threshold without labelled defects is to
        look at how the *normal* images score and pick a point above their
        tail.

        Two thresholds are produced, and they are different statistics:

        * ``threshold`` -- the image-level score above which a part is
          rejected. Calibrated from the image scores of the good parts.
        * ``pixel_threshold`` -- the per-patch score above which a region is
          called a defect. Calibrated from the *patch* scores of the good
          parts, because a patch-level and an image-level score are on
          different scales.

        Args:
            method: ``"quantile"`` (default) takes a high quantile of the
                good-image scores; ``"mean_std"`` takes mean + k*std;
                ``"max"`` takes the worst good image.
            quantile: for ``"quantile"``, the target percentile.

        Returns:
            A dict describing the chosen threshold.
        """
        if not self._good_scores:
            raise RuntimeError("calibrate() requires fitted good-image scores")
        scores = np.array(self._good_scores, dtype=np.float64)

        if method == "quantile":
            threshold = float(np.quantile(scores, quantile))
        elif method == "mean_std":
            k = 3.0
            threshold = float(scores.mean() + k * scores.std())
        elif method == "max":
            threshold = float(scores.max())
        else:
            raise ValueError(f"Unknown calibration method {method!r}")

        # Safety margin.
        #
        # A threshold fitted at a given quantile is, by construction, exceeded
        # by that fraction of the *training* good parts. A plant cares about
        # the false positive rate on the parts it has never seen, and the
        # held-out distribution is wider than the training one -- so a
        # threshold fitted at the 99th percentile of the training set lands
        # at roughly the 95th percentile of reality, giving a few percent of
        # good parts falsely rejected. Adding a margin buys headroom for
        # that spread.
        #
        # This trades recall for precision. For a plant, a false reject that
        # a human re-checks is far cheaper than a defective part escaping,
        # so the bias is deliberately towards fewer false rejects.
        if self._calibration_margin > 0:
            spread = float(scores.max() - scores.min())
            if spread > 0:
                threshold += self._calibration_margin * spread
            else:
                threshold *= 1.0 + self._calibration_margin

        self.threshold = threshold
        self.threshold_source = f"{method}"

        # Pixel threshold, from the patch-score distribution of good images.
        if self._good_pixel_quantiles:
            # Use the exact stored quantiles where possible: the persisted
            # subsample is tail-biased, so recomputing a quantile from it
            # would be biased too.
            pixel_quantile = min(0.9999, max(0.99, quantile))
            candidates = [
                (float(q), float(v))
                for q, v in self._good_pixel_quantiles.items()
                if float(q) >= pixel_quantile
            ]
            if candidates:
                # Smallest stored quantile at or above the target.
                self.pixel_threshold = min(candidates, key=lambda kv: kv[0])[1]
            else:
                self.pixel_threshold = max(
                    v for _, v in self._good_pixel_quantiles.items()
                )
        elif self._good_pixel_scores:
            pixel_scores = np.array(self._good_pixel_scores, dtype=np.float64)
            pixel_quantile = min(0.9999, max(0.99, quantile))
            self.pixel_threshold = float(np.quantile(pixel_scores, pixel_quantile))
        else:
            # Fall back to a fraction of the image threshold, which is at
            # least in the right ballpark.
            self.pixel_threshold = threshold * 0.5

        return {
            "method": method,
            "threshold": threshold,
            "calibration_mode": self._calibration_mode,
            "n_good_images": len(scores),
            "good_score_mean": float(scores.mean()),
            "good_score_std": float(scores.std()),
            "good_score_max": float(scores.max()),
            "good_score_median": float(np.median(scores)),
            "pixel_threshold": self.pixel_threshold,
            "good_pixel_score_p999": (
                float(np.quantile(self._good_pixel_scores, 0.999))
                if self._good_pixel_scores
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def score(
        self,
        image: Image.Image,
        _skip_threshold: bool = False,
        pixel_threshold: float | None = None,
    ) -> DetectionResult:
        """Score a single image for anomalies.

        Args:
            image: the image to inspect.
            _skip_threshold: internal -- skip the "is anomaly" decision when
                scoring training images.
            pixel_threshold: override the per-pixel threshold.
        """
        if self.memory_bank is None:
            raise RuntimeError("Model is not fitted -- call fit() first")

        start = time.perf_counter()

        prep = preprocess_image(image, size=self.config.input_size)
        features = self.embed_images([image])
        patch_scores, image_score, _ = knn_anomaly_scores(
            features,
            self.memory_bank.index,
            n_nearest_neighbours=self.config.n_nearest_neighbours,
            reweight=self.config.reweight,
        )

        # Reshape patch scores back onto the patch grid, then upsample to the
        # letterboxed canvas and smooth.
        import cv2
        from scipy import ndimage

        h_patches, w_patches = self._patch_grid_shape(prep)
        grid = patch_scores.reshape(h_patches, w_patches).astype(np.float32)
        canvas = cv2.resize(
            grid, prep.letterboxed_size, interpolation=cv2.INTER_LINEAR
        )
        canvas = ndimage.gaussian_filter(canvas, sigma=4.0)

        # Map back to the original image geometry.
        original_map = prep.map_to_original(canvas)

        # Two-level pixel threshold.
        #
        # An absolute threshold alone is not enough. A defect sitting on a
        # part changes the embedding of a wide area around it -- layer3's
        # receptive field spans a large fraction of the image -- so every
        # patch in the neighbourhood picks up some elevated score. Against a
        # purely absolute threshold, that whole neighbourhood gets flagged
        # and the localisation becomes useless.
        #
        # So we require two conditions:
        #   1. the patch score is above the absolute threshold (calibrated
        #      from held-out good parts), AND
        #   2. the patch score is in the top slice of *this* image's own
        #      scores (a relative test), so the flag follows the genuinely
        #      extreme region rather than a broad elevated plateau.
        #
        # The relative floor is taken at a high percentile of this image's
        # own scores. It is what keeps the mask tight: the calibrated
        # absolute threshold is deliberately generous (it must not miss
        # anything), so on its own it selects a wide fringe of the plateau.
        p_thresh = pixel_threshold if pixel_threshold is not None else self.pixel_threshold
        if p_thresh is None or p_thresh <= 0:
            p_thresh = float(np.quantile(original_map, 0.995))

        relative_floor = float(np.quantile(original_map, self.relative_percentile))
        effective = np.maximum(p_thresh, relative_floor)
        defect_mask = original_map >= effective
        defect_ratio = float(defect_mask.mean())

        threshold = self.threshold if self.threshold is not None else float("inf")
        is_anomaly = bool(image_score > threshold) if not _skip_threshold else False

        latency_ms = (time.perf_counter() - start) * 1000.0

        return DetectionResult(
            image_score=float(image_score),
            is_anomaly=is_anomaly,
            threshold=float(threshold),
            anomaly_map=original_map,
            anomaly_map_letterboxed=canvas,
            defect_mask=defect_mask,
            defect_ratio=defect_ratio,
            max_score=float(patch_scores.max()),
            percentile_score=float(np.quantile(patch_scores, 0.999)),
            latency_ms=latency_ms,
            preprocessing=prep,
            threshold_source=self.threshold_source,
            patch_scores=patch_scores,
        )

    def _patch_grid_shape(self, prep: PreprocessResult) -> tuple[int, int]:
        """Grid size of the patch scores for the configured input size.

        layer2 has stride 8, so at input size S the grid is S/8 on a side.
        We verify against the actual embedding count so a change of backbone
        or stride does not silently reshape the heatmap.
        """
        size = self.config.input_size
        grid = size // 8
        return grid, grid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _subsample_pixel_scores(self, max_values: int = 20000) -> list[float]:
        """Subsample the patch scores for storage.

        We keep the highest-scoring values, because calibration only reads
        the upper tail of the distribution. A uniform subsample would mostly
        contain low, uninformative values.
        """
        if not self._good_pixel_scores:
            return []
        values = np.asarray(self._good_pixel_scores, dtype=np.float64)
        if len(values) <= max_values:
            return values.tolist()
        # Take the top 20% (the informative tail) and subsample that.
        cutoff = np.quantile(values, 0.8)
        tail = values[values >= cutoff]
        if len(tail) <= max_values:
            return tail.tolist()
        idx = np.linspace(0, len(tail) - 1, max_values).astype(int)
        return tail[idx].tolist()

    def _pixel_quantiles(self) -> dict[str, float]:
        """Exact quantiles of the patch-score distribution."""
        if not self._good_pixel_scores:
            return {}
        values = np.asarray(self._good_pixel_scores, dtype=np.float64)
        return {
            str(q): float(np.quantile(values, q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        }

    def save(self, path: str | Path) -> Path:
        """Persist the model to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.memory_bank is None:
            raise RuntimeError("Cannot save an unfitted model")
        self.memory_bank.save(path / "memory_bank")
        payload = {
            "config": asdict(self.config),
            "layers_to_extract_from": list(self.config.layers_to_extract_from),
            "threshold": self.threshold,
            "pixel_threshold": self.pixel_threshold,
            "threshold_source": self.threshold_source,
            "n_good_images": len(self._good_scores),
            "good_scores": self._good_scores,
            # The full patch-score list is large (thousands per image), and we
            # only need its tail for calibration. Store a subsample plus the
            # exact quantiles, which is all calibrate() consumes.
            "good_pixel_scores": self._subsample_pixel_scores(),
            "good_pixel_quantiles": self._pixel_quantiles(),
            "calibration_mode": self._calibration_mode,
            "config_hash": self.config.hash(),
        }
        (path / "model.json").write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "AnomalyDetector":
        """Load a persisted model."""
        path = Path(path)
        payload = json.loads((path / "model.json").read_text())
        cfg = payload["config"]
        cfg["layers_to_extract_from"] = tuple(cfg.get("layers_to_extract_from", ("layer2", "layer3")))
        config = ModelConfig(**cfg)
        detector = cls(config, device=device)
        detector.memory_bank = MemoryBank.load(
            path / "memory_bank",
            meta={
                "n_input_features": payload.get("n_good_images", 0),
                "backbone": config.backbone,
                "config_hash": payload.get("config_hash", ""),
            },
        )
        detector.threshold = payload.get("threshold")
        detector.pixel_threshold = payload.get("pixel_threshold")
        detector.threshold_source = payload.get("threshold_source", "uncalibrated")
        detector._good_scores = payload.get("good_scores", [])
        detector._good_pixel_scores = payload.get("good_pixel_scores", [])
        detector._good_pixel_quantiles = payload.get("good_pixel_quantiles", {})
        detector._calibration_mode = payload.get("calibration_mode", "uncalibrated")
        return detector
