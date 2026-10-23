"""Evaluation harness for anomaly detection models.

Anomaly detection is unusual: the whole selling point is that you do *not*
have labelled defect data. So how do you know if a model works?

The honest answer has three parts, and this module implements all three:

1. **Image-level ranking metrics** (AUROC, AUPRO) need *some* labelled
   defects to be meaningful. We use MVTec AD's public test set for this --
   it is the community benchmark, and the numbers are comparable to
   published results. We are explicit that this is a benchmark, not a claim
   about a customer's line.

2. **Synthetic anomalies** let you test on *your own* data without labels.
   We generate defects of controlled size and severity (CutPaste, Perlin
   noise, blur, colour shift) and measure whether the model localises them.
   This is a real, deployable validation strategy: a plant can hold out a
   few images, inject synthetic defects, and confirm the line will catch
   defects of the size that matter.

3. **Good-image false positive rate** is the metric a plant actually cares
   about, and it is measurable with zero labels: what fraction of known-good
   parts does the system reject? A model with 99% AUROC but a 5% FPR on good
   parts is a production disaster, because it will scrap good product.

We report all three.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank method (ties handled)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks for ties, so equal scores get equal credit.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    sum_pos_ranks = ranks[labels].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupro(
    labels: np.ndarray,
    masks: list[np.ndarray],
    predictions: list[np.ndarray],
    max_fpr: float = 0.3,
) -> float:
    """PRO area up to *measured* pixel FPR, normalised by ``max_fpr``.

    Each connected defect region has equal weight. Background pixels from
    both good and defective test images determine the false-positive axis.
    This follows the MVTec AD PRO convention; no mask is ever used as its own
    prediction. A histogram of score ranks gives exact curve changes for
    typical 8-bit test maps without repeatedly thresholding every image.
    """
    from scipy import ndimage

    if len(labels) != len(masks) or len(labels) != len(predictions):
        raise ValueError("labels, masks and predictions must align")
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if not predictions:
        return float("nan")
    region_scores: list[np.ndarray] = []
    background_scores: list[np.ndarray] = []
    for label, mask, scores in zip(labels, masks, predictions):
        mask = np.asarray(mask, dtype=bool)
        scores = np.asarray(scores, dtype=np.float64)
        if mask.shape != scores.shape or not np.isfinite(scores).all():
            raise ValueError("mask and finite anomaly map must have matching shapes")
        if bool(label) != bool(mask.any()):
            raise ValueError("image label and ground-truth mask disagree")
        background_scores.append(scores[~mask].ravel())
        if label:
            regions, count = ndimage.label(mask)
            for region_id in range(1, count + 1):
                region_scores.append(scores[regions == region_id].ravel())
    if not region_scores:
        return float("nan")

    background = np.concatenate(background_scores)
    if not len(background):
        return float("nan")
    # Relevant thresholds are the background score quantiles up to max_fpr.
    # Include all distinct scores for small tests and a fixed grid for large
    # benchmarks to keep memory and runtime bounded.
    unique = np.unique(background)
    if len(unique) <= 2048:
        thresholds = unique
    else:
        thresholds = np.unique(np.quantile(background, np.linspace(1 - max_fpr, 1, 2048)))
    thresholds = np.r_[np.inf, thresholds[::-1], -np.inf]
    bg_sorted = np.sort(background)
    fprs = np.array([(len(background) - np.searchsorted(bg_sorted, t, side="right")) / len(background) for t in thresholds])
    pros = np.zeros(len(thresholds), dtype=np.float64)
    for region in region_scores:
        ordered = np.sort(region)
        pros += (len(region) - np.searchsorted(ordered, thresholds, side="right")) / len(region)
    pros /= len(region_scores)
    # Duplicated FPR coordinates can have different PRO; keep the best
    # attainable overlap at that operating point.
    order = np.argsort(fprs, kind="mergesort")
    fprs, pros = fprs[order], pros[order]
    distinct = np.unique(fprs)
    best = np.array([pros[fprs == x].max() for x in distinct])
    # A tied score can jump directly from FPR 0 to FPR 1. Interpolating
    # across that unattainable interval would award spurious localisation.
    # Hold the last attainable PRO until the next threshold change.
    inside = distinct < max_fpr
    x = np.r_[distinct[inside], max_fpr]
    y = best[inside]
    return float(np.sum(np.diff(x) * y) / max_fpr)


def pixel_auroc(
    labels: np.ndarray,
    masks: list[np.ndarray],
    predictions: list[np.ndarray],
) -> float:
    """Pixel-level AUROC, pooled over all images.

    Args:
        labels: ``(N,)`` boolean, True for anomalous images.
        masks: list of ``(H, W)`` boolean ground-truth masks.
        predictions: list of ``(H, W)`` float anomaly maps.

    Returns:
        The pixel-level AUROC, or NaN if there is nothing to score.
    """
    labels = np.asarray(labels).astype(bool)
    if len(labels) != len(masks) or len(labels) != len(predictions):
        raise ValueError("labels, masks and predictions must align")
    flat_labels: list[bool] = []
    flat_scores: list[float] = []
    for idx in range(len(labels)):
        mask = np.asarray(masks[idx], dtype=bool)
        score = np.asarray(predictions[idx], dtype=np.float64)
        if mask.shape != score.shape or not np.isfinite(score).all():
            raise ValueError("mask and finite anomaly map must have matching shapes")
        if labels[idx] != mask.any():
            raise ValueError("image label and ground-truth mask disagree")
        flat_labels.append(mask.ravel())
        flat_scores.append(score.ravel())
    if not flat_scores:
        return float("nan")
    return auroc(np.concatenate(flat_labels), np.concatenate(flat_scores))


def f1_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """Precision, recall and F1 at a fixed score threshold."""
    predictions = np.asarray(scores) > threshold
    labels = np.asarray(labels).astype(bool)
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ----------------------------------------------------------------------
# Synthetic anomalies
# ----------------------------------------------------------------------
def make_synthetic_anomaly(
    image: np.ndarray,
    rng: np.random.Generator,
    severity: float = 0.5,
    kind: str = "mixed",
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a synthetic defect into an image.

    These stand in for real defects when no labelled defect data exists.
    The four families mirror the categories of defect that actually show up
    on a production line:

    * ``scratch`` -- a thin, elongated, high-contrast line (surface damage)
    * ``blob`` -- a soft, discoloured patch (contamination, oil, stain)
    * ``hole`` -- a sharp dark region (missing material, chip)
    * ``shift`` -- a colour shift over a region (misprint, wrong finish)

    Args:
        image: ``(H, W, 3)`` uint8 image.
        rng: numpy random generator.
        severity: 0..1, controls defect size and contrast.
        kind: which defect family, or ``"mixed"`` to pick one at random.

    Returns:
        ``(corrupted_image, mask)`` where mask is ``(H, W)`` bool.
    """
    h, w = image.shape[:2]
    corrupted = image.copy()
    mask = np.zeros((h, w), dtype=bool)

    if kind == "mixed":
        kind = rng.choice(["scratch", "blob", "hole", "shift"])

    area = (0.005 + 0.05 * severity) * h * w
    side = int(np.sqrt(area)) + 2

    if kind == "scratch":
        length = int(side * rng.uniform(2.0, 4.0))
        thickness = max(1, int(1 + severity * 3))
        y = int(rng.integers(0, max(1, h - thickness)))
        x = int(rng.integers(0, max(1, w - length)))
        y2 = int(np.clip(y + rng.integers(-side // 2, side // 2 + 1), 0, h - 1))
        # Draw a line from (x, y) to (x+length, y2).
        for t in np.linspace(0, 1, length):
            yy = int(round(y + (y2 - y) * t))
            xx = int(round(x + length * t))
            yy0 = max(0, yy - thickness // 2)
            yy1 = min(h, yy + thickness // 2 + 1)
            xx0 = max(0, xx - thickness // 2)
            xx1 = min(w, xx + thickness // 2 + 1)
            patch = corrupted[yy0:yy1, xx0:xx1].astype(np.float32)
            delta = rng.choice([-1, 1]) * (60 + 120 * severity)
            corrupted[yy0:yy1, xx0:xx1] = np.clip(patch + delta, 0, 255).astype(np.uint8)
            mask[yy0:yy1, xx0:xx1] = True

    elif kind == "blob":
        cy = int(rng.integers(side, h - side)) if h > side * 2 else h // 2
        cx = int(rng.integers(side, w - side)) if w > side * 2 else w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = side / 2.0
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        delta = rng.choice([-1, 1]) * (30 + 70 * severity)
        patch = corrupted.astype(np.float32)
        patch[blob] = np.clip(patch[blob] + delta, 0, 255)
        corrupted = patch.astype(np.uint8)
        mask[blob] = True

    elif kind == "hole":
        cy = int(rng.integers(side, h - side)) if h > side * 2 else h // 2
        cx = int(rng.integers(side, w - side)) if w > side * 2 else w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = side / 2.0
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        patch = corrupted.astype(np.float32)
        patch[blob] = np.clip(patch[blob] * 0.25, 0, 255)
        corrupted = patch.astype(np.uint8)
        mask[blob] = True

    else:  # shift
        cy = int(rng.integers(side, h - side)) if h > side * 2 else h // 2
        cx = int(rng.integers(side, w - side)) if w > side * 2 else w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = side / 2.0
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        delta = rng.uniform(-60, 60, size=3) * (0.5 + severity)
        patch = corrupted.astype(np.float32)
        patch[blob] = np.clip(patch[blob] + delta, 0, 255)
        corrupted = patch.astype(np.uint8)
        mask[blob] = True

    return corrupted, mask


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """Results of an evaluation run."""

    dataset: str = ""
    model: str = ""
    n_train: int = 0
    n_test: int = 0
    n_anomalous: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    good_image_scores: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None
    notes: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2))
        return path

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.dataset}",
            f"Model:   {self.model}",
            f"Train:   {self.n_train} good images",
            f"Test:    {self.n_test} images ({self.n_anomalous} anomalous)",
            "",
            f"{'Metric':<28} {'Value':>10}",
            "-" * 40,
        ]
        for key, value in self.metrics.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                lines.append(f"{key:<28} {'n/a':>10}")
            else:
                lines.append(f"{key:<28} {value:>10.4f}")
        if self.threshold is not None:
            lines += ["", f"Threshold: {self.threshold:.4f}"]
        if self.good_image_scores:
            lines += ["", "Good-image score distribution:"]
            for key, value in self.good_image_scores.items():
                lines.append(f"  {key:<26} {value:>10.4f}")
        if self.notes:
            lines += ["", "Notes:"]
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)
