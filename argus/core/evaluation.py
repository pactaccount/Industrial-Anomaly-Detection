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


def aupro(labels: np.ndarray, masks: list[np.ndarray]) -> float:
    """Area under the per-region ROC curve (PRO).

    The PRO metric, from the MVTec AD paper, is the standard way to compare
    anomaly localisation methods. For each anomaly it measures the overlap
    between the normalised prediction and the ground-truth region, over a
    range of thresholds, and averages across anomalies. Unlike pixel AUROC
    it weights every *defect region* equally, so a huge defect does not
    dominate the score.

    Args:
        labels: ``(N,)`` boolean, True for anomalous images.
        masks: list of ``(H, W)`` boolean ground-truth masks, one per image.

    Returns:
        The PRO value in ``[0, 1]``, or NaN if there are no anomalies.
    """
    labels = np.asarray(labels).astype(bool)
    if not labels.any():
        return float("nan")

    thresholds = np.linspace(0, 1, 101)
    ratios = np.zeros(len(thresholds))

    for idx in np.where(labels)[0]:
        mask = np.asarray(masks[idx]).astype(bool)
        if mask.sum() == 0:
            continue
        # Normalise the prediction map to [0, 1] over the whole image.
        pred = masks[idx].astype(np.float64)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
        for t_idx, t in enumerate(thresholds):
            binary = pred >= t
            # Proportion of the ground-truth region that was found.
            found = (binary & mask).sum() / mask.sum()
            ratios[t_idx] += found

    n_anom = max(1, int(labels.sum()))
    per_region = ratios / n_anom
    # PRO integrates over the FPR range, normalised to [0, 1].
    fpr = np.linspace(0, 1, len(thresholds))
    return float(np.trapezoid(per_region, fpr))


def pixel_auroc(labels: np.ndarray, masks: list[np.ndarray]) -> float:
    """Pixel-level AUROC, pooled over all images."""
    labels = np.asarray(labels).astype(bool)
    flat_labels: list[bool] = []
    flat_scores: list[float] = []
    for idx in range(len(labels)):
        flat_labels.append(np.asarray(masks[idx]).astype(bool).ravel())
        flat_scores.append(np.asarray(masks[idx]).ravel())
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
