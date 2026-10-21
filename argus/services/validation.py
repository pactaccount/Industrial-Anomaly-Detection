"""Validation: measure how a model behaves on data it has never seen.

The research is unambiguous on this point. Before shipping, a plant needs to
know two numbers:

1. **What fraction of good parts will be falsely rejected?** This is
   measurable with zero labels -- you just need held-out good parts
   captured under current production conditions. A model with 99% AUROC but
   a 5% false reject rate is unusable, because it will scrap good product.
2. **Will the model catch defects of the size that matter?** Without
   labelled defects you cannot answer this with real data, but you *can*
   inject synthetic defects of controlled size and severity and check the
   model finds them. This is a lower bound on real-defect performance
   (published results put it roughly 5 points below real defects), but it
   is a real, deployable check.

This module implements both, plus a shift-robustness check: re-score the
held-out parts under simulated lighting and capture perturbations, because
lighting and exposure changes are the dominant real-world failure mode and
are worth measuring explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.evaluation import EvaluationReport, make_synthetic_anomaly


@dataclass
class ValidationResult:
    """The outcome of a validation run."""

    n_good: int = 0
    n_synthetic: int = 0
    # Fraction of good parts the model flagged.
    false_reject_rate: float = 0.0
    # Fraction of synthetic defects the model found.
    synthetic_detection_rate: float = 0.0
    # Breakdown of detection by injected defect size.
    detection_by_size: dict[str, float] = field(default_factory=dict)
    # Breakdown of detection by defect type.
    detection_by_type: dict[str, float] = field(default_factory=dict)
    # Good-part score statistics.
    good_score_mean: float = 0.0
    good_score_std: float = 0.0
    good_score_p95: float = 0.0
    good_score_max: float = 0.0
    threshold: float = 0.0
    # Degradation under simulated capture shifts.
    shift_robustness: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        lines = [
            f"Held-out good parts:      {self.n_good}",
            f"False reject rate:        {self.false_reject_rate * 100:.2f}%",
            f"Synthetic detection rate: {self.synthetic_detection_rate * 100:.2f}%"
            f"  ({self.n_synthetic} injected defects)",
            "",
            f"Threshold:                {self.threshold:.3f}",
            f"Good score mean ± std:    {self.good_score_mean:.3f} ± {self.good_score_std:.3f}",
            f"Good score p95 / max:     {self.good_score_p95:.3f} / {self.good_score_max:.3f}",
        ]
        if self.detection_by_size:
            lines += ["", "Detection rate by defect size:"]
            for size, rate in sorted(self.detection_by_size.items(), key=lambda kv: float(kv[0])):
                lines.append(f"  {size:<16} {rate * 100:6.1f}%")
        if self.detection_by_type:
            lines += ["", "Detection rate by defect type:"]
            for kind, rate in sorted(self.detection_by_type.items()):
                lines.append(f"  {kind:<16} {rate * 100:6.1f}%")
        if self.shift_robustness:
            lines += ["", "Robustness to capture shift (good-part false reject rate):"]
            for shift, stats in self.shift_robustness.items():
                lines.append(
                    f"  {shift:<16} {stats['false_reject_rate'] * 100:6.2f}%"
                    f"  (mean score {stats['mean_score']:.3f})"
                )
        if self.notes:
            lines += ["", "Notes:"]
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _perturb_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Scale the pixel values of an image by a factor."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    out = np.clip(array * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _perturb_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Scale the contrast of an image around its mean."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    mean = array.mean()
    out = np.clip((array - mean) * factor + mean, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def validate(
    detector,
    good_images: list[Image.Image],
    threshold: float | None = None,
    inject_synthetic: bool = True,
    synthetic_per_image: int = 1,
    severities: tuple[float, ...] = (0.3, 0.5, 0.7),
    defect_kinds: tuple[str, ...] = ("scratch", "blob", "hole", "shift"),
    test_shift: bool = True,
    seed: int = 0,
) -> ValidationResult:
    """Run a validation suite against a fitted detector.

    Args:
        detector: a fitted :class:`AnomalyDetector`.
        good_images: held-out good parts, captured under production
            conditions. These must NOT be in the model's training set.
        threshold: the operating threshold; defaults to the model's own.
        inject_synthetic: also test detection of injected defects.
        synthetic_per_image: how many synthetic defects to inject per image.
        severities: defect severities to test.
        defect_kinds: defect families to test.
        test_shift: also re-score under simulated lighting/capture changes.
        seed: seed for reproducible defect injection.
    """
    rng = np.random.default_rng(seed)
    result = ValidationResult()
    if not good_images:
        result.notes.append("No held-out images supplied.")
        return result

    threshold = threshold if threshold is not None else detector.threshold
    result.threshold = float(threshold or 0.0)

    # ---- Slice A: good parts only (no labels needed) -----------------
    good_scores: list[float] = []
    false_rejects = 0
    for image in good_images:
        score = detector.score(image, _skip_threshold=True).image_score
        good_scores.append(score)
        if threshold is not None and score > threshold:
            false_rejects += 1

    result.n_good = len(good_scores)
    result.false_reject_rate = false_rejects / len(good_scores)
    scores_arr = np.array(good_scores)
    result.good_score_mean = float(scores_arr.mean())
    result.good_score_std = float(scores_arr.std())
    result.good_score_p95 = float(np.percentile(scores_arr, 95))
    result.good_score_max = float(scores_arr.max())

    if threshold is not None and result.false_reject_rate > 0.05:
        result.notes.append(
            f"False reject rate is {result.false_reject_rate * 100:.1f}%, above the "
            "5% a line can usually tolerate. Consider recalibrating the threshold, "
            "adding more held-out good parts, or capturing under tighter lighting."
        )

    # ---- Slice C: synthetic defects ----------------------------------
    if inject_synthetic:
        by_size: dict[str, list[int]] = {}
        by_type: dict[str, list[int]] = {}
        detected = 0
        total = 0

        for image in good_images:
            base = np.asarray(image.convert("RGB"))
            for _ in range(synthetic_per_image):
                severity = float(rng.choice(severities))
                kind = str(rng.choice(list(defect_kinds)))
                corrupted, mask = make_synthetic_anomaly(
                    base, rng, severity=severity, kind=kind
                )
                score = detector.score(Image.fromarray(corrupted)).image_score
                hit = bool(threshold is not None and score > threshold)
                detected += int(hit)
                total += 1

                size_bucket = f"{int(severity * 100)}%"
                by_size.setdefault(size_bucket, []).append(int(hit))
                by_type.setdefault(kind, []).append(int(hit))

        result.n_synthetic = total
        result.synthetic_detection_rate = detected / total if total else 0.0
        result.detection_by_size = {
            k: sum(v) / len(v) for k, v in by_size.items()
        }
        result.detection_by_type = {
            k: sum(v) / len(v) for k, v in by_type.items()
        }
        result.notes.append(
            "Synthetic-defect detection is a lower bound on real-defect "
            "performance, not a substitute for it. Collect real defective "
            "parts from the reject station to measure the real rate."
        )

    # ---- Slice D: capture-shift robustness ---------------------------
    if test_shift:
        shifts = {
            "brighter ×1.2": lambda im: _perturb_brightness(im, 1.2),
            "dimmer ×0.8": lambda im: _perturb_brightness(im, 0.8),
            "contrast ×0.8": lambda im: _perturb_contrast(im, 0.8),
            "contrast ×1.2": lambda im: _perturb_contrast(im, 1.2),
        }
        for name, fn in shifts.items():
            shifted_scores = [
                detector.score(fn(im), _skip_threshold=True).image_score
                for im in good_images
            ]
            arr = np.array(shifted_scores)
            fr = (
                float((arr > threshold).mean())
                if threshold is not None
                else float("nan")
            )
            result.shift_robustness[name] = {
                "false_reject_rate": fr,
                "mean_score": float(arr.mean()),
                "max_score": float(arr.max()),
            }
        result.notes.append(
            "Capture-shift rows show what happens to the false reject rate if "
            "lighting or camera settings drift. Large increases mean the "
            "model needs recalibrating or tighter capture control."
        )

    return result


def validate_from_directory(
    detector,
    directory: str | Path,
    **kwargs,
) -> ValidationResult:
    """Validate using every image in a directory of held-out good parts."""
    directory = Path(directory)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No images found in {directory}")
    images = []
    for path in files:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return validate(detector, images, **kwargs)
