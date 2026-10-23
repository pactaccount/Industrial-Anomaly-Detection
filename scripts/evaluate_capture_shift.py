"""Measure fixed-threshold sensitivity to synthetic brightness changes.

This is a stress check on held-out test images, not an MVTec AD2 benchmark
or evidence of robustness to real camera and illumination changes.

Usage: python scripts/evaluate_capture_shift.py --data data/mvtec_ad \
  --model data/demo_instance/models/<project>/<model>/v1 --out results/shift.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.core.model import AnomalyDetector  # noqa: E402
from scripts.evaluate_mvtec import collect_test_samples  # noqa: E402


def evaluate(data: Path, model: Path) -> dict:
    good, bad, _ = collect_test_samples(data / "metal_nut")
    detector = AnomalyDetector.load(model)
    rows = []
    for factor in (0.8, 1.0, 1.2):
        scores_good, scores_bad = [], []
        for path in good + bad:
            with Image.open(path) as image:
                changed = ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)
            score = detector.score(changed).image_score
            (scores_good if path in good else scores_bad).append(score)
        false_rejects = sum(score > detector.threshold for score in scores_good)
        detected = sum(score > detector.threshold for score in scores_bad)
        rows.append({
            "brightness_factor": factor,
            "good_n": len(scores_good),
            "false_rejects": false_rejects,
            "false_reject_rate": false_rejects / len(scores_good),
            "defect_n": len(scores_bad),
            "detected_defects": detected,
            "defect_recall": detected / len(scores_bad),
        })
    return {
        "dataset": "MVTec AD metal_nut test split",
        "intervention": "PIL brightness scaling of RGB pixels at 0.8, 1.0, 1.2",
        "threshold": detector.threshold,
        "model": str(model),
        "interpretation": "synthetic sensitivity check only; not real lighting or camera validation",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", default="results/metal_nut_shift.json", type=Path)
    args = parser.parse_args()
    report = evaluate(args.data, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["rows"], indent=2))


if __name__ == "__main__":
    main()
