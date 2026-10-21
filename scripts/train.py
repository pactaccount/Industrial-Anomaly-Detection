"""Train an anomaly detection model from a folder of good images.

Usage::

    python scripts/train.py --images data/good_parts/ \
        --name "Gear housing" --part-number GH-2200 --out data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.core.model import AnomalyDetector, ModelConfig  # noqa: E402
from argus.core.evaluation import EvaluationReport  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Folder of known-good images")
    parser.add_argument("--name", default="model")
    parser.add_argument("--part-number", default="")
    parser.add_argument("--line", default="")
    parser.add_argument("--out", default="./data", help="Output directory for artefacts")
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        choices=["wide_resnet50_2", "resnet50", "resnet18"],
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--coreset-percentage",
        type=float,
        default=0.1,
        help="Fraction of patch features to keep in the memory bank",
    )
    parser.add_argument(
        "--calibration",
        default="quantile",
        choices=["quantile", "mean_std", "max"],
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
        help="Safety margin added to the threshold, as a fraction of the "
        "good-score spread. Trades recall for a lower false reject rate.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise SystemExit(f"Not a directory: {images_dir}")

    config = ModelConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        coreset_percentage=args.coreset_percentage,
        calibration_margin=args.margin,
    )

    detector = AnomalyDetector(config)
    print(f"Training on {images_dir} ...")
    detector.fit(images_dir, verbose=True)
    info = detector.calibrate(method=args.calibration, quantile=args.quantile)

    out = Path(args.out) / "models" / "local" / args.name.replace(" ", "_")
    detector.save(out)

    report = EvaluationReport(
        dataset="local",
        model=f"{args.name} / {args.backbone}",
        n_train=info["n_good_images"],
        n_test=0,
        n_anomalous=0,
        metrics={
            "threshold": info["threshold"],
            "pixel_threshold": info["pixel_threshold"],
            "calibration_mode": info["calibration_mode"],
            "memory_bank_size": detector.memory_bank.size,
            "compression_ratio": detector.memory_bank.compression_ratio,
        },
        good_image_scores={
            "mean": info["good_score_mean"],
            "std": info["good_score_std"],
            "max": info["good_score_max"],
            "median": info["good_score_median"],
        },
        threshold=info["threshold"],
    )

    print()
    print(report.summary())
    print()
    print(f"Model written to {out}")
    (out / "training_report.json").write_text(json.dumps(report.__dict__, indent=2, default=str))

    if info["calibration_mode"] == "in_sample":
        print(
            "\nWARNING: fewer than 2 training images, so the threshold was "
            "fitted in-sample and is optimistic. Train on more images, or "
            "validate against held-out good parts before trusting it."
        )


if __name__ == "__main__":
    main()
