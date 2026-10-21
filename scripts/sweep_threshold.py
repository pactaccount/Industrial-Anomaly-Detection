"""Sweep the calibration margin and measure the operating-point tradeoff.

The calibrated threshold is a business decision, not a technical one: it
trades recall (catching defects) against the false reject rate (scrapping
good parts). Both matter, and the right balance depends on the cost of a
missed defect versus the cost of a false reject.

This script trains once per category, then evaluates the *same* scores
against a range of thresholds, so the tradeoff is measured rather than
guessed.

Usage::

    python scripts/sweep_threshold.py --data /path/to/mvtec_ad \
        --categories bottle,grid,pill --backbone resnet18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.core.evaluation import auroc  # noqa: E402
from argus.core.model import AnomalyDetector, ModelConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--categories", default="bottle,grid,pill,screw")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument(
        "--quantiles",
        default="0.95,0.99,0.995",
        help="Quantiles of the good-score distribution to try",
    )
    parser.add_argument(
        "--margins",
        default="0.0,0.05,0.10,0.25,0.50",
        help="Safety margins, as a fraction of the good-score spread",
    )
    parser.add_argument("--out", default="results/threshold_sweep.json")
    args = parser.parse_args()

    data_root = Path(args.data)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    margins = [float(m) for m in args.margins.split(",")]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    config = ModelConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        coreset_percentage=args.coreset_percentage,
    )

    all_results = []

    for category in categories:
        cat_dir = data_root / category
        train_files = sorted((cat_dir / "train" / "good").glob("*.png"))
        if not train_files:
            print(f"! {category}: no training images")
            continue

        detector = AnomalyDetector(config)
        detector.fit(train_files)

        # Score the test set once.
        labels: list[bool] = []
        scores: list[float] = []
        for defect_dir in sorted((cat_dir / "test").iterdir()):
            if not defect_dir.is_dir():
                continue
            for path in sorted(defect_dir.glob("*.png")):
                with Image.open(path) as img:
                    result = detector.score(img.convert("RGB"), _skip_threshold=True)
                labels.append(defect_dir.name != "good")
                scores.append(result.image_score)

        labels_arr = np.array(labels)
        scores_arr = np.array(scores)
        good = scores_arr[~labels_arr]
        bad = scores_arr[labels_arr]

        print(f"\n=== {category} ===")
        print(f"  good: n={len(good)} mean={good.mean():.3f} p95={np.quantile(good,0.95):.3f} max={good.max():.3f}")
        print(f"  bad:  n={len(bad)} mean={bad.mean():.3f} p50={np.quantile(bad,0.5):.3f}")
        print(f"  AUROC: {auroc(labels_arr, scores_arr):.4f}")

        rows = []
        for quantile in quantiles:
            base = float(np.quantile(good, quantile))
            spread = float(good.max() - good.min())
            for margin in margins:
                threshold = base + margin * spread
                predicted = scores_arr > threshold
                tp = int((predicted & labels_arr).sum())
                fp = int((predicted & ~labels_arr).sum())
                fn = int((~predicted & labels_arr).sum())
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )
                fpr = float((scores_arr[~labels_arr] > threshold).mean())
                rows.append(
                    {
                        "quantile": quantile,
                        "margin": margin,
                        "threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "fpr": fpr,
                    }
                )

        rows.sort(key=lambda r: -r["f1"])
        print(f"  {'quantile':>9}{'margin':>8}{'threshold':>11}{'prec':>8}{'recall':>8}{'F1':>8}{'FPR':>8}")
        for r in rows:
            print(
                f"  {r['quantile']:>9.3f}{r['margin']:>8.2f}{r['threshold']:>11.3f}"
                f"{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}{r['fpr']:>8.4f}"
            )

        all_results.append({"category": category, "rows": rows})

    # Aggregate: mean F1 and mean FPR at each (quantile, margin) setting.
    if all_results:
        print("\n=== AGGREGATE (mean over categories) ===")
        print(f"  {'quantile':>9}{'margin':>8}{'F1':>8}{'FPR':>8}{'precision':>11}{'recall':>9}")
        combos = {}
        for entry in all_results:
            for r in entry["rows"]:
                key = (r["quantile"], r["margin"])
                combos.setdefault(key, []).append(r)
        summary = []
        for (quantile, margin), rs in sorted(combos.items()):
            m = {
                "quantile": quantile,
                "margin": margin,
                "f1": float(np.mean([r["f1"] for r in rs])),
                "fpr": float(np.mean([r["fpr"] for r in rs])),
                "precision": float(np.mean([r["precision"] for r in rs])),
                "recall": float(np.mean([r["recall"] for r in rs])),
            }
            summary.append(m)
            print(
                f"  {quantile:>9.3f}{margin:>8.2f}{m['f1']:>8.4f}{m['fpr']:>8.4f}"
                f"{m['precision']:>11.4f}{m['recall']:>9.4f}"
            )

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"per_category": all_results, "summary": summary}, indent=2)
        )
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
