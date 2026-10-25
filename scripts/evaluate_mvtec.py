"""Evaluate the detector on MVTec AD.

Runs the full protocol used in the PatchCore paper: for each category,
train on the ``train/good`` split, then evaluate on the ``test`` split.
Reports image-level AUROC, pixel-level AUROC and PRO, plus the false
positive rate on good images -- the number a plant actually cares about.

Usage::

    python scripts/evaluate_mvtec.py --data /path/to/mvtec_ad \
        --categories bottle,cable --backbone resnet50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.core.evaluation import (  # noqa: E402
    EvaluationReport,
    auroc,
    aupro,
    f1_at_threshold,
    pixel_auroc,
)
from argus.core.model import AnomalyDetector, ModelConfig  # noqa: E402


def list_categories(data_root: Path) -> list[str]:
    return sorted(
        d.name
        for d in data_root.iterdir()
        if d.is_dir() and (d / "train" / "good").exists()
    )


def collect_test_samples(cat_dir: Path) -> tuple[list[Path], list[Path], dict[Path, Path]]:
    """Read the official test split; fail closed on bad manifests or leakage."""
    train_dir = cat_dir / "train" / "good"
    test_dir = cat_dir / "test"
    manifest = load_manifest(cat_dir)
    good_files: list[Path] = []
    bad_files: list[Path] = []
    mask_paths: dict[Path, Path] = {}
    if manifest:
        for entry in manifest:
            if entry.get("split") not in {"train", "test"}:
                raise ValueError(f"Unknown manifest split: {entry.get('split')}")
            if entry.get("split") != "test":
                continue
            image_path = (cat_dir / entry["image"]).resolve()
            if not image_path.is_relative_to(test_dir.resolve()) or not image_path.is_file():
                raise ValueError(f"Invalid test image in manifest: {image_path}")
            defect = image_path.parent.name
            if defect != entry["defect"]:
                raise ValueError(f"Defect label disagrees with path: {image_path}")
            (good_files if defect == "good" else bad_files).append(image_path)
            if defect != "good" and entry.get("mask"):
                mask = (cat_dir / entry["mask"]).resolve()
                if not mask.is_relative_to((cat_dir / "ground_truth").resolve()) or not mask.is_file():
                    raise ValueError(f"Invalid ground-truth mask: {mask}")
                mask_paths[image_path] = mask
    else:
        for defect_dir in sorted(test_dir.iterdir()):
            if defect_dir.is_dir():
                files = sorted(p.resolve() for p in defect_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
                (good_files if defect_dir.name == "good" else bad_files).extend(files)
        for image_path in bad_files:
            candidate = cat_dir / "ground_truth" / image_path.parent.name / f"{image_path.stem}_mask.png"
            if candidate.is_file():
                mask_paths[image_path] = candidate.resolve()
    if not good_files or not bad_files:
        raise ValueError(f"Need both good and defective test images in {cat_dir}")
    files = good_files + bad_files
    if len(files) != len(set(files)):
        raise ValueError("Duplicate test paths")
    train_hashes = {hashlib.sha256(p.read_bytes()).digest() for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    if any(hashlib.sha256(p.read_bytes()).digest() in train_hashes for p in files):
        raise ValueError("Train/test content overlap detected")
    return good_files, bad_files, mask_paths


def evaluate_category(
    data_root: Path,
    category: str,
    config: ModelConfig,
    max_test: int | None = None,
    dataset_label: str = "MVTec AD",
) -> EvaluationReport:
    """Train on the good split and evaluate on the test split."""
    cat_dir = data_root / category
    train_dir = cat_dir / "train" / "good"
    train_files = sorted(p for p in train_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not train_files:
        raise ValueError(f"No training images for {category}")

    good_files, bad_files, mask_paths = collect_test_samples(cat_dir)
    if max_test:
        good_files = good_files[:max_test]
        bad_files = bad_files[:max_test]
        mask_paths = {p: m for p, m in mask_paths.items() if p in bad_files}

    detector = AnomalyDetector(config)
    start = time.perf_counter()
    detector.fit(train_files)
    fit_seconds = time.perf_counter() - start

    # Calibrate the threshold from the good training images.
    calib = detector.calibrate(method="quantile", quantile=0.95)

    labels: list[bool] = []
    scores: list[float] = []
    maps: list[np.ndarray] = []
    latencies: list[float] = []
    all_files: list[Path] = []

    for path in good_files + bad_files:
        with Image.open(path) as img:
            image = img.convert("RGB")
        result = detector.score(image)
        labels.append(path.parent.name != "good")
        scores.append(result.image_score)
        # Evaluate localisation at the network input resolution. This keeps
        # benchmark memory bounded and states the resolution of the claim.
        import cv2

        maps.append(cv2.resize(result.anomaly_map.astype(np.float32),
                               (config.input_size, config.input_size),
                               interpolation=cv2.INTER_LINEAR))
        latencies.append(result.latency_ms)
        all_files.append(path)

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    # Image-level metrics.
    image_auroc = auroc(labels_arr, scores_arr)

    # Pixel-level metrics need a ground-truth mask per test image.
    pixel_auroc_value = float("nan")
    pro_value = float("nan")
    if len(mask_paths) == len(bad_files):
        masks = load_ground_truth_masks(all_files, mask_paths, maps)
        # The predictions must be the model's anomaly maps -- scoring the
        # ground-truth masks against themselves would always return 1.0.
        pixel_auroc_value = pixel_auroc(labels_arr, masks, maps)
        pro_value = aupro(labels_arr, masks, maps)

    # False positive rate on good images at the calibrated threshold.
    good_mask = ~labels_arr
    if good_mask.any():
        fpr = float((scores_arr[good_mask] > calib["threshold"]).mean())
    else:
        fpr = float("nan")

    precision, recall, f1 = f1_at_threshold(
        labels_arr, scores_arr, calib["threshold"]
    )

    report = EvaluationReport(
        dataset=dataset_label,
        model=f"{category} / {config.backbone}",
        n_train=len(train_files),
        n_test=len(labels_arr),
        n_anomalous=int(labels_arr.sum()),
        metrics={
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc_value,
            "aupro": pro_value,
            "f1_at_threshold": f1,
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "good_image_fpr": fpr,
            "mean_latency_ms": float(np.mean(latencies)),
            "fit_seconds": fit_seconds,
            "memory_bank_size": (
                detector.memory_bank.size if detector.memory_bank else 0
            ),
            "threshold": calib["threshold"],
            "pixel_threshold": calib["pixel_threshold"],
            "metric_map_size": config.input_size,
        },
        good_image_scores={
            "mean": calib["good_score_mean"],
            "std": calib["good_score_std"],
            "max": calib["good_score_max"],
            "median": calib["good_score_median"],
        },
        threshold=calib["threshold"],
        samples=[
            {
                "image": str(path.relative_to(cat_dir.resolve())),
                "label": "anomaly" if label else "good",
                "score": float(score),
                "predicted_anomaly": bool(score > calib["threshold"]),
                "latency_ms": float(latency),
            }
            for path, label, score, latency in zip(all_files, labels, scores, latencies)
        ],
    )

    if len(mask_paths) != len(bad_files):
        report.notes.append(
            f"Pixel metrics unavailable: {len(bad_files) - len(mask_paths)} "
            "defective test images lack ground-truth masks."
        )
    if max_test:
        report.notes.append("Partial test subset (--max-test); not a full-category benchmark.")
    return report


def load_manifest(cat_dir: Path) -> list[dict] | None:
    """Read the per-category manifest linking test images to their masks."""
    manifest_path = cat_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest: {manifest_path}") from exc
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Manifest has no samples list: {manifest_path}")
    return samples


def load_ground_truth_masks(
    files: list[Path],
    mask_paths: dict[Path, Path],
    maps: list[np.ndarray],
) -> list[np.ndarray]:
    """Load the ground-truth mask for each test image, resized to the map."""
    masks: list[np.ndarray] = []
    for path, amap in zip(files, maps):
        mask_path = mask_paths.get(path)
        if mask_path is not None and mask_path.exists():
            with Image.open(mask_path) as mask_img:
                mask = mask_img.convert("L").resize(
                    (amap.shape[1], amap.shape[0]), Image.NEAREST
                )
                masks.append(np.array(mask) > 127)
        else:
            # Good image, or a defective image with no mask available.
            masks.append(np.zeros(amap.shape, dtype=bool))
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="MVTec AD root directory")
    parser.add_argument(
        "--categories", default="all", help="Comma-separated categories, or 'all'"
    )
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--n-nearest-neighbours", type=int, default=1)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--out", default="results/mvtec_eval.json")
    parser.add_argument("--dataset-label", default="MVTec AD",
                        help="Report label when evaluating another dataset in MVTec layout, e.g. VisA")
    args = parser.parse_args()

    data_root = Path(args.data)
    categories = (
        list_categories(data_root)
        if args.categories == "all"
        else [c.strip() for c in args.categories.split(",")]
    )

    config = ModelConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        coreset_percentage=args.coreset_percentage,
        n_nearest_neighbours=args.n_nearest_neighbours,
    )

    reports: list[EvaluationReport] = []
    for category in categories:
        print(f"\n=== {category} ===", flush=True)
        report = evaluate_category(
            data_root, category, config, max_test=args.max_test,
            dataset_label=args.dataset_label,
        )
        reports.append(report)
        print(report.summary(), flush=True)

    # Aggregate.
    summary = {
        "dataset": args.dataset_label,
        "protocol": "train/good only; test/good and test/defects only; duplicate-content check",
        "metric_protocol": "image AUROC; pixel AUROC; region PRO area over measured pixel FPR 0-0.3, divided by 0.3",
        "partial_test": args.max_test is not None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "backbone": args.backbone,
        "input_size": args.input_size,
        "coreset_percentage": args.coreset_percentage,
        "mean_image_auroc": float(
            np.mean([r.metrics["image_auroc"] for r in reports])
        ),
        "mean_good_fpr": float(
            np.nanmean([r.metrics["good_image_fpr"] for r in reports])
        ),
        "mean_latency_ms": float(
            np.mean([r.metrics["mean_latency_ms"] for r in reports])
        ),
        "categories": [r.model for r in reports],
    }
    if all(not np.isnan(r.metrics["aupro"]) for r in reports):
        summary["mean_aupro_0_3"] = float(np.mean([r.metrics["aupro"] for r in reports]))
        summary["mean_pixel_auroc"] = float(
            np.mean([r.metrics["pixel_auroc"] for r in reports])
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"summary": summary, "reports": [r.__dict__ for r in reports]},
            indent=2,
            default=str,
        )
    )

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
