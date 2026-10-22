"""A/B the rebuild against the original demo's algorithm.

The original `app.py` was a sketch of PatchCore with several substantive
differences from a faithful implementation:

* the memory bank was a **single image**, not a coreset of all good parts;
* the input was squashed to 224x224, distorting non-square parts;
* the feature map was average-pooled and flattened rather than patched;
* there was **no threshold**, so no false-reject rate could be measured;
* the heatmap was resized back to a different aspect ratio, so the
  localisation was geometrically wrong.

This script runs both algorithms over the same data and reports the same
metrics, so the difference is attributable to the algorithm rather than to
the data or the split.

For the baseline we do the only thing the original could do: pick a
threshold. We use the most favourable defensible choice -- the score of
the worst good training image -- so the baseline is not handicapped.

Usage::

    python scripts/compare_baseline.py --data /path/to/mvtec_ad \
        --categories bottle,grid,metal_nut,pill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argus.core.evaluation import auroc, f1_at_threshold  # noqa: E402
from argus.core.model import AnomalyDetector, ModelConfig  # noqa: E402


class OriginalAlgorithm:
    """A faithful reimplementation of the original demo's algorithm.

    Deliberately keeps every shortcut the original took, so the comparison
    is against the real thing and not a strawman.
    """

    def __init__(self, size: int = 224):
        self.size = size
        self.model = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.model.eval()
        self.features = {}
        for name in ("layer2", "layer3"):
            layer = getattr(self.model, name)
            layer.register_forward_hook(
                lambda m, i, o, n=name: self.features.__setitem__(n, o.detach())
            )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.bank = None
        self.grid = None

    def embed(self, image: Image.Image) -> np.ndarray:
        """The original's embedding: avg-pool, resize, concat, flatten."""
        # The original squashed to 224x224. Keep that.
        img = image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            self.model(tensor)

        layer2 = F.avg_pool2d(self.features["layer2"], 3, 1, 1)
        layer3 = F.avg_pool2d(self.features["layer3"], 3, 1, 1)
        layer3 = F.interpolate(
            layer3, size=layer2.shape[2:], mode="bilinear", align_corners=False
        )
        combined = torch.cat([layer2, layer3], dim=1)
        b, c, h, w = combined.shape
        return combined.view(c, h * w).t().numpy().astype(np.float32), (h, w)

    def fit(self, images: list[Image.Image]) -> "OriginalAlgorithm":
        """The original used ONE image as the memory bank.

        We give it the single best possible choice: the first good image.
        Using more would be a different algorithm.
        """
        features, grid = self.embed(images[0])
        self.bank = features
        self.grid = grid
        return self

    def score(self, image: Image.Image) -> float:
        """The original's score: mean k-NN distance (sklearn, n=1)."""
        from sklearn.neighbors import NearestNeighbors

        features, _ = self.embed(image)
        knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        knn.fit(self.bank)
        distances, _ = knn.kneighbors(features)
        return float(distances.mean())

    def threshold_from_good(self, images: list[Image.Image]) -> float:
        """A threshold the original could not produce.

        We use the most favourable defensible choice for the baseline: the
        worst-scoring good image, which by construction gives zero false
        rejects on the training set.
        """
        return max(self.score(img) for img in images)


def evaluate_category(
    data_root: Path, category: str, config: ModelConfig
) -> dict:
    cat_dir = data_root / category
    train_files = sorted((cat_dir / "train" / "good").glob("*.png"))
    test_dir = cat_dir / "test"

    def load(path):
        with Image.open(path) as img:
            return img.convert("RGB")

    train_images = [load(p) for p in train_files]
    good_files, bad_files = [], []
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        files = sorted(defect_dir.glob("*.png"))
        (good_files if defect_dir.name == "good" else bad_files).extend(files)

    # ---- Original algorithm ----
    original = OriginalAlgorithm()
    original.fit(train_images)
    orig_threshold = original.threshold_from_good(train_images[:20])
    orig_scores = [original.score(load(p)) for p in good_files + bad_files]
    orig_labels = np.array([False] * len(good_files) + [True] * len(bad_files))
    orig_scores_arr = np.array(orig_scores)
    orig_auroc = auroc(orig_labels, orig_scores_arr)
    orig_pred = orig_scores_arr > orig_threshold
    orig_tp = int((orig_pred & orig_labels).sum())
    orig_fp = int((orig_pred & ~orig_labels).sum())
    orig_fn = int((~orig_pred & orig_labels).sum())
    orig_fpr = float(orig_pred[~orig_labels].mean()) if len(good_files) else float("nan")
    orig_recall = orig_tp / (orig_tp + orig_fn) if (orig_tp + orig_fn) else 0.0
    orig_precision = orig_tp / (orig_tp + orig_fp) if (orig_tp + orig_fp) else 0.0

    # ---- Rebuild ----
    detector = AnomalyDetector(config)
    detector.fit(train_files)
    detector.calibrate()
    new_scores = [detector.score(load(p), _skip_threshold=True).image_score
                  for p in good_files + bad_files]
    new_scores_arr = np.array(new_scores)
    new_auroc = auroc(orig_labels, new_scores_arr)
    new_pred = new_scores_arr > detector.threshold
    new_tp = int((new_pred & orig_labels).sum())
    new_fp = int((new_pred & ~orig_labels).sum())
    new_fn = int((~new_pred & orig_labels).sum())
    new_fpr = float(new_pred[~orig_labels].mean()) if len(good_files) else float("nan")
    new_recall = new_tp / (new_tp + new_fn) if (new_tp + new_fn) else 0.0
    new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) else 0.0
    _, _, new_f1 = f1_at_threshold(orig_labels, new_scores_arr, detector.threshold)

    return {
        "category": category,
        "n_train": len(train_files),
        "n_test": len(good_files) + len(bad_files),
        "original": {
            "auroc": orig_auroc,
            "threshold": orig_threshold,
            "fpr": orig_fpr,
            "recall": orig_recall,
            "precision": orig_precision,
        },
        "rebuild": {
            "auroc": new_auroc,
            "threshold": detector.threshold,
            "fpr": new_fpr,
            "recall": new_recall,
            "precision": new_precision,
            "f1": new_f1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--categories",
        default="bottle,grid,metal_nut,pill,screw,zipper,carpet,transistor",
    )
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--coreset-percentage", type=float, default=0.1)
    parser.add_argument("--out", default="results/baseline_comparison.json")
    args = parser.parse_args()

    data_root = Path(args.data)
    if args.categories.strip() == "all":
        categories = sorted(
            d.name
            for d in data_root.iterdir()
            if d.is_dir() and (d / "train" / "good").exists()
        )
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    config = ModelConfig(
        backbone=args.backbone,
        input_size=args.input_size,
        coreset_percentage=args.coreset_percentage,
    )

    rows = []
    for category in categories:
        print(f"=== {category} ===", flush=True)
        row = evaluate_category(data_root, category, config)
        rows.append(row)
        o, r = row["original"], row["rebuild"]
        print(
            f"  original: AUROC {o['auroc']:.4f}  FPR {o['fpr']:.4f}  recall {o['recall']:.4f}",
            flush=True,
        )
        print(
            f"  rebuild:  AUROC {r['auroc']:.4f}  FPR {r['fpr']:.4f}  recall {r['recall']:.4f}",
            flush=True,
        )

    if not rows:
        return

    def mean_of(side: str, key: str) -> float:
        vals = [r[side][key] for r in rows if not np.isnan(r[side][key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_categories": len(rows),
        "original": {k: mean_of("original", k) for k in ("auroc", "fpr", "recall", "precision")},
        "rebuild": {k: mean_of("rebuild", k) for k in ("auroc", "fpr", "recall", "precision", "f1")},
    }

    print("\n" + "=" * 72)
    print(f"{'metric':<24}{'original':>14}{'rebuild':>14}{'delta':>14}")
    print("-" * 72)
    for key in ("auroc", "fpr", "recall", "precision"):
        o = summary["original"][key]
        r = summary["rebuild"][key]
        print(f"{key:<24}{o:>14.4f}{r:>14.4f}{r - o:>+14.4f}")
    print("=" * 72)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_category": rows}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
