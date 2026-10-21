"""Prepare the MVTec AD dataset from the HuggingFace mirror.

The mirror stores images flat across numbered directories with a
``samples.json`` describing category, defect type and train/test split. This
script reorganises it into the standard MVTec layout::

    <out>/<category>/train/good/*.png
    <out>/<category>/test/<defect>/*.png
    <out>/<category>/ground_truth/<defect>/*_mask.png

Ground-truth masks are not in this mirror, so we produce the directory
structure and a manifest; masks can be added separately if available.

Usage::

    python scripts/prepare_mvtec.py --source <hf_dir> --out <target>
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def load_manifest(samples_json: Path) -> list[dict]:
    """Read the mirror's samples.json into a flat list of records."""
    data = json.loads(samples_json.read_text())
    records = []
    for sample in data["samples"]:
        category = sample.get("category", {}).get("label", "unknown")
        defect = sample.get("defect", {}).get("label", "unknown")
        split = sample.get("split", "train")
        records.append(
            {
                "filepath": sample["filepath"],
                "category": category,
                "defect": defect,
                "split": split,
            }
        )
    return records


def reorganise(source: Path, out: Path) -> dict:
    """Copy images into the standard MVTec directory layout."""
    manifest = load_manifest(source / "samples.json")
    stats: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for record in manifest:
        src = source / record["filepath"]
        if not src.exists():
            continue
        category = record["category"]
        defect = record["defect"]
        split = record["split"]

        if split == "train":
            dest_dir = out / category / "train" / "good"
        else:
            dest_dir = out / category / "test" / defect
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / Path(record["filepath"]).name
        if not dest.exists():
            shutil.copy2(src, dest)
        stats[category][f"{split}/{defect}"] += 1

    # Write a manifest per category so the evaluator can find the labels.
    for category in stats:
        manifest_path = out / category / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "category": category,
                    "train_good": sorted(
                        str(p.relative_to(out / category / "train" / "good"))
                        for p in (out / category / "train" / "good").glob("*.png")
                    ),
                    "test": {
                        defect: sorted(
                            str(
                                p.relative_to(out / category / "test" / defect)
                            )
                            for p in (out / category / "test" / defect).glob("*.png")
                        )
                        for defect in stats[category]
                        if defect.startswith("test/")
                    },
                },
                indent=2,
            )
        )

    return {k: dict(v) for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="HuggingFace mirror directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = reorganise(source, out)
    print(f"Prepared {len(stats)} categories into {out}")
    for category, counts in sorted(stats.items()):
        total = sum(counts.values())
        train = counts.get("train/good", 0)
        test = total - train
        print(f"  {category:<14} train={train:<4} test={test}")


if __name__ == "__main__":
    main()
