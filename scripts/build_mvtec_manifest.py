"""Build a manifest linking MVTec AD test images to their ground-truth masks.

The HuggingFace mirror renames files (``000-94.png``) while the ground-truth
masks keep the original MVTec names (``000_mask.png``). ``samples.json``
records the correct pairing via each defective sample's ``defect_mask``
field, so we use that to write a per-category manifest the evaluator can
consume.

Usage::

    python scripts/build_mvtec_manifest.py --source <hf_dir> --out <mvtec_root>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="HuggingFace mirror directory")
    parser.add_argument("--out", required=True, help="Prepared MVTec root")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    data = json.loads((source / "samples.json").read_text())

    # category -> list of {image, mask, defect}
    manifest: dict[str, list[dict]] = defaultdict(list)
    mask_missing = 0

    for sample in data["samples"]:
        category = sample.get("category", {}).get("label", "unknown")
        defect = sample.get("defect", {}).get("label", "unknown")
        split = sample.get("split", "train")
        filename = Path(sample["filepath"]).name

        if split == "train":
            image_rel = f"train/good/{filename}"
            mask_rel = None
        else:
            image_rel = f"test/{defect}/{filename}"
            mask_field = sample.get("defect_mask")
            if mask_field and mask_field.get("mask_path"):
                # The mask path points into the mirror's fields/ tree; the
                # prepared dataset keeps the mask under ground_truth/.
                mask_name = Path(mask_field["mask_path"]).name
                mask_rel = f"ground_truth/{defect}/{mask_name}"
            else:
                mask_rel = None
                if defect != "good":
                    mask_missing += 1

        manifest[category].append(
            {
                "image": image_rel,
                "mask": mask_rel,
                "defect": defect,
                "split": split,
            }
        )

    # Resolve masks to their real location in the prepared tree and check
    # they exist.
    for category, entries in manifest.items():
        gt_root = out / category / "ground_truth"
        for entry in entries:
            if not entry["mask"]:
                continue
            # The mirror disambiguates filenames: the image `002-94.png` and
            # its mask `002_mask-67.png` both carry a numeric suffix, but
            # their shared base id is `002`. The ground-truth masks use the
            # clean MVTec names (`002_mask.png`) and live in a per-defect
            # directory, so match on the base id within that directory.
            mask_name = Path(entry["mask"]).name
            base_id = mask_name.split("_mask")[0]
            defect = entry["defect"]
            candidates = [
                gt_root / defect / f"{base_id}_mask.png",
                gt_root / "ground_truth" / defect / f"{base_id}_mask.png",
                gt_root / defect / mask_name,
                gt_root / "ground_truth" / defect / mask_name,
                gt_root / f"{base_id}_mask.png",
                gt_root / mask_name,
            ]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                entry["mask"] = None
                mask_missing += 1
                continue
            entry["mask"] = str(resolved.relative_to(out / category))

        out_file = out / category / "manifest.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps({"category": category, "samples": entries}, indent=1))

    total = sum(len(v) for v in manifest.values())
    defective = sum(
        1
        for entries in manifest.values()
        for e in entries
        if e["split"] == "test" and e["defect"] != "good"
    )
    with_mask = sum(
        1
        for entries in manifest.values()
        for e in entries
        if e["mask"]
    )
    print(f"Categories: {len(manifest)}")
    print(f"Samples:   {total} ({defective} defective test images)")
    print(f"With mask: {with_mask}")
    if with_mask != defective:
        print(f"MISSING:   {defective - with_mask} defective images have no mask")


if __name__ == "__main__":
    main()
