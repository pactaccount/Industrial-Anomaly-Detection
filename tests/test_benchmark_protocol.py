"""Guard the public benchmark against train/test leakage and missing masks."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_mvtec import collect_test_samples


def _save(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_manifest_only_uses_test_split(tmp_path):
    cat = tmp_path / "metal_nut"
    _save(cat / "train/good/000.png", 10)
    _save(cat / "test/good/001.png", 20)
    _save(cat / "test/bent/002.png", 30)
    _save(cat / "ground_truth/bent/002_mask.png", 255)
    entries = [
        {"split": "train", "image": "train/good/000.png", "defect": "good"},
        {"split": "test", "image": "test/good/001.png", "defect": "good"},
        {"split": "test", "image": "test/bent/002.png", "defect": "bent", "mask": "ground_truth/bent/002_mask.png"},
    ]
    (cat / "manifest.json").write_text(json.dumps({"samples": entries}))
    good, bad, masks = collect_test_samples(cat)
    assert [p.name for p in good] == ["001.png"]
    assert [p.name for p in bad] == ["002.png"]
    assert len(masks) == 1


def test_duplicate_train_test_content_fails(tmp_path):
    cat = tmp_path / "metal_nut"
    _save(cat / "train/good/000.png", 10)
    _save(cat / "test/good/001.png", 10)
    _save(cat / "test/bent/002.png", 30)
    with pytest.raises(ValueError, match="overlap"):
        collect_test_samples(cat)
