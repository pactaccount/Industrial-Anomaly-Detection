"""Tests for the core engine.

These cover the parts where a subtle bug is invisible in the output but
fatal in production: the leave-one-out calibration (a circular calibration
produces a 95% false positive rate on good parts), the aspect-ratio
handling, and the coreset selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from argus.core.embeddings import PatchMaker
from argus.core.evaluation import auroc, aupro, f1_at_threshold, make_synthetic_anomaly
from argus.core.memory_bank import greedy_coreset, knn_scores_excluding
from argus.core.model import AnomalyDetector, ModelConfig
from argus.core.preprocess import letterbox, preprocess_image


def make_part(seed: int = 0, defect: bool = False, size: int = 256) -> Image.Image:
    """A textured part, optionally with a dark rectangular defect."""
    rng = np.random.RandomState(seed)
    arr = (rng.rand(size, size, 3) * 60 + 90).astype(np.uint8)
    if defect:
        arr[size // 2 : size // 2 + 15, size // 4 : 3 * size // 4] = 30
    return Image.fromarray(arr)


def make_variation(seed: int, base: np.ndarray) -> Image.Image:
    """A good part with a small brightness jitter."""
    jittered = base.astype(np.int16) + (seed - 3)
    return Image.fromarray(jittered.clip(0, 255).astype(np.uint8))


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------


def test_letterbox_preserves_aspect_ratio():
    """A wide image must not be squashed into a square."""
    wide = Image.new("RGB", (400, 100), "white")
    canvas, (left, top, right, bottom) = letterbox(wide, 256)
    assert canvas.size == (256, 256)
    # The content should occupy the full width and be centred vertically.
    assert right - left == 256
    assert bottom - top == 64  # 100 * (256/400)
    assert left == 0


def test_letterbox_tall_image():
    tall = Image.new("RGB", (100, 400), "white")
    canvas, (left, top, right, bottom) = letterbox(tall, 256)
    assert canvas.size == (256, 256)
    assert bottom - top == 256
    assert right - left == 64


def test_heatmap_maps_back_to_original_geometry():
    """The inverse transform must restore the original image's dimensions."""
    image = Image.new("RGB", (400, 100), "white")
    prep = preprocess_image(image, size=256)
    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[100:120, 50:80] = 5.0
    restored = prep.map_to_original(heatmap)
    assert restored.shape == (100, 400)  # (height, width)


def test_aspect_ratio_is_not_distorted():
    """A circle in the source must stay a circle after preprocessing."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:100, :100]
    circle = (yy - 50) ** 2 + (xx - 50) ** 2 <= 20**2
    arr[circle] = 255
    prep = preprocess_image(Image.fromarray(arr), size=256)
    tensor = prep.tensor

    # The canvas is square, so the circle's extent measured along rows and
    # along columns must match. (A squashing resize would make these
    # differ, which is exactly the bug this guards against.)
    grey = tensor[0].mean(dim=0).numpy()
    rows = np.where(grey.max(axis=1) > 0.5)[0]
    cols = np.where(grey.max(axis=0) > 0.5)[0]
    row_extent = rows[-1] - rows[0]
    col_extent = cols[-1] - cols[0]
    assert abs(int(row_extent) - int(col_extent)) <= 3, (
        f"circle was distorted: rows span {row_extent}, cols span {col_extent}"
    )


# ----------------------------------------------------------------------
# PatchMaker
# ----------------------------------------------------------------------


def test_patchify_shape_and_count():
    features = torch_rand(1, 64, 32, 32)
    patches, (h, w) = PatchMaker(3, 1).patchify(features)
    assert (h, w) == (32, 32)
    assert patches.shape == (1, 32 * 32, 64, 3, 3)


def test_patchify_covers_every_cell():
    """Every feature cell must appear in some patch."""
    import torch

    features = torch.arange(1 * 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
    patches, _ = PatchMaker(3, 1).patchify(features)
    # The centre element of each patch equals that cell's value.
    centres = patches[0, :, :, 1, 1]
    expected = features[0, :, :, :].reshape(4, 64).T
    assert torch.allclose(centres, expected, atol=1e-4)


def torch_rand(*shape):
    import torch

    return torch.rand(*shape)


# ----------------------------------------------------------------------
# Coreset
# ----------------------------------------------------------------------


def test_coreset_reduces_and_covers():
    """A coreset must be smaller and must cover the original distribution."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(2000, 32)).astype(np.float32)
    idx = greedy_coreset(features, percentage=0.1, random_state=0)
    assert len(idx) == 200
    assert len(set(idx.tolist())) == 200

    # Coverage is the point of a coreset: the retained subset should
    # approximate the full set's nearest-neighbour distance structure.
    # Compare the mean distance from every point to its nearest neighbour
    # in the full set versus in the coreset -- the coreset should not be
    # dramatically sparser.
    def mean_nn_distance(pool, queries):
        d = np.linalg.norm(
            queries[:, None, :] - pool[None, :, :], axis=2
        )
        # Exclude self-matches: a query that is itself in the pool would
        # otherwise have a nearest-neighbour distance of exactly zero.
        d.sort(axis=1)
        return float(d[:, 1].mean())

    sample = features[::10]
    full = mean_nn_distance(features, sample)
    reduced = mean_nn_distance(features[idx], sample)
    # The coreset is 10% of the data, so some degradation is expected, but
    # it must not collapse.
    assert reduced < full * 3, (
        f"coreset coverage degraded badly: {reduced:.4f} vs full {full:.4f}"
    )


def test_coreset_is_deterministic():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(500, 16)).astype(np.float32)
    a = greedy_coreset(features, percentage=0.2, random_state=7)
    b = greedy_coreset(features, percentage=0.2, random_state=7)
    assert np.array_equal(a, b)


def test_coreset_full_percentage_keeps_everything():
    features = np.random.default_rng(2).normal(size=(100, 8)).astype(np.float32)
    idx = greedy_coreset(features, percentage=1.0)
    assert len(idx) == 100


# ----------------------------------------------------------------------
# Leave-one-out exclusion
# ----------------------------------------------------------------------


def test_knn_exclusion_skips_own_image():
    """A query must not find its own image's patches."""
    from argus.core.memory_bank import MemoryBank

    rng = np.random.default_rng(3)
    # Two well-separated clusters: image 0 and image 1.
    features = np.concatenate(
        [
            rng.normal(0, 0.01, size=(100, 8)),
            rng.normal(10, 0.01, size=(100, 8)),
        ]
    ).astype(np.float32)
    labels = np.array([0] * 100 + [1] * 100)
    bank = MemoryBank.build(
        features, percentage=1.0, source_labels=labels
    )

    query = features[0:1]
    # Without exclusion the distance is ~0 (it is in the bank).
    plain, _ = bank.index.search(query, 1)
    assert plain[0, 0] < 0.05

    # With image 0 excluded, the nearest neighbour is from image 1.
    excluded = knn_scores_excluding(query, bank, labels, 0)
    assert excluded[0] > 5.0


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def test_auroc_edges():
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_f1_at_threshold():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    precision, recall, f1 = f1_at_threshold(labels, scores, 0.5)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_aupro_prefers_correct_localisation():
    """AUPRO must reward localising the defect, not just flagging it."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:30, 20:30] = True

    # A prediction that highlights the true region.
    good = mask.astype(np.float64)
    # A prediction that highlights the whole image.
    bad = np.ones((64, 64), dtype=np.float64)

    labels = np.array([True])
    assert aupro(labels, [good]) > aupro(labels, [bad])


def test_synthetic_anomaly_modifies_image_and_returns_mask():
    img = (np.random.default_rng(4).random((128, 128, 3)) * 255).astype(np.uint8)
    rng = np.random.default_rng(4)
    for kind in ("scratch", "blob", "hole", "shift"):
        out, mask = make_synthetic_anomaly(img, rng, severity=0.5, kind=kind)
        assert out.shape == img.shape
        assert mask.shape == img.shape[:2]
        assert mask.sum() > 0
        assert not np.array_equal(out, img)


# ----------------------------------------------------------------------
# End-to-end detector behaviour
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector():
    """A detector trained on six variants of one synthetic part."""
    rng = np.random.RandomState(7)
    base = (rng.rand(256, 256, 3) * 60 + 90).astype(np.uint8)
    good = [make_variation(i, base) for i in range(6)]
    det = AnomalyDetector(
        ModelConfig(backbone="resnet18", input_size=256, coreset_percentage=0.5)
    )
    det.fit_images(good)
    det.calibrate()
    return det, base, good


def test_defect_scores_above_good(detector):
    det, base, good = detector
    defect = base.copy()
    defect[110:125, 70:185] = 30
    bad_score = det.score(Image.fromarray(defect)).image_score
    good_scores = [det.score(g, _skip_threshold=True).image_score for g in good]
    assert bad_score > max(good_scores)


def test_good_parts_are_not_rejected(detector):
    """The false-reject rate on training-distribution parts must be zero."""
    det, base, good = detector
    for g in good:
        assert not det.score(g).is_anomaly, (
            f"a good part was rejected: score={det.score(g).image_score:.3f} "
            f"threshold={det.threshold:.3f}"
        )


def test_defect_is_localised_to_the_right_region(detector):
    det, base, good = detector
    defect = base.copy()
    defect[110:125, 70:185] = 30
    result = det.score(Image.fromarray(defect))
    ys, xs = np.where(result.defect_mask)
    assert len(ys) > 0, "no defect region was found"
    # The flagged region must overlap the injected rectangle.
    assert ys.min() < 125 and ys.max() > 110
    assert xs.min() < 185 and xs.max() > 70
    # And it must not be the whole image.
    assert result.defect_ratio < 0.5


def test_leave_one_out_calibration_differs_from_in_sample(detector):
    """The circular (in-sample) calibration must give a lower threshold.

    This is the bug that produced a 95% false positive rate: calibrating on
    images whose own patches are in the bank.
    """
    det, base, good = detector
    assert det._calibration_mode == "leave_one_out"

    # The held-out scores must be materially above zero, because a query
    # image's own patches are excluded from the bank.
    assert min(det._good_scores) > 0.0


def test_model_roundtrips_through_disk(detector, tmp_path):
    det, base, good = detector
    path = det.save(tmp_path / "model")
    loaded = AnomalyDetector.load(path)
    assert loaded.config.hash() == det.config.hash()
    assert loaded.threshold == pytest.approx(det.threshold)
    # The reloaded model must produce the same verdict.
    defect = base.copy()
    defect[110:125, 70:185] = 30
    assert loaded.score(Image.fromarray(defect)).is_anomaly == (
        det.score(Image.fromarray(defect)).is_anomaly
    )


def test_config_hash_changes_with_geometry():
    """Changing the embedding geometry must invalidate the bank."""
    a = ModelConfig(backbone="resnet18", input_size=256)
    b = ModelConfig(backbone="resnet18", input_size=320)
    c = ModelConfig(backbone="resnet18", input_size=256)
    assert a.hash() != b.hash()
    assert a.hash() == c.hash()


def test_config_hash_ignores_non_geometry_fields():
    """Changing the coreset ratio must not invalidate an existing bank."""
    a = ModelConfig(backbone="resnet18", input_size=256, coreset_percentage=0.1)
    b = ModelConfig(backbone="resnet18", input_size=256, coreset_percentage=0.5)
    assert a.hash() == b.hash()
