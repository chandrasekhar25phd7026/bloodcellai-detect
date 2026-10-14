import numpy as np
import pytest
from PIL import Image

from bloodcellai_detect.duplicates import (
    ahash_bits, dhash_bits, combined_fingerprint, hamming, find_near_duplicates,
    discover_images, filename_stem_overlap, scan_datasets, scan_datasets_from_locked_gt,
)


def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_identical_images_have_zero_hamming_distance(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    _save(p1, img)
    _save(p2, img)  # byte-identical content, different filename
    d = hamming(combined_fingerprint(p1, 8), combined_fingerprint(p2, 8))
    assert d == 0


def test_very_different_images_have_large_hamming_distance(tmp_path):
    # NOTE: two perfectly UNIFORM images (e.g. solid black vs solid white) hash identically
    # under both ahash and dhash -- every pixel equals the image's own mean / its neighbour,
    # regardless of the image's absolute brightness. That is a real, known property of these
    # two specific hash algorithms (not a bug in this port), so the "very different" pair
    # here uses structured patterns, which these hashes are actually designed to separate.
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    xs, ys = np.meshgrid(np.arange(32), np.arange(32))
    checkerboard = (((xs // 4) + (ys // 4)) % 2 * 255).astype(np.uint8)
    checkerboard = np.stack([checkerboard] * 3, axis=-1)
    p1, p2 = tmp_path / "noisy.png", tmp_path / "checker.png"
    _save(p1, noisy)
    _save(p2, checkerboard)
    d = hamming(combined_fingerprint(p1, 8), combined_fingerprint(p2, 8))
    assert d > 20


def test_uniform_images_are_a_known_hashing_blind_spot(tmp_path):
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    p1, p2 = tmp_path / "black.png", tmp_path / "white.png"
    _save(p1, black)
    _save(p2, white)
    d = hamming(combined_fingerprint(p1, 8), combined_fingerprint(p2, 8))
    assert d == 0  # disclosed limitation: ahash/dhash cannot tell two uniform images apart


def test_find_near_duplicates_detects_identical_pair_across_bands(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    noise = img.astype(np.int16)
    noise[0, 0, 0] = (noise[0, 0, 0] + 5) % 255  # tiny perturbation
    p1, p2, p3 = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    _save(p1, img)
    _save(p2, noise)
    _save(p3, np.zeros((32, 32, 3), dtype=np.uint8))  # unrelated
    items = [(("ds", str(p)), combined_fingerprint(p, 8)) for p in (p1, p2, p3)]
    pairs = find_near_duplicates(items, n_bands=8, hamming_threshold=6)
    keys = {frozenset((a, b)) for a, b, _ in pairs}
    assert frozenset((("ds", str(p1)), ("ds", str(p2)))) in keys
    assert frozenset((("ds", str(p1)), ("ds", str(p3)))) not in keys


def test_discover_images_caps_with_fixed_seed_sample(tmp_path):
    for i in range(10):
        _save(tmp_path / f"{i}.png", np.zeros((4, 4, 3), dtype=np.uint8))
    full, capped = discover_images(tmp_path, cap=3, seed=0)
    assert len(full) == 10 and len(capped) == 3
    full2, capped2 = discover_images(tmp_path, cap=3, seed=0)
    assert capped == capped2  # same seed -> same sample


def test_filename_stem_overlap_finds_shared_stems(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for name in ("1.png", "2.png", "3.png"):
        _save(a / name, np.zeros((4, 4, 3), dtype=np.uint8))
    for name in ("2.png", "3.png", "4.png"):
        _save(b / name, np.zeros((4, 4, 3), dtype=np.uint8))
    full_a, _ = discover_images(a)
    full_b, _ = discover_images(b)
    overlap = filename_stem_overlap({"a": full_a, "b": full_b})
    assert overlap[("a", "b")]["n_common_stems"] == 2


def test_scan_datasets_end_to_end_within_and_cross(tmp_path):
    ds1, ds2 = tmp_path / "ds1", tmp_path / "ds2"
    ds1.mkdir(); ds2.mkdir()
    rng = np.random.default_rng(2)
    shared = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    _save(ds1 / "x1.png", shared)
    _save(ds1 / "x2.png", shared)  # within-dataset near-duplicate
    _save(ds2 / "y1.png", shared)  # cross-dataset near-duplicate of ds1's images
    _save(ds2 / "y2.png", rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    result = scan_datasets({"ds1": ds1, "ds2": ds2}, hamming_threshold=6, max_images_per_dataset=None)
    assert result["n_images_fingerprinted"] == 4
    assert result["n_within_dataset"] >= 1
    assert result["n_cross_dataset"] >= 1
    assert "ds1" in result["within_dataset_matrix"]


def test_scan_datasets_from_locked_gt_skips_unset_paths(tmp_path):
    import csv
    ds1 = tmp_path / "ds1"
    ds1.mkdir()
    _save(ds1 / "x.png", np.zeros((8, 8, 3), dtype=np.uint8))
    gt = tmp_path / "gt.csv"
    with open(gt, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "kaggle_path_hint"])
        w.writeheader()
        w.writerow({"dataset_id": "ds1", "kaggle_path_hint": str(ds1)})
        w.writerow({"dataset_id": "ds2", "kaggle_path_hint": "NOT_YET_ATTACHED"})
    result = scan_datasets_from_locked_gt(gt)
    assert result["n_datasets"] == 1  # ds2 skipped
