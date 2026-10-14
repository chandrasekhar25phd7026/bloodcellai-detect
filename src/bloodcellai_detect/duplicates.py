"""
Near-duplicate and cross-dataset leakage screen.

PROVENANCE
  Ported from the Kaggle notebook `bdqi-v2`, cell 292 (`%writefile
  paperA_e7_duplicates_leakage.py`, the final of three revisions -- v282
  and v288 are earlier drafts, kept only in `docs/PROVENANCE.md`'s
  history, not in this file), the script behind Section 4.4 of the
  manuscript (Tables 9 and 10: within- and cross-dataset near-duplicate
  candidates). The core logic (hashing, banded search, filename-stem
  overlap) is unchanged; the dataset-specific driver (`run`, which read
  dataset paths from a specific locked ground-truth CSV and wrote fixed
  output filenames) is generalized below into `scan_datasets`, which
  takes a plain `{dataset_id: root_path}` mapping. A thin wrapper,
  `scan_datasets_from_locked_gt`, reproduces the original's CSV-reading
  convenience for users who already have a `bloodcellai-detect lock`
  ground-truth file.

METHOD, STATED PLAINLY (see the module docstring in the original script)
  Average-hash + difference-hash (64 bits each, 128 bits combined), chosen
  over phash/DCT specifically to avoid a scipy dependency. Near-duplicate
  search uses banding (LSH-style): the fingerprint is split into
  equal-size bands, and two images are compared in full only if they
  share at least one band exactly. This is approximate -- a true
  near-duplicate whose bits differ across every band is missed -- and
  that tradeoff is reported, not hidden. Patient/slide-identifier
  metadata is NOT determined from image bytes; a worksheet is written
  for that to be filled in by hand from each dataset's real
  documentation.

Requires Pillow and NumPy: ``pip install "bloodcellai-detect[validate]"``.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "duplicates.py needs Pillow and NumPy: pip install \"bloodcellai-detect[validate]\""
    ) from e

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- hashing
def ahash_bits(img, hash_size: int):
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    return (arr > arr.mean()).flatten()


def dhash_bits(img, hash_size: int):
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def combined_fingerprint(img_path, hash_size: int):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        a, d = ahash_bits(im, hash_size), dhash_bits(im, hash_size)
    return np.concatenate([a, d])


def hamming(a, b) -> int:
    return int(np.count_nonzero(a != b))


# --------------------------------------------------------------------------- discovery and exact stem overlap
def discover_images(root: Path, exts=IMAGE_EXTS, cap: Optional[int] = None, seed: int = 0):
    """Returns (full_list, capped_list). full_list is every real image file found (used for
    the cheap, exact filename-stem overlap check, which needs the complete file set to be
    meaningful). capped_list is the (possibly downsampled) set used for perceptual hashing."""
    root = Path(root)
    if not root.exists():
        return [], []
    full = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    capped = full
    if cap is not None and len(full) > cap:
        rng = random.Random(seed)
        capped = sorted(rng.sample(full, cap))
    return full, capped


def filename_stem_overlap(full_files_by_dataset: Dict[str, List[Path]]) -> Dict[Tuple[str, str], Dict]:
    """Exact filename-stem overlap between every pair of datasets -- cheap, exact, computed
    on the full uncapped file listing. A stronger, independent signal than hashing for two
    exports of the same underlying photographs."""
    stems = {ds_id: {p.stem for p in files} for ds_id, files in full_files_by_dataset.items()}
    ds_ids = sorted(stems.keys())
    overlap = {}
    for i in range(len(ds_ids)):
        for j in range(i + 1, len(ds_ids)):
            a, b = ds_ids[i], ds_ids[j]
            common = stems[a] & stems[b]
            if common:
                overlap[(a, b)] = dict(n_common_stems=len(common), n_a=len(stems[a]), n_b=len(stems[b]),
                                        sample=sorted(common)[:5])
    return overlap


# --------------------------------------------------------------------------- banded near-duplicate search
def band_key(fingerprint, band_idx: int, band_width: int) -> int:
    start = band_idx * band_width
    bits = fingerprint[start:start + band_width]
    return int("".join("1" if b else "0" for b in bits), 2)


def find_near_duplicates(items: Sequence[Tuple[object, object]], n_bands: int, hamming_threshold: int):
    """items: list of (key, fingerprint). Returns [(key_a, key_b, hamming_dist), ...] for every
    candidate pair sharing >=1 band exactly and within the Hamming threshold, sorted ascending
    by distance. Approximate (LSH-style): a true near-duplicate pair differing across every
    single band is missed; more bands trade recall for more candidate pairs to verify."""
    n_bits = len(items[0][1]) if items else 0
    band_width = n_bits // n_bands
    buckets = [defaultdict(list) for _ in range(n_bands)]
    for idx, (_, fp) in enumerate(items):
        for b in range(n_bands):
            buckets[b][band_key(fp, b, band_width)].append(idx)

    candidate_pairs = set()
    for b in range(n_bands):
        for bucket_idxs in buckets[b].values():
            if len(bucket_idxs) < 2:
                continue
            for i in range(len(bucket_idxs)):
                for j in range(i + 1, len(bucket_idxs)):
                    a, bb = bucket_idxs[i], bucket_idxs[j]
                    candidate_pairs.add((a, bb) if a < bb else (bb, a))

    results = []
    for a, b in candidate_pairs:
        d = hamming(items[a][1], items[b][1])
        if d <= hamming_threshold:
            results.append((items[a][0], items[b][0], d))
    results.sort(key=lambda r: r[2])
    return results


# --------------------------------------------------------------------------- scan driver (generalized)
def scan_datasets(dataset_roots: Dict[str, Path], hash_size: int = 8, n_bands: int = 8,
                   hamming_threshold: int = 6, max_images_per_dataset: Optional[int] = 3000,
                   sample_seed: int = 20260926, max_review_sample: int = 200,
                   image_exts=IMAGE_EXTS) -> Dict[str, object]:
    """Fingerprints every image under each root in `dataset_roots` ({dataset_id: path}),
    finds near-duplicate candidates within and across datasets, and checks exact filename-stem
    overlap. Does no dataset-specific path handling -- point it at whatever roots you have;
    see `scan_datasets_from_locked_gt` to read them from a `bloodcellai-detect lock` file."""
    all_items, per_dataset_counts, full_files_by_dataset = [], {}, {}
    for ds_id, root in sorted(dataset_roots.items()):
        full_files, capped_files = discover_images(Path(root), image_exts, max_images_per_dataset, sample_seed)
        full_files_by_dataset[ds_id] = full_files
        per_dataset_counts[ds_id] = len(capped_files)
        for p in capped_files:
            try:
                all_items.append(((ds_id, str(p)), combined_fingerprint(p, hash_size)))
            except Exception:
                continue  # unreadable image file; skipped, not counted

    stem_overlap = filename_stem_overlap(full_files_by_dataset)
    pairs = find_near_duplicates(all_items, n_bands, hamming_threshold)
    within_dataset = [p for p in pairs if p[0][0] == p[1][0]]
    cross_dataset = [p for p in pairs if p[0][0] != p[1][0]]

    cross_dataset_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    for (ds_a, _), (ds_b, _), _ in cross_dataset:
        cross_dataset_matrix[tuple(sorted((ds_a, ds_b)))] += 1
    within_dataset_matrix: Dict[str, int] = defaultdict(int)
    for (ds_a, _), _, _ in within_dataset:
        within_dataset_matrix[ds_a] += 1

    # per-pair-key sampling (not a single global top-N), so one dataset's internal redundancy
    # cannot starve every other pair's review sample -- see module docstring.
    per_key_pairs: Dict[Tuple[str, str], list] = defaultdict(list)
    for (ds_a, path_a), (ds_b, path_b), d in pairs:
        per_key_pairs[tuple(sorted((ds_a, ds_b)))].append(((ds_a, path_a), (ds_b, path_b), d))
    per_key_cap = max(1, max_review_sample // max(1, len(per_key_pairs)))
    review_sample = [row for plist in per_key_pairs.values() for row in plist[:per_key_cap]]

    # targeted, quantitative check: do shared-stem files across a dataset pair actually hash
    # as near-duplicates, or is the stem match a naming coincidence (e.g. both number 1,2,3...)?
    stem_content_checks = {}
    for (a, b), info in stem_overlap.items():
        files_a = {p.stem: p for p in full_files_by_dataset.get(a, [])}
        files_b = {p.stem: p for p in full_files_by_dataset.get(b, [])}
        checks = []
        for stem in info["sample"]:
            pa, pb = files_a.get(stem), files_b.get(stem)
            if pa is None or pb is None:
                continue
            try:
                d = hamming(combined_fingerprint(pa, hash_size), combined_fingerprint(pb, hash_size))
                checks.append(dict(stem=stem, path_a=str(pa), path_b=str(pb), hamming_distance=d,
                                    likely_same_content=(d <= hamming_threshold)))
            except Exception as e:
                checks.append(dict(stem=stem, error=str(e)))
        stem_content_checks[f"{a}|{b}"] = checks

    return dict(
        n_datasets=len(dataset_roots), n_images_fingerprinted=len(all_items),
        n_candidate_pairs=len(pairs), n_within_dataset=len(within_dataset), n_cross_dataset=len(cross_dataset),
        within_dataset_matrix=dict(within_dataset_matrix),
        cross_dataset_matrix={f"{a}|{b}": n for (a, b), n in cross_dataset_matrix.items()},
        filename_stem_overlap={f"{a}|{b}": info for (a, b), info in stem_overlap.items()},
        stem_content_checks=stem_content_checks,
        per_dataset_image_counts=per_dataset_counts,
        per_dataset_full_image_counts={k: len(v) for k, v in full_files_by_dataset.items()},
        review_sample=[dict(dataset_a=a, path_a=pa, dataset_b=b, path_b=pb, hamming_distance=d,
                             scope=("within_dataset" if a == b else "cross_dataset"))
                       for (a, pa), (b, pb), d in review_sample],
    )


def scan_datasets_from_locked_gt(gt_csv_path: Path, **kwargs) -> Dict[str, object]:
    """Convenience wrapper: reads {dataset_id: kaggle_path_hint} from a ground-truth CSV
    already written and locked via `bloodcellai_detect.lock` (same file shape as the E1
    audit), then calls `scan_datasets`. Rows with no path set are skipped."""
    roots = {}
    with open(gt_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds_id, hint = row.get("dataset_id"), row.get("kaggle_path_hint")
            if ds_id and hint and hint not in ("FILL_ME_IN", "NOT_YET_ATTACHED", ""):
                roots[ds_id] = Path(hint)
    return scan_datasets(roots, **kwargs)
