"""
Detector v1.5 (EXPERIMENTAL, post-hoc). Adds directory- and pairing-based
evidence for SEGMENTATION datasets on top of the frozen v1.4 rules.

WHY THIS EXISTS
  v1.4's segmentation rule (rule 5) only looks at FILE names containing
  "mask"/"segmentation" (or stems "gt"/"groundtruth"). Real segmentation
  releases usually put masks in a folder (masks/, GrTh/, SegmentationClass/,
  RealMask/, BinaryMask/) with the SAME file names as the images, or store a
  mask next to each image with the same stem and a different extension
  (001.bmp + 001.png). None of those layouts is caught by v1.4. In the
  15-dataset evaluation the segmentation rule was never exercised on a real
  mask-bearing dataset (the attached LISC copy had no masks).

DESIGN (monotonic, minimal-diff)
  1. Run frozen v1.4.
  2. If v1.4 returned a STRONG annotation result (confidence >= 0.65 and
     format not in classification_folders/unlabelled/abstain/segmentation_masks),
     return it unchanged. Strong VOC/COCO/YOLO/point/comma-delimited evidence
     is never overridden by mask heuristics.
  3. Otherwise look for segmentation evidence:
       S1 mask-named directories next to non-mask image directories
       S2 same-folder image pairs with the same stem and different extension
     and, when Pillow is available, confirm that the candidate mask images
     contain few distinct colours (real masks do; photographs do not).
  4. If evidence is found return segmentation_masks / segmentation, else
     return the v1.4 result unchanged.

REVISION NOTE (2026-10-13, after first contact with real data)
  The first draft of v1.5 was written against synthetic layouts and then run on
  three real mask-bearing downloads (Raabin-WBC ground truths in two packagings,
  Zheng WBC Dataset 1 and Dataset 2). It failed on both Raabin layouts for two
  reasons found on the real files: (a) ground-truth masks stored as JPEG carry
  ~35 distinct colours, so a "<= 16 colours" test rejected them, and (b) a folder
  called "Train_GT_bmp" was not recognised because "gt" sits in the middle of the
  name. Both rules were therefore revised. The real files used for this revision
  are DEVELOPMENT data for v1.5 and must not be reported as held-out for v1.5.
  (They remain a genuine held-out test for the frozen v1.4.)

STATUS
  Validated on synthetic layouts and on the three real downloads above only.
  Treat any number produced with v1.5 as POST-HOC (developed after seeing
  failures) and report it separately from frozen-v1.4 results.
"""
import os
import re
from collections import defaultdict
from pathlib import Path

from .rules_v1_4 import IMAGE_EXTS, detect_format_and_role as detect_v1_4

try:  # Pillow and NumPy are optional; without them only names/pairing/size are used
    from PIL import Image
    import numpy  # noqa: F401
except Exception:  # pragma: no cover
    Image = None

# A directory looks like a mask folder if any word in its name is one of these tokens
# (names are split on anything that is not a letter or digit, so "Train_GT_bmp" -> train, gt, bmp),
# or if its name with separators removed contains "groundtruth" ("Ground Truth", "ground_truth").
_MASK_TOKENS = {
    "gt", "gts", "mask", "masks", "grth", "groundtruth", "groundtruths",
    "segmentation", "segmentationclass", "segmentationobject",
    "binarymask", "realmask", "gtfine",
}
_MASK_TOKEN_PREFIX = re.compile(r"^(grth|mask)\d*$")


MASK_PEAK_COUNT = 3        # number of dominant grey levels examined
MASK_PEAK_WINDOW = 8       # +/- grey levels around each peak (absorbs JPEG ringing)
MASK_PEAK_COVERAGE = 0.95  # share of pixels that must fall near those peaks
SAMPLE_FOR_PIXEL_CHECK = 10
MIN_IMAGES_PER_SIDE = 3
MIN_PAIR_GROUPS = 5


def _is_mask_dir_name(name: str) -> bool:
    n = name.strip().lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    if any(t in _MASK_TOKENS or _MASK_TOKEN_PREFIX.match(t) for t in tokens):
        return True
    return "groundtruth" in "".join(tokens)


def _mask_like(path: Path) -> bool:
    """True if the image is dominated by a few grey levels, as masks are.

    A mask has almost all pixels at 2-4 levels (background, nucleus, cytoplasm), also when it was saved
    as JPEG (the levels blur slightly). A stained smear photograph spreads over many levels.
    Measured on real files: masks 1.000, photographs 0.18 to 0.86 (peak coverage, 3 peaks, +/-8 levels).
    """
    try:
        import numpy as np
        with Image.open(path) as im:
            a = np.asarray(im.convert("L"))
        hist = np.bincount(a.ravel(), minlength=256).astype(float)
        peaks = np.argsort(hist)[::-1][:MASK_PEAK_COUNT]
        keep = np.zeros(256, dtype=bool)
        for p in peaks:
            keep[max(0, p - MASK_PEAK_WINDOW): p + MASK_PEAK_WINDOW + 1] = True
        return bool(hist[keep].sum() / hist.sum() >= MASK_PEAK_COVERAGE)
    except Exception:
        return False


def _pixel_check(paths):
    """Returns (verified: bool|None, n_checked). None = Pillow/NumPy unavailable."""
    if Image is None:
        return None, 0
    sample = paths[:SAMPLE_FOR_PIXEL_CHECK]
    if not sample:
        return False, 0
    ok = sum(1 for p in sample if _mask_like(p))
    return (ok / len(sample) >= 0.7), len(sample)


def segmentation_evidence(root: Path, image_files):
    """Returns (found: bool, evidence: dict)."""
    ev = {}
    # S1: mask-named directories
    mask_dir_imgs, other_imgs = [], []
    # Ignore directories shared by EVERY image (for example an archive wrapper folder that happens to
    # be called "GrTh"): only folders that separate some images from others carry information.
    shared = len(Path(os.path.commonpath([str(p.parent) for p in image_files])).relative_to(root).parts) if image_files else 0
    for p in image_files:
        parts = p.relative_to(root).parts[:-1][shared:]
        (mask_dir_imgs if any(_is_mask_dir_name(x) for x in parts) else other_imgs).append(p)
    if len(mask_dir_imgs) >= MIN_IMAGES_PER_SIDE and len(other_imgs) >= MIN_IMAGES_PER_SIDE:
        verified, n = _pixel_check(sorted(mask_dir_imgs))
        if verified is None or verified:
            ev.update({"s1_mask_dir_images": len(mask_dir_imgs), "s1_other_images": len(other_imgs),
                       "s1_pixel_check": "pillow_unavailable_names_only" if verified is None else f"ok({n} sampled)"})
            return True, ev
        ev["s1_rejected"] = f"mask-named dirs but pixel check failed ({n} sampled)"
    # S2: same-folder, same-stem pairs with different extensions
    groups = defaultdict(list)
    for p in image_files:
        groups[(p.parent, p.stem.lower())].append(p)
    paired = [g for g in groups.values() if len({x.suffix.lower() for x in g}) >= 2]
    if len(paired) >= max(MIN_PAIR_GROUPS, int(0.5 * len(groups))):
        smaller = [min(g, key=lambda x: x.stat().st_size) for g in paired]
        verified, n = _pixel_check(sorted(smaller))
        if verified is None:  # no Pillow: fall back to a size ratio test
            ratios = []
            for g in paired[:SAMPLE_FOR_PIXEL_CHECK]:
                sizes = sorted(x.stat().st_size for x in g)
                ratios.append(sizes[0] <= 0.5 * sizes[-1])
            verified = sum(ratios) / max(1, len(ratios)) >= 0.7
            ev["s2_pixel_check"] = "pillow_unavailable_size_ratio"
        else:
            ev["s2_pixel_check"] = f"ok({n} sampled)" if verified else f"failed({n} sampled)"
        if verified:
            ev.update({"s2_pair_groups": len(paired), "s2_total_groups": len(groups)})
            return True, ev
    return False, ev


_STRONG_EXCLUDE = {"classification_folders", "unlabelled", "abstain", "segmentation_masks"}


def detect_format_and_role(root: Path, max_files_scanned=20000):
    root = Path(root)
    fmt, role, conf, evidence = detect_v1_4(root, max_files_scanned=max_files_scanned)
    if fmt not in _STRONG_EXCLUDE and conf >= 0.65:
        return fmt, role, conf, evidence
    if not root.exists():
        return fmt, role, conf, evidence
    image_files = []
    for i, p in enumerate(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            image_files.append(p)
        if i > max_files_scanned:
            break
    found, seg_ev = segmentation_evidence(root, image_files)
    if found:
        evidence = dict(evidence)
        evidence.update(seg_ev)
        evidence["v1_4_result"] = f"{fmt}/{role}"
        return "segmentation_masks", "segmentation", 0.75, evidence
    return fmt, role, conf, evidence
