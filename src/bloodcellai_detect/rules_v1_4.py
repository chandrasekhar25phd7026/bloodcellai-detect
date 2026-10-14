"""
Frozen format/role detection rules, detector v1.4.

PROVENANCE
  Extracted verbatim (logic and comments unchanged) from the Kaggle notebook
  `bdqi-v2` (cell 299, the `%writefile paperA_e1_audit.py` cell, "v1.4").
  This is the exact rule set that produced the headline figures reported in
  the manuscript (format accuracy 13/15, role accuracy 14/15, abstain rate 0/15).

DO NOT EDIT THE RULES IN THIS FILE.
  Any behavioural change belongs in a new, separately named version
  (see rules_v1_5.py) so that published numbers stay reproducible.

Only the module header (imports, IMAGE_EXTS) was added for packaging.
"""
import json
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def _looks_like_voc_xml(text):
    return "<annotation>" in text and "<object>" in text and "<bndbox>" in text


def _find_classification_root(root: Path, max_depth: int = 5) -> Path:
    """v1.3: descends through single-child wrapper directories before rule 6
    looks for sibling class folders. Handles two real cases identically,
    because they look identical to the detector -- a chain of directories
    with nothing else of substance in them until you reach the real content:
    (a) the mutation stress test's extra_nesting mutation, which wraps a
    dataset in root/extra/nested/folders/<contents>, and (b) an ordinary
    Kaggle zip attachment that unpacks into one top-level wrapper folder
    around the real class folders. Stops descending the moment a level has
    more than one child, or an image file sitting directly in it, or
    max_depth is hit, so it never skips past real content -- only past
    pass-through wrappers.

    v1.4: relaxed from "zero files of any kind" to "zero IMAGE files"
    directly in the wrapper level. Root-caused against a real attachment
    (aml_cytomorphology_lmu): its dataset root has real sidecar metadata
    files (abbreviations.txt, annotations.dat) sitting next to a single
    data/ subdirectory, which itself wraps another single data/ subdirectory
    before the real 15 class folders appear -- a double-nested wrapper with
    root-level metadata that v1.3's stricter check refused to descend past
    at all, since it required the wrapper level to have zero files of any
    kind. Sidecar files that aren't images carry no image content that
    descending past them could hide, so they no longer block descent. This
    stays safe against false positives because rule 6 separately requires
    zero xml_files/json_files/paired-txt+image files anywhere in the WHOLE
    recursive subtree scanned from the original root (see detect_format_and_role)
    before it will call classification_folders -- a real annotation file
    sitting at any wrapper level still correctly blocks that rule, regardless
    of what this function descends through."""
    current = root
    depth = 0
    while depth < max_depth and current.is_dir():
        try:
            entries = list(current.iterdir())
        except OSError:
            break
        dirs = [e for e in entries if e.is_dir()]
        image_files_here = [e for e in entries if e.is_file() and e.suffix.lower() in IMAGE_EXTS]
        if len(dirs) == 1 and len(image_files_here) == 0:
            current = dirs[0]
            depth += 1
            continue
        break
    return current


def detect_format_and_role(root: Path, max_files_scanned=20000):
    """Returns (format, role, confidence, evidence_dict). Predicts 'abstain'
    for both format and role when no signal clears the bar — abstaining is a
    correct, scoreable outcome, not a failure to hide."""
    if not root.exists():
        return "abstain", "abstain", 0.0, {"reason": f"path does not exist: {root}"}

    files = []
    for i, p in enumerate(root.rglob("*")):
        if p.is_file():
            files.append(p)
        if i > max_files_scanned:
            break

    ext_counts = Counter(p.suffix.lower() for p in files)
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    xml_files = [p for p in files if p.suffix.lower() == ".xml"]
    json_files = [p for p in files if p.suffix.lower() == ".json"]
    txt_files = [p for p in files if p.suffix.lower() == ".txt"]

    evidence = {"n_files_scanned": len(files), "n_images": len(image_files),
                "ext_counts": dict(ext_counts.most_common(10))}

    # 1. VOC XML
    if xml_files:
        sample = xml_files[:5]
        hits = sum(1 for xf in sample if _looks_like_voc_xml(_safe_read(xf)))
        if hits >= max(1, len(sample) // 2):
            evidence["voc_hits"] = f"{hits}/{len(sample)}"
            return "voc_xml", "detection_benchmark", 0.9, evidence

    # 2. COCO JSON
    for jf in json_files[:5]:
        data = _safe_json(jf)
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= set(data.keys()):
            anns = data.get("annotations", [])
            has_bbox = any(isinstance(a, dict) and "bbox" in a for a in anns[:20])
            if has_bbox:
                evidence["coco_schema"] = True
                return "coco_json", "detection_benchmark", 0.9, evidence

    # 3. YOLO txt (paired with images, 5-field lines)
    if txt_files and image_files:
        checked, matched = 0, 0
        for tf in txt_files[:20]:
            lines = [l.strip() for l in _safe_read(tf).splitlines() if l.strip()]
            if not lines:
                continue
            checked += 1
            sample_lines = lines[:5]
            if all(len(l.split()) == 5 and all(_is_float(x) for x in l.split()) for l in sample_lines):
                matched += 1
        if checked and matched / checked > 0.6:
            evidence["yolo_txt_ratio"] = f"{matched}/{checked}"
            return "yolo_txt", "detection_benchmark", 0.85, evidence

    # 3b. Plain-text POINT labels, not YOLO (added after a real sandbox test on
    # chula_rbc12's actual GitHub files showed this exact gap: its labels are
    # ".txt" files with lines like "369 429 4" -- x, y, class -- which are 3
    # fields, not YOLO's 5, so rule 3 above correctly rejects them but nothing
    # else caught them either and the detector fell through to abstain. This
    # is the plan's own motivating case, so a v1 that abstains on it is a real,
    # reportable finding -- this rule is the fix, verified against those same
    # real files before being kept.
    if txt_files and image_files:
        checked, point_like = 0, 0
        for tf in txt_files[:20]:
            lines = [l.strip() for l in _safe_read(tf).splitlines() if l.strip()]
            if not lines:
                continue
            checked += 1
            sample_lines = lines[:8]
            if all(2 <= len(l.split()) <= 3 and all(_is_float(x) for x in l.split()) for l in sample_lines):
                point_like += 1
        if checked and point_like / checked > 0.6:
            evidence["point_txt_ratio"] = f"{point_like}/{checked}"
            return "point_labels", "classification_only", 0.65, evidence

    # 3c. Comma-delimited "id,class,comment,shape_type,count,x1,y1,x2,y2,..."
    # annotation lines (added after real sandbox verification against the
    # user's actual malaria "Polygon Set" / "Point Set" files: header line
    # "187,5312,2988" then rows like
    # "1-1,Uninfected,No_comment,Polygon,11,2414.05,1707,...", or
    # "1-1,Parasitized,No_Comment,Point,1,3833.3,306.6"). Neither rule 3
    # (YOLO, whitespace-5-field) nor rule 3b (whitespace point pairs) can see
    # this, because these lines have no whitespace at all -- they are comma-
    # joined, so l.split() treats a whole line as one token. This is a
    # distinct format from chula_rbc12's whitespace-separated point txt, not
    # a duplicate of it. The literal "Polygon"/"Point" field is a strong,
    # explicit signal, so this rule reads it directly rather than inferring
    # shape from coordinate count.
    if txt_files and image_files:
        shape_votes = Counter()
        checked = 0
        for tf in txt_files[:20]:
            lines = [l.strip() for l in _safe_read(tf).splitlines() if l.strip()]
            for line in lines[1:9]:  # skip the header count/width/height line
                fields = line.split(",")
                if len(fields) >= 5 and fields[3].strip().lower() in ("polygon", "point"):
                    shape_votes[fields[3].strip().lower()] += 1
            if lines:
                checked += 1
        total_votes = sum(shape_votes.values())
        if total_votes >= 2:
            dominant, count = shape_votes.most_common(1)[0]
            if count / total_votes > 0.6:
                evidence["comma_shape_votes"] = dict(shape_votes)
                if dominant == "polygon":
                    return "custom_polygon_json", "detection_benchmark", 0.65, evidence
                else:
                    return "point_labels", "classification_only", 0.65, evidence

    # 4. Custom polygon / bbox / point JSON (doesn't match full COCO schema)
    for jf in json_files[:5]:
        text = json.dumps(_safe_json(jf) or {}).lower()[:4000]
        if "polygon" in text:
            evidence["keyword"] = "polygon"
            return "custom_polygon_json", "detection_benchmark", 0.6, evidence
        if "bbox" in text or "bounding" in text:
            evidence["keyword"] = "bbox/bounding"
            return "custom_bbox_json", "detection_benchmark", 0.6, evidence
        if "point" in text and '"x"' in text and '"y"' in text:
            evidence["keyword"] = "point x/y"
            return "point_labels", "classification_only", 0.55, evidence

    # 5. Segmentation masks (filename-based heuristic)
    mask_like = [p for p in files if "mask" in p.name.lower() or "segmentation" in p.name.lower()
                 or p.stem.lower() in {"gt", "groundtruth"}]
    if mask_like and image_files:
        evidence["mask_like_count"] = len(mask_like)
        return "segmentation_masks", "segmentation", 0.7, evidence

    # 6. Classification folders: >=2 subdirs, each mostly images, no annotation files found.
    # v1.3: look for sibling class folders under _find_classification_root(root), not root
    # itself -- see that function's docstring and the v1.3 changelog above for why (the
    # extra_nesting mutation, and real Kaggle zip-wrapper folders, both hide the real class
    # folders one or more empty wrapper levels below root).
    #
    # v1.4: the old guard `not (txt_files and image_files)` blocked this rule if EVEN ONE
    # .txt file existed anywhere in the whole tree alongside any images -- root-caused
    # against aml_cytomorphology_lmu's real files: a single legend file (abbreviations.txt,
    # 1 file) alongside 75+ real .tiff images blocked classification_folders outright, even
    # though the class folders were correctly found. Anything reaching rule 6 has already
    # failed rules 3/3b/3c's specific per-line txt-content checks (YOLO 5-field, point 2-3
    # field, comma-delimited polygon/point), so a small number of incidental txt files
    # (legends, READMEs, license text) shouldn't be treated the same as a real per-image
    # annotation scheme those rules didn't happen to recognize. Replaced the presence check
    # with a ratio check: only treat txt files as a real annotation signal (and block this
    # rule) when they're a substantial fraction of the image count, consistent with genuine
    # per-image pairing (e.g. chula_rbc12: 623 txt vs 738 images, ratio 0.84 -- still
    # correctly blocked here, though rule 3b already catches it earlier in practice).
    txt_to_image_ratio = (len(txt_files) / len(image_files)) if image_files else 0.0
    looks_like_real_txt_annotations = len(txt_files) >= 3 and txt_to_image_ratio >= 0.1
    class_root = _find_classification_root(root)
    subdirs = [d for d in class_root.iterdir() if d.is_dir()] if class_root.is_dir() else []
    if len(subdirs) >= 2 and not xml_files and not json_files and not looks_like_real_txt_annotations:
        class_like = sum(1 for d in subdirs[:30] if any(p.suffix.lower() in IMAGE_EXTS for p in d.rglob("*")))
        if class_like >= max(2, int(len(subdirs) * 0.6)):
            evidence["class_like_subdirs"] = f"{class_like}/{len(subdirs)}"
            if class_root != root:
                evidence["descended_wrapper_dirs"] = str(class_root.relative_to(root))
            return "classification_folders", "classification_only", 0.8, evidence

    # 7. Unlabelled fallback
    if image_files and not (xml_files or json_files or txt_files):
        evidence["reason"] = "images found, no annotation files of any known kind"
        return "unlabelled", "unlabelled", 0.5, evidence

    evidence["reason"] = "no confident signal matched any rule"
    return "abstain", "abstain", 0.0, evidence


def _safe_read(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""


def _safe_json(p: Path):
    try:
        return json.loads(p.read_text(errors="ignore"))
    except Exception:
        return None
