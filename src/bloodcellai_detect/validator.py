"""
Structural annotation validator and defect-injection benchmark.

PROVENANCE
  Ported from the Kaggle notebook `bdqi-v2`, cell 279 (`%writefile
  paperA_e4_defect_detection.py`), the script behind Section 4.3 of the
  manuscript (Tables 7 and 8: structurally detectable vs. structurally
  undetectable defect types). The core logic (`validate_annotations`,
  `inject_defects`, `score_defect_type`) is unchanged; the dataset-specific
  driver (`run_dataset`, which hardcoded BCCD and malaria_polygon parsing
  and Kaggle paths) is generalized below into `run_defect_benchmark`, which
  takes already-parsed object lists keyed by dataset name instead of
  hardcoding which two datasets exist or where they live. The VOC XML and
  malaria-polygon parsers are kept, unchanged, as convenience readers.

WHAT THIS DOES AND DOES NOT CHECK
  The validator is geometry- and duplication-only: it never looks at image
  content. It flags a box as "zero_area" (collapsed width or height),
  "out_of_bounds" (lies outside the image), or "duplicate" (another object
  with the same class and the same rounded coordinates). A class label that
  is simply wrong, or a box shifted by a small amount that stays in bounds,
  looks structurally identical to a correct one -- the benchmark below
  reports this as a disclosed limitation (STRUCTURALLY_UNDETECTABLE), not a
  bug to silently work around.
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Obj = Dict[str, object]  # {"cls_name": str, "x1": float, "y1": float, "x2": float, "y2": float}

DEFECT_TYPES = ("zero_area", "off_image", "duplicate_line", "class_flip", "shifted_box")
STRUCTURALLY_UNDETECTABLE = {"class_flip", "shifted_box"}

_VALIDATOR_LABEL_FOR_DEFECT = {
    "zero_area": "zero_area",
    "off_image": "out_of_bounds",
    "duplicate_line": "duplicate",
    "class_flip": None,   # structurally undetectable by design
    "shifted_box": None,  # structurally undetectable by design
}


# --------------------------------------------------------------------------- parsers (convenience readers)
def parse_voc_xml(path) -> Tuple[List[Obj], Tuple[float, float]]:
    """Reads a Pascal VOC XML annotation file. Returns (objects, (width, height))."""
    root = ET.parse(path).getroot()
    sz = root.find("size")
    w, h = float(sz.findtext("width")), float(sz.findtext("height"))
    objs: List[Obj] = []
    for o in root.findall("object"):
        name = o.findtext("name")
        bb = o.find("bndbox")
        try:
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            continue
        objs.append(dict(cls_name=name, x1=x1, y1=y1, x2=x2, y2=y2))
    return objs, (w, h)


def parse_malaria_polygon(path) -> Tuple[List[Obj], Tuple[float, float]]:
    """Reads this project's comma-delimited polygon annotation format (header line
    `<id>,<width>,<height>`, then one `...,Polygon,<n>,x1,y1,x2,y2,...` line per object),
    reducing each polygon to its axis-aligned bounding box."""
    lines = [l for l in Path(path).read_text(errors="ignore").splitlines() if l.strip()]
    head = [x.strip() for x in lines[0].split(",")]
    w, h = float(head[1]), float(head[2])
    objs: List[Obj] = []
    for l in lines[1:]:
        f = [x.strip() for x in l.split(",")]
        if len(f) < 7 or f[3].lower() != "polygon":
            continue
        label = f[1]
        try:
            nums = [float(x) for x in f[5:]]
        except ValueError:
            continue
        pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        if len(pts) < 3:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        objs.append(dict(cls_name=label, x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return objs, (w, h)


# --------------------------------------------------------------------------- defect injection (the answer key)
def inject_defects(objs: Sequence[Obj], w: float, h: float, defect_type: str, rate: float,
                    rng: random.Random, all_class_names: Sequence[str]) -> Tuple[List[Obj], List[Optional[str]]]:
    """Returns (defected_objs, injection_log). injection_log[i] is the defect type applied
    to defected_objs[i], or None. Never mutates the input list."""
    objs = [dict(o) for o in objs]
    n = len(objs)
    k = max(1, round(n * rate)) if n > 0 else 0
    idxs = rng.sample(range(n), min(k, n)) if n > 0 else []
    log: List[Optional[str]] = [None] * n

    extra_objs: List[Tuple[Obj, str]] = []  # duplicate_line ADDS an object rather than mutating one
    for i in idxs:
        o = objs[i]
        if defect_type == "zero_area":
            o["x2"] = o["x1"]
            log[i] = "zero_area"
        elif defect_type == "off_image":
            width = o["x2"] - o["x1"]
            o["x1"] = w + 10.0
            o["x2"] = o["x1"] + max(width, 1.0)
            log[i] = "off_image"
        elif defect_type == "duplicate_line":
            extra_objs.append((dict(o), "duplicate_line"))
            # Both halves of the pair are the defect: the validator correctly flags both
            # the original and its duplicate as "duplicate" (duplication is a property of
            # the pair), so the answer key marks the original too.
            log[i] = "duplicate_line"
        elif defect_type == "class_flip":
            other = [c for c in all_class_names if c != o["cls_name"]]
            if other:
                o["cls_name"] = rng.choice(other)
                log[i] = "class_flip"
        elif defect_type == "shifted_box":
            bw, bh = o["x2"] - o["x1"], o["y2"] - o["y1"]
            max_dx = max(1.0, min(bw, w - bw) * 0.3)
            max_dy = max(1.0, min(bh, h - bh) * 0.3)
            dx, dy = rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)
            nx1, ny1, nx2, ny2 = o["x1"] + dx, o["y1"] + dy, o["x2"] + dx, o["y2"] + dy
            if 0 <= nx1 and nx2 <= w and 0 <= ny1 and ny2 <= h:
                o["x1"], o["y1"], o["x2"], o["y2"] = nx1, ny1, nx2, ny2
                log[i] = "shifted_box"
            # else: shift would push off-image -- skip; only genuinely in-bounds shifts count.
        else:
            raise ValueError(defect_type)

    result_objs = objs + [o for o, _ in extra_objs]
    result_log = log + [t for _, t in extra_objs]
    return result_objs, result_log


# --------------------------------------------------------------------------- the validator under test
def validate_annotations(objs: Sequence[Obj], w: float, h: float, tol: float = 1.0) -> List[Set[str]]:
    """Structural-only validator (no image content, no semantic reasoning). Returns one
    flag-set per object: a subset of {"zero_area", "out_of_bounds", "duplicate"}."""
    flags: List[Set[str]] = [set() for _ in objs]
    seen: Dict[tuple, int] = {}
    for i, o in enumerate(objs):
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        if not (x2 > x1 and y2 > y1):
            flags[i].add("zero_area")
        if x1 < -tol or y1 < -tol or x2 > w + tol or y2 > h + tol:
            flags[i].add("out_of_bounds")
        key = (o["cls_name"], round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        if key in seen:
            flags[i].add("duplicate")
            flags[seen[key]].add("duplicate")
        else:
            seen[key] = i
    return flags


def score_defect_type(objs: Sequence[Obj], w: float, h: float, injection_log: Sequence[Optional[str]],
                       defect_type: str) -> Dict[str, object]:
    """Precision/recall/F1 for one defect type's injection run."""
    flags = validate_annotations(objs, w, h)
    validator_label = _VALIDATOR_LABEL_FOR_DEFECT[defect_type]
    injected_idx = {i for i, t in enumerate(injection_log) if t == defect_type}
    flagged_idx = set() if validator_label is None else {i for i, f in enumerate(flags) if validator_label in f}
    tp, fp, fn = len(injected_idx & flagged_idx), len(flagged_idx - injected_idx), len(injected_idx - flagged_idx)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if (recall is not None and (precision + recall) > 0) \
        else (0.0 if recall == 0 else None)
    return dict(defect_type=defect_type, structurally_undetectable=(validator_label is None),
                n_injected=len(injected_idx), tp=tp, fp=fp, fn=fn,
                precision=round(precision, 4), recall=(round(recall, 4) if recall is not None else None),
                f1=(round(f1, 4) if f1 is not None else None))


# --------------------------------------------------------------------------- benchmark driver (generalized)
PerFile = Tuple[str, List[Obj], float, float]  # (file_id, objects, width, height)


def run_defect_benchmark(datasets: Dict[str, List[PerFile]], injection_rates: Sequence[float] = (0.01, 0.05, 0.10),
                          seed: int = 20260926) -> Dict[str, object]:
    """Runs the full injection+scoring grid (defect_type x rate) on each dataset's already-
    parsed objects. `datasets` maps a dataset name to a list of (file_id, objects, width,
    height) tuples -- this package does no dataset-specific parsing or path handling; use
    `parse_voc_xml` / `parse_malaria_polygon` (or your own parser) to build that list first.
    Mirrors `run_dataset` from the original script, generalized beyond BCCD/malaria_polygon."""
    rng = random.Random(seed)
    results = {}
    for name, per_file in datasets.items():
        all_class_names = sorted({o["cls_name"] for _, objs, _, _ in per_file for o in objs})
        n_clean_objects = sum(len(objs) for _, objs, _, _ in per_file)

        results_by_rate = {}
        for rate in injection_rates:
            per_type = {dt: dict(tp=0, fp=0, fn=0, n_injected=0) for dt in DEFECT_TYPES}
            for defect_type in DEFECT_TYPES:
                for _, objs, w, h in per_file:
                    defected, log = inject_defects(objs, w, h, defect_type, rate, rng, all_class_names)
                    s = score_defect_type(defected, w, h, log, defect_type)
                    for k in ("tp", "fp", "fn", "n_injected"):
                        per_type[defect_type][k] += s[k]
            rate_summary = {}
            for dt, agg in per_type.items():
                tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
                precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
                f1 = (2 * precision * recall / (precision + recall)) if (recall is not None and (precision + recall) > 0) \
                    else (0.0 if recall == 0 else None)
                rate_summary[dt] = dict(structurally_undetectable=(dt in STRUCTURALLY_UNDETECTABLE),
                                         n_injected=agg["n_injected"], tp=tp, fp=fp, fn=fn,
                                         precision=round(precision, 4),
                                         recall=(round(recall, 4) if recall is not None else None),
                                         f1=(round(f1, 4) if f1 is not None else None))
            results_by_rate[rate] = rate_summary

        clean_flag_counts: Counter = Counter()
        for _, objs, w, h in per_file:
            for f in validate_annotations(objs, w, h):
                for label in f:
                    clean_flag_counts[label] += 1

        results[name] = dict(dataset=name, n_files=len(per_file), n_clean_objects=n_clean_objects,
                              classes=all_class_names, false_alarms_on_clean_data=dict(clean_flag_counts),
                              results_by_injection_rate={str(r): v for r, v in results_by_rate.items()})
    return results


def check_real_defects(per_file: List[PerFile]) -> Dict[str, object]:
    """Re-confirms the validator catches REAL (not injected) defects already present in a
    dataset's own original annotations, independent of the synthetic injection study above."""
    n_raw, n_flagged_zero_area, examples = 0, 0, []
    for file_id, objs, w, h in per_file:
        n_raw += len(objs)
        for i, f in enumerate(validate_annotations(objs, w, h)):
            if "zero_area" in f:
                n_flagged_zero_area += 1
                examples.append(dict(file=file_id, obj_index=i, cls_name=objs[i]["cls_name"],
                                      box=[objs[i]["x1"], objs[i]["y1"], objs[i]["x2"], objs[i]["y2"]]))
    return dict(n_raw_objects=n_raw, n_real_defects_found=n_flagged_zero_area, examples=examples)
