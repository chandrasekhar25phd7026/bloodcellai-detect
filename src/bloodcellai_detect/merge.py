"""
Union-ontology construction and reliable multi-dataset merge export.

WHY THIS MODULE EXISTS
  Two gaps, closed together because they share the same underlying work.

  1. Section 2.2 of the manuscript describes mapping each dataset's native class names onto
     a shared union ontology, but the only worked example given is the specific 2-dataset,
     4-class mapping used by the companion cross-dataset study (Paper 2). Nothing in this
     paper's own Technical Validation independently confirms the mapping mechanism itself --
     that every native class name ends up somewhere, with nothing silently dropped -- across
     the datasets this paper actually covers.
  2. The manuscript's own motivation is that pooling training data from more than one source
     is "a manual, error-prone job" -- but nothing in the paper actually produces a pooled,
     deduplicated, class-harmonized training set as a result. The individual pipeline stages
     (format detection, conversion fidelity, defect detection, duplicate screening) are each
     validated separately; this module runs them together, end to end, on the datasets this
     paper's zoo actually contains, and reports what comes out.

DESIGN PRINCIPLE (same as web/harmonize.js's class discovery)
  A native class name is NEVER silently dropped or silently merged into something else by
  guesswork. It is placed in the union ontology by exactly one of three routes, and every
  placement is in the returned report:
    - IDENTICAL_NAME: the same native name already appears in another dataset -> shares that
      union class automatically.
    - EXPLICIT_OVERRIDE: the caller supplied `(dataset, native_name) -> union_name` in
      `overrides`, stated as a disclosed interpretive choice (e.g. Uninfected -> RBC).
    - UNMATCHED_KEPT_AS_NEW: no identical name and no override -- the native name becomes
      its own union class, flagged for a human to review, rather than guessed into an
      existing one.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .validator import Obj, PerFile

PerFileWithImage = Tuple[str, Path, List[Obj], float, float]  # (file_id, image_path, objects, width, height)


@dataclass
class UnionOntology:
    native_to_union: Dict[Tuple[str, str], str]           # (dataset, native_name) -> union_name
    union_classes: List[str]                                # sorted, contiguous order = export id order
    route_by_native: Dict[Tuple[str, str], str]             # (dataset, native_name) -> one of the 3 routes above
    overrides_applied: Dict[Tuple[str, str], str]            # subset of native_to_union that came from `overrides`
    unused_overrides: Dict[Tuple[str, str], str] = field(default_factory=dict)  # see build_union_ontology

    def union_id(self, union_name: str) -> int:
        return self.union_classes.index(union_name)


def build_union_ontology(datasets: Dict[str, List[PerFile]],
                          overrides: Optional[Dict[Tuple[str, str], str]] = None) -> UnionOntology:
    """datasets: {dataset_name: [(file_id, objects, width, height), ...]}. overrides: optional
    {(dataset_name, native_class_name): union_class_name} for disclosed interpretive choices.
    Every native class name in every dataset ends up mapped by exactly one of the three
    routes named in the module docstring; none is silently dropped."""
    overrides = overrides or {}
    native_by_dataset: Dict[str, Set[str]] = defaultdict(set)
    for ds_name, per_file in datasets.items():
        for _, objs, _, _ in per_file:
            for o in objs:
                native_by_dataset[ds_name].add(o["cls_name"])

    native_to_union: Dict[Tuple[str, str], str] = {}
    route_by_native: Dict[Tuple[str, str], str] = {}
    seen_names: Set[str] = set()

    # Pass 1: explicit overrides take priority and are recorded as such, regardless of name
    # collisions -- but ONLY for a (dataset, native_name) pair that genuinely appears in that
    # dataset's own discovered classes. An override for a class the data doesn't actually
    # contain (for example because that dataset loaded zero objects) is never silently applied
    # as if it were real: it is reported separately in `unused_overrides`, which is exactly the
    # signal that something upstream (a loader, a path) needs checking.
    unused_overrides: Dict[Tuple[str, str], str] = {}
    for (ds_name, native_name), union_name in overrides.items():
        if native_name in native_by_dataset.get(ds_name, set()):
            native_to_union[(ds_name, native_name)] = union_name
            route_by_native[(ds_name, native_name)] = "EXPLICIT_OVERRIDE"
            seen_names.add(union_name)
        else:
            unused_overrides[(ds_name, native_name)] = union_name

    # Pass 2: identical native names across datasets (not already overridden) share a class
    # automatically. Decided from the FULL cross-dataset name count computed up front, not
    # dataset-by-dataset, so that (e.g.) dataset "a" processed before "b" still recognises a
    # name "b" also uses.
    remaining = {(ds, n) for ds in datasets for n in native_by_dataset[ds] if (ds, n) not in native_to_union}
    name_dataset_count: Dict[str, Set[str]] = defaultdict(set)
    for ds, n in remaining:
        name_dataset_count[n].add(ds)
    for ds_name in sorted(datasets):
        for native_name in sorted(native_by_dataset[ds_name]):
            key = (ds_name, native_name)
            if key in native_to_union:
                continue
            shares_with_another_dataset = len(name_dataset_count[native_name]) > 1
            if native_name in seen_names or shares_with_another_dataset:
                native_to_union[key] = native_name
                route_by_native[key] = "IDENTICAL_NAME"
                seen_names.add(native_name)

    # Pass 3: anything left becomes its own new union class, explicitly flagged (never guessed).
    for ds_name in sorted(datasets):
        for native_name in sorted(native_by_dataset[ds_name]):
            key = (ds_name, native_name)
            if key not in native_to_union:
                native_to_union[key] = native_name
                route_by_native[key] = "UNMATCHED_KEPT_AS_NEW"
                seen_names.add(native_name)

    union_classes = sorted({v for v in native_to_union.values()})
    overrides_applied = {k: v for k, v in native_to_union.items() if route_by_native[k] == "EXPLICIT_OVERRIDE"}
    if unused_overrides:
        print(f"build_union_ontology: {len(unused_overrides)} override(s) did not match any "
              f"class actually found in the data -- the dataset likely loaded zero objects, "
              f"or the override has a typo. Not applied: {unused_overrides}")
    return UnionOntology(native_to_union, union_classes, route_by_native, overrides_applied, unused_overrides)


@dataclass
class MergeResult:
    n_source_datasets: int
    n_images_before_dedup: int
    n_images_excluded_as_duplicate: int
    n_images_in_merged_export: int
    n_native_classes_total: int
    n_union_classes: int
    route_counts: Dict[str, int]
    per_dataset_image_counts: Dict[str, int]
    excluded_duplicate_pairs: List[Tuple[str, str]] = field(default_factory=list)
    output_dir: Optional[str] = None


def merge_datasets(datasets: Dict[str, List[PerFileWithImage]], ontology: UnionOntology,
                    duplicate_pairs: Optional[Sequence[Tuple[Tuple[str, str], Tuple[str, str]]]] = None,
                    out_dir: Optional[Path] = None,
                    prefer_dataset_order: Optional[Sequence[str]] = None) -> MergeResult:
    """Merges already-parsed, already-image-located per-file records from multiple datasets into
    one YOLO-format export, excluding images flagged as near-duplicates by
    `bloodcellai_detect.duplicates.scan_datasets` (pass its `all_pairs` -- not `review_sample`,
    which is capped for human review and can under-count real duplicates -- filtered to the
    pairs you've confirmed are real, as `duplicate_pairs`: each item is
    `((dataset_a, path_a), (dataset_b, path_b))`). Writes images/, labels/, data.yaml, and
    provenance.csv (every merged file's original dataset and file_id) if `out_dir` is given;
    always returns the summary counts needed for Section 4.7's table regardless.

    `prefer_dataset_order`: for a confirmed duplicate pair, which side to KEEP is not a
    question this function can answer on its own -- it has no way to know that, say, one
    dataset's annotations were produced by re-labeling the other's (as documented for txl_pbc
    and bccd in Section 4.6.3). Pass dataset names in the order you want them preferred (e.g.
    `["txl_pbc", "bccd"]` to keep txl_pbc's re-annotation over bccd's original whenever they
    duplicate); a dataset earlier in the list is kept over one later in it. Without this
    argument, the only available tiebreak is alphabetical-by-dataset-name, which is disclosed
    here as an uninformed default, not a considered choice -- pass a real preference for any
    case where you have grounds for one (as Section 4.7 now does)."""
    duplicate_pairs = duplicate_pairs or []
    preference_rank = {name: i for i, name in enumerate(prefer_dataset_order)} if prefer_dataset_order else {}
    excluded_paths: Set[Tuple[str, str]] = set()
    for (ds_a, path_a), (ds_b, path_b) in duplicate_pairs:
        a, b = (ds_a, path_a), (ds_b, path_b)
        if preference_rank:
            # lower rank = more preferred = kept; unlisted datasets rank last (least preferred)
            rank_a = preference_rank.get(ds_a, len(preference_rank))
            rank_b = preference_rank.get(ds_b, len(preference_rank))
            drop = b if rank_a <= rank_b else a
        else:
            # keep the first occurrence (by dataset name, then file path) of each duplicate pair, drop the rest
            drop = sorted([a, b])[1]
        excluded_paths.add(drop)

    if out_dir:
        out_dir = Path(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    provenance_rows = []
    n_before, n_excluded, n_final = 0, 0, 0

    for ds_name, per_file in sorted(datasets.items()):
        for file_id, image_path, objs, w, h in per_file:
            n_before += 1
            if (ds_name, str(image_path)) in excluded_paths:
                n_excluded += 1
                continue
            n_final += 1
            output_id = f"{ds_name}__{Path(file_id).stem}"
            lines = []
            for o in objs:
                union_name = ontology.native_to_union[(ds_name, o["cls_name"])]
                cid = ontology.union_id(union_name)
                xc, yc = ((o["x1"] + o["x2"]) / 2) / w, ((o["y1"] + o["y2"]) / 2) / h
                bw, bh = (o["x2"] - o["x1"]) / w, (o["y2"] - o["y1"]) / h
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            row = dict(output_id=output_id, source_dataset=ds_name, source_file_id=file_id,
                       source_image_path=str(image_path), n_objects=len(objs),
                       image_sha256="", label_sha256="")
            if out_dir:
                ext = Path(image_path).suffix or ".jpg"
                image_out = out_dir / "images" / f"{output_id}{ext}"
                label_out = out_dir / "labels" / f"{output_id}.txt"
                shutil.copy2(image_path, image_out)
                label_text = "\n".join(lines) + ("\n" if lines else "")
                label_out.write_text(label_text)
                row["image_sha256"] = hashlib.sha256(image_out.read_bytes()).hexdigest()
                row["label_sha256"] = hashlib.sha256(label_text.encode()).hexdigest()
            provenance_rows.append(row)

    if out_dir:
        (out_dir / "data.yaml").write_text(
            "train: images\nval: images\ntest: images\n"
            f"nc: {len(ontology.union_classes)}\n"
            f"names: [{', '.join(json.dumps(n) for n in ontology.union_classes)}]\n"
        )
        with open(out_dir / "provenance.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=["output_id", "source_dataset", "source_file_id",
                                               "source_image_path", "n_objects",
                                               "image_sha256", "label_sha256"])
            w_.writeheader()
            w_.writerows(provenance_rows)
        # self-consistency check: every label file has a matching image and vice versa
        img_stems = {p.stem for p in (out_dir / "images").glob("*")}
        lbl_stems = {p.stem for p in (out_dir / "labels").glob("*")}
        assert img_stems == lbl_stems, f"merge export is NOT self-consistent: {img_stems ^ lbl_stems}"

    route_counts = defaultdict(int)
    for route in ontology.route_by_native.values():
        route_counts[route] += 1
    n_native_total = len(ontology.route_by_native)

    return MergeResult(
        n_source_datasets=len(datasets), n_images_before_dedup=n_before,
        n_images_excluded_as_duplicate=n_excluded, n_images_in_merged_export=n_final,
        n_native_classes_total=n_native_total, n_union_classes=len(ontology.union_classes),
        route_counts=dict(route_counts),
        per_dataset_image_counts={k: len(v) for k, v in datasets.items()},
        excluded_duplicate_pairs=[(a[1], b[1]) for a, b in duplicate_pairs],
        output_dir=str(out_dir) if out_dir else None,
    )


# ============================================================================
# Role-aware merging: point this at a folder of dataset folders, and it
# merges each role-group the way that role needs. Built to close the gap
# between "detects role" (Section 4.1) and "only demonstrated merging for
# detection-role datasets" (Section 4.7's first version).
# ============================================================================

def load_classification_folder(root, images_root=None) -> List[PerFileWithImage]:
    """Generic loader for any dataset this project's detector calls
    'classification_folders': one image per file, its class = its immediate
    parent folder name, no real box -- represented as a single full-frame
    object (x1=y1=0, x2=width, y2=height), exactly as web/harmonize.js's
    classification handling already does. This is why classification merging
    needs no new core logic: `merge_datasets` already treats "one object per
    image, box = the whole frame" as an ordinary case."""
    from PIL import Image
    root = Path(root)
    out = []
    for img_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in
                            {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
        parent = img_path.parent.name
        if parent == root.name:
            continue  # an image directly in the dataset root, not inside any class folder
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        objs = [dict(cls_name=parent, x1=0, y1=0, x2=float(w), y2=float(h))]
        out.append((str(img_path.relative_to(root)), img_path, objs, float(w), float(h)))
    return out


@dataclass
class SegmentationMergeResult:
    n_source_datasets: int
    n_pairs_before_dedup: int
    n_pairs_excluded_as_duplicate: int
    n_pairs_in_merged_export: int
    per_dataset_pair_counts: Dict[str, int]
    mask_semantics_reconciled: bool = False  # always False here -- see module docstring
    output_dir: Optional[str] = None


def merge_segmentation_datasets(datasets: Dict[str, List[Tuple[str, Path, Path]]],
                                 duplicate_pairs: Optional[Sequence] = None,
                                 out_dir: Optional[Path] = None) -> SegmentationMergeResult:
    """datasets: {dataset_name: [(file_id, image_path, mask_path), ...]}. Pools image/mask
    pairs from multiple segmentation-role datasets into one images/ + masks/ + provenance.csv
    export, excluding pairs you've confirmed are duplicates (pass the near-duplicate screen's
    candidates, reviewed by eye, the same way as the detection merge).

    WHAT THIS DELIBERATELY DOES NOT DO: it does not attempt to reconcile different datasets'
    mask pixel-value conventions (e.g. one dataset's mask marking "nucleus" as pixel value 255,
    another's marking it as 1). Unlike a class NAME, a mask's pixel semantics cannot be read
    from a file name or a directory structure -- confirming it needs looking at the actual
    pixel values against each dataset's own documentation, which this function cannot do
    safely on your behalf. Masks are pooled as-is, with a per-dataset manifest, so you can
    check and reconcile this yourself before using the pooled masks for anything -- this is a
    disclosed limitation, not a silent assumption that every dataset's masks already agree."""
    duplicate_pairs = duplicate_pairs or []
    excluded = set()
    for (ds_a, path_a), (ds_b, path_b) in duplicate_pairs:
        pair = sorted([(ds_a, path_a), (ds_b, path_b)])
        excluded.add(tuple(pair[1]))

    if out_dir:
        out_dir = Path(out_dir)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows, n_before, n_excluded, n_final = [], 0, 0, 0
    for ds_name, per_file in sorted(datasets.items()):
        for file_id, image_path, mask_path in per_file:
            n_before += 1
            if (ds_name, str(image_path)) in excluded:
                n_excluded += 1
                continue
            n_final += 1
            output_id = f"{ds_name}__{Path(file_id).stem}"
            rows.append(dict(output_id=output_id, source_dataset=ds_name, source_file_id=file_id,
                              source_image_path=str(image_path), source_mask_path=str(mask_path)))
            if out_dir:
                shutil.copy2(image_path, out_dir / "images" / f"{output_id}{Path(image_path).suffix}")
                shutil.copy2(mask_path, out_dir / "masks" / f"{output_id}{Path(mask_path).suffix}")

    if out_dir:
        with open(out_dir / "provenance.csv", "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=["output_id", "source_dataset", "source_file_id",
                                               "source_image_path", "source_mask_path"])
            w_.writeheader()
            w_.writerows(rows)
        img_stems = {p.stem for p in (out_dir / "images").glob("*")}
        msk_stems = {p.stem for p in (out_dir / "masks").glob("*")}
        assert img_stems == msk_stems, f"segmentation merge export is NOT self-consistent: {img_stems ^ msk_stems}"

    return SegmentationMergeResult(
        n_source_datasets=len(datasets), n_pairs_before_dedup=n_before,
        n_pairs_excluded_as_duplicate=n_excluded, n_pairs_in_merged_export=n_final,
        per_dataset_pair_counts={k: len(v) for k, v in datasets.items()},
        output_dir=str(out_dir) if out_dir else None,
    )


def group_by_role(dataset_roots: Dict[str, Path], detector_version: str = "1.4") -> Dict[str, Dict[str, Path]]:
    """Runs this project's own format/role detector (Section 4.1) on every dataset root and
    groups them by detected role. Returns {role: {dataset_name: path}}; a dataset the detector
    abstains on, or calls "unlabelled", is placed under that label too -- never silently
    dropped, same policy as the detector itself. Needs the `bloodcellai_detect.detector`
    module (imported lazily here to avoid a hard dependency for callers who only need the
    merge logic on roles they've already determined themselves)."""
    from .detector import detect_format_and_role
    groups: Dict[str, Dict[str, Path]] = {}
    evidence_by_dataset = {}
    for name, root in dataset_roots.items():
        fmt, role, confidence, evidence = detect_format_and_role(Path(root), version=detector_version)
        groups.setdefault(role, {})[name] = Path(root)
        evidence_by_dataset[name] = dict(format=fmt, role=role, confidence=confidence)
    return groups, evidence_by_dataset


# ============================================================================
# Machine-readable mapping audit, checksummed provenance, and export-integrity
# audit. Added in response to external review: a class-mapping table needs to
# be inspectable and re-derivable by anyone, not just asserted in prose.
# ============================================================================

# Short, fixed rationale text for each kind of mapping this project has actually
# made. Keyed by (route, union_name) so it stays tied to a real, already-stated
# decision (Section 2.2, Section 4.6.3, Section 4.7) rather than invented per row.
_MAPPING_RATIONALE = {
    ("IDENTICAL_NAME", None): "Same literal label text in both datasets; no interpretive choice was made.",
    ("EXPLICIT_OVERRIDE", "RBC"): "Native label names red blood cells using different wording; no subtype or status distinction is collapsed.",
    ("EXPLICIT_OVERRIDE", "WBC"): "Native label names white blood cells/leukocytes using different wording; no subtype distinction is collapsed.",
    ("EXPLICIT_OVERRIDE", "Parasite"): "Infected-stage or parasitized-cell label grouped into one Parasite class; this is an interpretive choice that discards life-cycle-stage or infected/uninfected-RBC distinctions the source dataset itself records.",
    ("UNMATCHED_KEPT_AS_NEW", None): "No identical native label in another dataset and no explicit override given; kept as its own union class rather than guessed into an existing one.",
}


def export_class_mapping_audit(ontology: "UnionOntology", native_object_counts: Dict[Tuple[str, str], int],
                                out_path: Path) -> None:
    """Writes the union ontology as a machine-readable CSV: one row per native class,
    with the columns an external review specifically asked for (source dataset, native
    label, canonical label, mapping type, information loss, rationale), plus the real
    object count for that native label so a reader can verify nothing was silently
    dropped. `native_object_counts`: {(dataset, native_label): object_count}, computed
    from the same parsed per-file data passed to build_union_ontology -- BEFORE any
    duplicate-image exclusion, so this reflects mapping integrity specifically, not the
    separate, already-disclosed effect of the duplicate screen (Section 4.4/4.7)."""
    rows = []
    for (ds, native), union in sorted(ontology.native_to_union.items()):
        route = ontology.route_by_native[(ds, native)]
        is_lossy = (route == "EXPLICIT_OVERRIDE" and union == "Parasite")
        key = (route, union) if route == "EXPLICIT_OVERRIDE" else (route, None)
        rationale = _MAPPING_RATIONALE.get(key, _MAPPING_RATIONALE.get((route, None), ""))
        if route == "IDENTICAL_NAME":
            mapping_type = "exact_match"
        elif route == "UNMATCHED_KEPT_AS_NEW":
            mapping_type = "unmatched_new_class"
        elif is_lossy:
            mapping_type = "semantic_override"
        else:
            mapping_type = "synonym"
        rows.append(dict(
            source_dataset=ds, native_label=native, canonical_label=union,
            mapping_type=mapping_type,
            information_loss=("yes" if is_lossy else "no"),
            rationale=rationale,
            native_object_count=native_object_counts.get((ds, native), ""),
        ))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_dataset", "native_label", "canonical_label",
                                          "mapping_type", "information_loss", "rationale",
                                          "native_object_count"])
        w.writeheader()
        w.writerows(rows)


def verify_class_counts(ontology: "UnionOntology", native_object_counts: Dict[Tuple[str, str], int]
                         ) -> Dict[str, int]:
    """Sums native_object_counts (computed before any duplicate exclusion) into union
    classes via the ontology's own mapping, so the total objects per native label can
    be checked against the total objects per union label -- nothing should be lost or
    gained at the mapping step itself (duplicate-image exclusion, a separate, later,
    already-disclosed step, is NOT reflected here; it removes whole images and their
    objects for a documented reason, not a mapping failure)."""
    union_counts: Dict[str, int] = defaultdict(int)
    for (ds, native), union in ontology.native_to_union.items():
        union_counts[union] += native_object_counts.get((ds, native), 0)
    total_native = sum(native_object_counts.values())
    total_union = sum(union_counts.values())
    assert total_native == total_union, (
        f"class-count mismatch at the mapping step: {total_native} native objects vs "
        f"{total_union} after mapping to union classes -- this should never happen and "
        f"indicates a real bug, not an expected difference.")
    return dict(union_counts)


def audit_export(out_dir: Path) -> Dict[str, object]:
    """Automated check on an already-written merge_datasets() export: every image has
    exactly one label file and vice versa, and every class id in every label file is a
    valid index into the union ontology (0 <= id < n_union_classes). Returns a dict
    with pass/fail counts and the first few concrete failures, if any -- never just a
    boolean, so a real problem is never reported as a bare 'failed'."""
    out_dir = Path(out_dir)
    image_stems = {p.stem for p in (out_dir / "images").glob("*") if p.is_file()}
    label_stems = {p.stem for p in (out_dir / "labels").glob("*") if p.is_file()}
    images_without_label = sorted(image_stems - label_stems)
    labels_without_image = sorted(label_stems - image_stems)


    metadata_errors = []
    try:
        data_yaml_text = (out_dir / "data.yaml").read_text(encoding="utf-8")
        nc_line = next(
            line for line in data_yaml_text.splitlines()
            if line.strip().startswith("nc:")
        )
        n_union_classes = int(nc_line.split(":", 1)[1].strip())
        if n_union_classes < 0:
            raise ValueError("Class count cannot be negative")
    except (OSError, StopIteration, ValueError) as exc:
        n_union_classes = None
        metadata_errors.append(
            f"data.yaml: unable to determine valid class count: {exc}"
        )


    bad_class_ids = []
    n_lines_checked = 0
    for label_path in sorted((out_dir / "labels").glob("*")):
        for line_no, line in enumerate(open(label_path), start=1):
            line = line.strip()
            if not line:
                continue
            n_lines_checked += 1
            parts = line.split()
            if len(parts) != 5:
                bad_class_ids.append(f"{label_path.name}:{line_no}: malformed line (expected 5 fields, got {len(parts)})")
                continue
            cid = int(float(parts[0]))
            if n_union_classes is not None and not (0 <= cid < n_union_classes):
                bad_class_ids.append(f"{label_path.name}:{line_no}: class id {cid} out of range [0, {n_union_classes})")

    return dict(
        n_images=len(image_stems), n_labels=len(label_stems), metadata_errors=metadata_errors,
        images_without_label=images_without_label, labels_without_image=labels_without_image,
        n_label_lines_checked=n_lines_checked, bad_class_id_lines=bad_class_ids[:20],
        n_bad_class_id_lines_total=len(bad_class_ids),
        passed=(
            not images_without_label
            and not labels_without_image
            and not bad_class_ids
            and not metadata_errors
        ),
    )
