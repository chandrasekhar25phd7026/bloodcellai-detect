"""Ground-truth locking (ported from `paperA_e1_lock_ground_truth.py`, notebook cell 225).

Discipline: the true format and role of every dataset are written down BEFORE the
detector is run, hashed (sha256) and locked. The audit refuses to run if the file
changes afterwards. An existing lock is never overwritten.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

VALID_FORMATS = {
    "voc_xml", "coco_json", "yolo_txt", "custom_polygon_json",
    "custom_bbox_json", "point_labels", "classification_folders",
    "segmentation_masks", "mixed", "unlabelled",
}
VALID_ROLES = {"detection_benchmark", "classification_only", "segmentation", "unlabelled"}
REQUIRED_COLUMNS = {
    "dataset_id", "dataset_name", "source_url", "on_kaggle", "kaggle_path_hint",
    "true_format", "true_role", "n_images_expected", "n_classes_expected",
    "license", "notes", "filled_by", "fill_date",
}
PLACEHOLDER_FILLED_BY = {"", "DRAFT_FROM_VERIFIED_RESEARCH", "FILL_YOUR_NAME"}
UNSET_PATHS = {"", "FILL_ME_IN"}   # NOT_YET_ATTACHED is a deliberate, recorded decision


def lock_path_for(gt_csv: Path) -> Path:
    gt_csv = Path(gt_csv)
    return gt_csv.with_suffix(".lock.json")


def validate(rows):
    errors, seen = [], set()
    for i, row in enumerate(rows, start=2):
        rid = row.get("dataset_id", "").strip()
        if not rid:
            errors.append(f"line {i}: empty dataset_id")
        if rid in seen:
            errors.append(f"line {i}: duplicate dataset_id '{rid}'")
        seen.add(rid)
        if row.get("true_format", "").strip() not in VALID_FORMATS:
            errors.append(f"line {i} ({rid}): true_format not in taxonomy {sorted(VALID_FORMATS)}")
        if row.get("true_role", "").strip() not in VALID_ROLES:
            errors.append(f"line {i} ({rid}): true_role not in taxonomy {sorted(VALID_ROLES)}")
        if row.get("kaggle_path_hint", "").strip() in UNSET_PATHS:
            errors.append(f"line {i} ({rid}): kaggle_path_hint is unset. Fill in the real path, or write "
                          f"NOT_YET_ATTACHED if the dataset is deliberately not tested yet.")
        if row.get("filled_by", "").strip() in PLACEHOLDER_FILLED_BY:
            errors.append(f"line {i} ({rid}): filled_by is a placeholder. Put your own name once you have "
                          f"checked this row against the real folder or the dataset documentation.")
    return errors


def read_rows(gt_csv: Path):
    gt_csv = Path(gt_csv)
    with open(gt_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Ground-truth CSV is missing required columns: {sorted(missing)}")
        return list(reader)


def lock(gt_csv: Path):
    gt_csv = Path(gt_csv)
    lp = lock_path_for(gt_csv)
    raw = gt_csv.read_bytes()
    rows = read_rows(gt_csv)
    errors = validate(rows)
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} problem(s). Fix before locking:\n")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    digest = hashlib.sha256(raw).hexdigest()
    if lp.exists():
        existing = json.loads(lp.read_text())
        if existing["sha256"] == digest:
            print("Lock already exists and matches the current file exactly; nothing to do.")
            return existing
        sys.exit(f"REFUSING TO OVERWRITE existing lock {lp}: its hash differs from the current file. "
                 f"If the change is an intentional correction, keep the old lock under a versioned name.")
    data = {
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ground_truth_file": str(gt_csv), "sha256": digest, "n_rows": len(rows),
        "dataset_ids": sorted(r["dataset_id"] for r in rows),
        "filled_by": sorted({r["filled_by"] for r in rows}),
    }
    lp.write_text(json.dumps(data, indent=2))
    print(f"LOCKED {len(rows)} rows, sha256={digest[:16]}...")
    return data


def verify_lock_or_die(gt_csv: Path):
    gt_csv = Path(gt_csv)
    lp = lock_path_for(gt_csv)
    if not lp.exists():
        sys.exit(f"No ground-truth lock at {lp}. Run `bloodcellai-detect lock --gt {gt_csv}` first.")
    data = json.loads(lp.read_text())
    if hashlib.sha256(gt_csv.read_bytes()).hexdigest() != data["sha256"]:
        sys.exit("GROUND TRUTH HAS CHANGED SINCE LOCKING; refusing to run the audit. "
                 "Re-lock deliberately under a new version name if the change was intentional.")
    return data
