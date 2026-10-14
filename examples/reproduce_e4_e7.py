"""
Reproduce the manuscript's Section 4.3 (structural defect validator, Tables 7-8) and
Section 4.4 (near-duplicate/leakage screen, Tables 9-10) on your own copies of the real
datasets. This script does no path discovery or downloading -- point the two constants
below at your own BCCD and malaria_polygon folders (for the defect benchmark), and use
your own `bloodcellai-detect lock`-ed ground-truth CSV for the duplicate screen.

Usage:
    python reproduce_e4_e7.py defects   /path/to/bccd /path/to/malaria_polygon
    python reproduce_e4_e7.py duplicates /path/to/locked_ground_truth.csv
"""
import csv
import json
import sys
from pathlib import Path

from bloodcellai_detect.validator import (
    parse_voc_xml, parse_malaria_polygon, run_defect_benchmark, check_real_defects,
)


def load_bccd(root: Path):
    root = Path(root)
    ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
    per_file = []
    for p in sorted(ann_dir.glob("*.xml")):
        if not (img_dir / f"{p.stem}.jpg").exists():
            continue
        objs, (w, h) = parse_voc_xml(p)
        objs = [o for o in objs if o["x2"] > o["x1"] and o["y2"] > o["y1"]]  # clean subset for injection
        if objs:
            per_file.append((p.stem, objs, w, h))
    return per_file


def load_malaria_polygon(root: Path):
    per_file = []
    for p in sorted(Path(root).rglob("*.txt")):
        objs, (w, h) = parse_malaria_polygon(p)
        if objs:
            per_file.append((p.stem, objs, w, h))
    return per_file


def run_defects(bccd_root: str, malaria_root: str, out_dir: str = "e4_defect_detection"):
    bccd = load_bccd(bccd_root)
    malaria = load_malaria_polygon(malaria_root)
    results = run_defect_benchmark({"bccd": bccd, "malaria_polygon": malaria})
    real_defects = check_real_defects(bccd)  # BCCD's 2 real, not-injected zero-area boxes

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table_defect_detection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "injection_rate", "defect_type", "structurally_undetectable",
                    "n_injected", "tp", "fp", "fn", "precision", "recall", "f1"])
        for ds_name, r in results.items():
            for rate_str, per_type in r["results_by_injection_rate"].items():
                for dt, s in per_type.items():
                    w.writerow([ds_name, rate_str, dt, s["structurally_undetectable"],
                                s["n_injected"], s["tp"], s["fp"], s["fn"], s["precision"], s["recall"], s["f1"]])
    (out / "real_defects_bccd.json").write_text(json.dumps(real_defects, indent=2))
    print(json.dumps(results, indent=2, default=str))
    print(json.dumps(real_defects, indent=2))
    print(f"\nWrote {out}/table_defect_detection.csv and {out}/real_defects_bccd.json")


def run_duplicates(gt_csv: str, out_dir: str = "e7_duplicates_leakage"):
    from bloodcellai_detect.duplicates import scan_datasets_from_locked_gt
    result = scan_datasets_from_locked_gt(Path(gt_csv))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table_near_duplicate_review_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_a", "path_a", "dataset_b", "path_b", "hamming_distance", "scope"])
        for row in result["review_sample"]:
            w.writerow([row["dataset_a"], row["path_a"], row["dataset_b"], row["path_b"],
                        row["hamming_distance"], row["scope"]])
    (out / "scan_summary.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "review_sample"}, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "review_sample"}, indent=2, default=str))
    print(f"\nReview sample: {len(result['review_sample'])} rows")
    print(f"Wrote {out}/table_near_duplicate_review_sample.csv and {out}/scan_summary.json")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("defects", "duplicates"):
        sys.exit(__doc__)
    if sys.argv[1] == "defects":
        if len(sys.argv) != 4:
            sys.exit("usage: python reproduce_e4_e7.py defects /path/to/bccd /path/to/malaria_polygon")
        run_defects(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) != 3:
            sys.exit("usage: python reproduce_e4_e7.py duplicates /path/to/locked_ground_truth.csv")
        run_duplicates(sys.argv[2])
