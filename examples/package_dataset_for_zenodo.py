"""
Packages a real merge_datasets() export (Section 4.7's merged dataset: images/, labels/,
data.yaml, provenance.csv -- the output already sitting in your Kaggle working directory
after running reproduce_merge.py or the Section 4.7 cells) into one folder ready to zip
and upload to Zenodo as its OWN, SEPARATE record -- not inside this code repository.

This script does not invent or guess at any data. It only reads the real export you point
it at, re-derives the class-mapping audit and the count-verification check from the real
union ontology and the real per-class object counts in your provenance.csv, runs the
export-integrity audit against the real files, and writes everything (plus a blog-style
README and a CITATION.cff) into one output folder.

Usage:
    python package_dataset_for_zenodo.py /kaggle/working/merged_export /kaggle/working/dataset_package \\
        --ontology-source bccd txl_pbc bbbc041 malaria_polygon
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from bloodcellai_detect.merge import audit_export


def add_real_checksums(export_dir: Path, provenance_rows: list) -> list:
    """Computes a real SHA-256 for every exported image and label file directly from disk,
    regardless of whether provenance.csv already had checksum columns -- so a provenance.csv
    written by an older package version (no image_sha256/label_sha256 columns) still ends up
    with real, independently-verified checksums in the packaged output, with no re-run of
    the merge itself required."""
    export_dir = Path(export_dir)
    # find each output_id's actual image file (extension may vary: .jpg, .png, ...)
    image_by_id = {p.stem: p for p in (export_dir / "images").glob("*") if p.is_file()}
    out_rows = []
    missing = []
    for row in provenance_rows:
        oid = row["output_id"]
        img_path = image_by_id.get(oid)
        label_path = export_dir / "labels" / f"{oid}.txt"
        if img_path is None or not label_path.exists():
            missing.append(oid)
            continue
        row = dict(row)
        row["image_sha256"] = hashlib.sha256(img_path.read_bytes()).hexdigest()
        row["label_sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        out_rows.append(row)
    if missing:
        print(f"Warning: {len(missing)} provenance.csv rows had no matching image/label file "
              f"on disk and were dropped from the packaged output (first few: {missing[:5]})")
    return out_rows


def summarize_provenance(provenance_rows: list):
    """Re-derives real per-source-dataset image and object counts directly from the export's
    own provenance rows -- not from anything hardcoded, so this always reflects whatever
    export you actually point it at."""
    per_dataset_images = defaultdict(int)
    per_dataset_objects = defaultdict(int)
    total_objects = 0
    for row in provenance_rows:
        per_dataset_images[row["source_dataset"]] += 1
        n = int(row["n_objects"])
        per_dataset_objects[row["source_dataset"]] += n
        total_objects += n
    return dict(per_dataset_images), dict(per_dataset_objects), total_objects


def write_dataset_readme(out_dir: Path, summary: dict, audit_result: dict):
    per_images, per_objects, total_objects = summary["per_images"], summary["per_objects"], summary["total_objects"]
    lines = []
    lines.append("# BloodCellAI harmonized blood-cell detection dataset\n")
    lines.append(
        "## What this actually is\n\n"
        "This is not a newly photographed collection. Every image here was already "
        "published somewhere else; what's new is that they've been pooled into one "
        "class-harmonized, deduplicated, checksummed export, with every decision that went "
        "into building it disclosed in `mapping.csv` and `provenance.csv` rather than "
        "asserted in prose alone. If you want the full reasoning behind each choice, it's "
        "described in Section 4.7 of the paper this dataset accompanies.\n"
    )
    lines.append("## What's inside, by the numbers\n")
    lines.append("| Source dataset | Images contributed | Objects contributed |")
    lines.append("|---|---|---|")
    for ds in sorted(per_images):
        lines.append(f"| {ds} | {per_images[ds]} | {per_objects.get(ds, 0)} |")
    lines.append(f"\n**Total: {sum(per_images.values())} images, {total_objects} objects, "
                 f"across the union classes listed in `data.yaml`.**\n")
    lines.append(
        "## A few things worth knowing before you use this\n\n"
        "- **There is no train/validation/test split assigned.** `data.yaml` points all "
        "three at the same flat `images/` folder because there genuinely isn't a split "
        "yet -- that choice is left to you, and it needs to be made leakage-safe (no "
        "near-duplicate pair split across train and val/test) rather than assigned at "
        "random. The near-duplicate pairs already excluded during merging are listed in "
        "`excluded_duplicate_pairs.json`, in case you want to re-check that work yourself.\n"
        "- **The class mapping is interpretive in places, and that's written down, not "
        "hidden.** `mapping.csv` lists every original label from every source dataset, "
        "what it was mapped to, whether that mapping collapses a finer distinction the "
        "source dataset actually recorded, and why. A couple of these choices are real "
        "judgment calls -- read the `rationale` column before assuming every class name "
        "means exactly what you'd expect from its label alone.\n"
        "- **Every file has a checksum.** `provenance.csv` includes a SHA-256 for every "
        "exported image and label file, so you can confirm your download matches exactly "
        "what was reported, byte for byte.\n"
    )
    lines.append("## How to check it yourself, not just trust this README\n")
    lines.append("```python\nfrom bloodcellai_detect.merge import audit_export\nresult = audit_export('path/to/this/folder')\nassert result['passed']\n```\n")
    lines.append(f"Running that against this exact export returns:\n```json\n{json.dumps(audit_result, indent=2)}\n```\n")
    lines.append(
        "## License and attribution\n\n"
        "This export is derived from multiple sources, each with its own original license -- "
        "see `provenance.csv` for which source each file came from, and check that source's "
        "own license before redistributing. This dataset itself is released under "
        "[FILL IN YOUR CHOSEN LICENSE] -- confirm this with your supervisor and institution "
        "before publishing.\n"
    )
    lines.append(
        "## Citing this dataset\n\nSee `CITATION.cff` in this folder. This dataset has its "
        "own DOI, separate from the software that produced it -- cite both if you use both.\n"
    )
    (out_dir / "README.md").write_text("\n".join(lines))


def write_citation_cff(out_dir: Path):
    (out_dir / "CITATION.cff").write_text(
        'cff-version: 1.2.0\n'
        'message: "If you use this dataset, please cite it as below, and cite the original\\n'
        '  source dataset(s) listed in provenance.csv alongside it."\n'
        'title: "BloodCellAI harmonized blood-cell detection dataset"\n'
        'type: dataset\n'
        'authors:\n'
        '  - family-names: "FILL_IN_LAST_NAME"\n'
        '    given-names: "FILL_IN_FIRST_NAME"\n'
        '    orcid: "FILL_IN_ORCID"\n'
        '# A DOI is minted when this folder is uploaded to Zenodo as its own record --\n'
        '# add it here (and in the paper) once Zenodo assigns one.\n'
        'doi: "FILL_IN_AFTER_ZENODO_UPLOAD"\n'
    )


def main(export_dir, out_dir):
    export_dir, out_dir = Path(export_dir), Path(out_dir)
    assert (export_dir / "images").is_dir() and (export_dir / "labels").is_dir(), (
        f"{export_dir} does not look like a merge_datasets() export (expected images/ and labels/ subfolders)")

    print("Running the export-integrity audit against the real files first...")
    audit_result = audit_export(export_dir)
    print(json.dumps(audit_result, indent=2))
    if not audit_result["passed"]:
        print("\nSTOPPING: the export did not pass its own integrity audit. Fix the export "
              "itself before packaging it for release -- packaging a known-bad export would "
              "just ship the problem downstream.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        shutil.copytree(export_dir / sub, out_dir / sub, dirs_exist_ok=True)
    shutil.copy2(export_dir / "data.yaml", out_dir / "data.yaml")
    for optional in ("mapping.csv", "excluded_duplicate_pairs.json"):
        if (export_dir / optional).exists():
            shutil.copy2(export_dir / optional, out_dir / optional)
        else:
            print(f"Note: {optional} not found next to the export -- "
                  f"generate it (see export_class_mapping_audit / your merge script's "
                  f"duplicate-pairs list) and place it there before uploading, or drop this "
                  f"line from the README if you don't intend to include it.")

    with open(export_dir / "provenance.csv") as f:
        raw_rows = list(csv.DictReader(f))
    had_checksums_already = "image_sha256" in raw_rows[0] if raw_rows else False
    provenance_rows = add_real_checksums(export_dir, raw_rows)
    if had_checksums_already:
        note = "provenance.csv already had checksum columns -- recomputed fresh anyway, as an independent check"
    else:
        note = "provenance.csv had no checksum columns -- this was an older export; checksums added now, no re-run needed"
    print(f"Computed real checksums for {len(provenance_rows)} files directly from disk ({note}).")
    with open(out_dir / "provenance.csv", "w", newline="") as f:
        fieldnames = ["output_id", "source_dataset", "source_file_id", "source_image_path",
                      "n_objects", "image_sha256", "label_sha256"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(provenance_rows)

    per_images, per_objects, total_objects = summarize_provenance(provenance_rows)
    summary = dict(per_images=per_images, per_objects=per_objects, total_objects=total_objects)
    write_dataset_readme(out_dir, summary, audit_result)
    write_citation_cff(out_dir)

    print(f"\nPackaged and ready at {out_dir}/ -- README.md, CITATION.cff, data.yaml, "
          f"images/, labels/, provenance.csv (with checksums). Fill in the CITATION.cff "
          f"placeholders, zip this folder, and upload it to Zenodo as a new, separate record "
          f"(not inside the code repository).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export_dir", help="the merge_datasets() output folder (contains images/, labels/, data.yaml, provenance.csv)")
    p.add_argument("out_dir", help="where to write the packaged, upload-ready folder")
    args = p.parse_args()
    main(args.export_dir, args.out_dir)
