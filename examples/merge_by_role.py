"""
Point this at any folder of dataset folders. It detects each one's role with the frozen
detector (Section 4.1), groups datasets by role, and merges each role-group the way that
role needs:

  detection_benchmark  -> union-ontology class merge (you supply a per-format parser and,
                           where native names differ, explicit overrides -- see
                           reproduce_merge.py for the bccd/txl_pbc/bbbc041/malaria_polygon
                           example this project itself uses).
  classification_only  -> the SAME merge machinery as detection: a classification image is
                           just "one object, whole-frame box" (see
                           load_classification_folder), so no separate code is needed here.
  segmentation         -> pools image/mask pairs and runs the duplicate screen across them,
                           but does NOT reconcile different datasets' mask pixel-value
                           conventions -- that is a disclosed limitation (see
                           merge_segmentation_datasets' docstring), not something this
                           script guesses at.
  unlabelled / abstain  -> reported, not merged with anything: there is nothing to combine.

Usage:
    python merge_by_role.py /path/to/dataset_a /path/to/dataset_b /path/to/dataset_c ...
"""
import sys
from pathlib import Path

from bloodcellai_detect.merge import group_by_role, load_classification_folder


def main(dataset_roots):
    roots = {Path(r).name: Path(r) for r in dataset_roots}
    groups, evidence = group_by_role(roots)

    print("=== Detected role for each dataset ===")
    for name, e in evidence.items():
        print(f"  {name}: format={e['format']}  role={e['role']}  confidence={e['confidence']}")

    print("\n=== Groups ===")
    for role, members in groups.items():
        print(f"  {role}: {sorted(members.keys())}")

    if "classification_only" in groups:
        print("\n=== Classification-role group: ready to merge with the SAME machinery as detection ===")
        from bloodcellai_detect.merge import build_union_ontology, merge_datasets
        datasets = {name: load_classification_folder(root) for name, root in groups["classification_only"].items()}
        parse_only = {k: [(fid, o, w, h) for fid, _, o, w, h in v] for k, v in datasets.items()}
        ontology = build_union_ontology(parse_only)  # add `overrides={...}` for any class-name mismatches
        print("Union classes:", ontology.union_classes)
        result = merge_datasets(datasets, ontology, out_dir=Path("merged_classification"))
        print(result)

    if "detection_benchmark" in groups:
        print("\n=== Detection-role group found ===")
        print("These need a per-format parser (VOC XML / YOLO text / COCO JSON / your own custom format)")
        print("before they can be merged -- see examples/reproduce_merge.py for a worked example")
        print("(bccd + txl_pbc + bbbc041 + malaria_polygon, this project's own Section 4.7).")

    if "segmentation" in groups:
        print("\n=== Segmentation-role group found ===")
        print("These need (file_id, image_path, mask_path) triples per dataset before merging --")
        print("see merge_segmentation_datasets' docstring. Remember: mask pixel-value semantics")
        print("across datasets are NOT reconciled automatically; check them yourself first.")

    other = {r: m for r, m in groups.items() if r not in
             ("classification_only", "detection_benchmark", "segmentation")}
    if other:
        print("\n=== Not mergeable (no role-appropriate merger, or nothing to combine) ===")
        for role, members in other.items():
            print(f"  {role}: {sorted(members.keys())}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
