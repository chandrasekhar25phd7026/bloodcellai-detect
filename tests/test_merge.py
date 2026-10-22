import csv

import numpy as np
import pytest
from PIL import Image

from bloodcellai_detect.merge import (
    build_union_ontology, merge_datasets, export_class_mapping_audit,
    verify_class_counts, audit_export,
)


def box(cls_name, x1, y1, x2, y2):
    return dict(cls_name=cls_name, x1=x1, y1=y1, x2=x2, y2=y2)


def test_identical_names_auto_match_across_datasets():
    datasets = {
        "a": [("f1", [box("RBC", 0, 0, 10, 10), box("WBC", 20, 20, 30, 30)], 100, 100)],
        "b": [("f2", [box("RBC", 0, 0, 10, 10)], 100, 100)],
    }
    ont = build_union_ontology(datasets)
    assert ont.route_by_native[("a", "RBC")] == "IDENTICAL_NAME"
    assert ont.route_by_native[("b", "RBC")] == "IDENTICAL_NAME"
    assert set(ont.union_classes) == {"RBC", "WBC"}


def test_unmatched_name_kept_as_new_class_not_dropped():
    datasets = {"a": [("f1", [box("Platelet", 0, 0, 10, 10)], 100, 100)]}
    ont = build_union_ontology(datasets)
    assert ont.route_by_native[("a", "Platelet")] == "UNMATCHED_KEPT_AS_NEW"
    assert "Platelet" in ont.union_classes  # never silently dropped


def test_explicit_override_is_disclosed_and_applied():
    datasets = {
        "malaria": [("f1", [box("Uninfected", 0, 0, 10, 10), box("Parasitized", 20, 20, 30, 30)], 100, 100)],
        "bccd": [("f2", [box("RBC", 0, 0, 10, 10)], 100, 100)],
    }
    overrides = {("malaria", "Uninfected"): "RBC", ("malaria", "Parasitized"): "Parasite"}
    ont = build_union_ontology(datasets, overrides=overrides)
    assert ont.native_to_union[("malaria", "Uninfected")] == "RBC"
    assert ont.route_by_native[("malaria", "Uninfected")] == "EXPLICIT_OVERRIDE"
    assert ont.native_to_union[("bccd", "RBC")] == "RBC"  # bccd's own RBC still auto-matches
    assert "Parasite" in ont.union_classes


def test_override_for_a_dataset_with_zero_loaded_objects_is_not_silently_applied():
    # Regression test: an earlier version applied every override unconditionally, so a dataset
    # that failed to load (e.g. a wrong path) could still appear to "contribute" classes via
    # its overrides, masking the real failure. Caught 2026-10-16 from a real Kaggle run where
    # three of four datasets loaded zero objects.
    datasets = {"bccd": [], "malaria_polygon": []}  # both failed to load -- zero objects
    overrides = {("malaria_polygon", "Uninfected"): "RBC"}
    ont = build_union_ontology(datasets, overrides=overrides)
    assert ont.route_by_native == {}
    assert ont.union_classes == []
    assert ont.unused_overrides == {("malaria_polygon", "Uninfected"): "RBC"}


def test_override_still_applies_normally_when_the_class_is_really_there():
    datasets = {"malaria_polygon": [("f1", [box("Uninfected", 0, 0, 10, 10)], 100, 100)]}
    overrides = {("malaria_polygon", "Uninfected"): "RBC"}
    ont = build_union_ontology(datasets, overrides=overrides)
    assert ont.native_to_union[("malaria_polygon", "Uninfected")] == "RBC"
    assert ont.unused_overrides == {}


def test_no_native_class_is_ever_missing_from_the_report():
    datasets = {
        "a": [("f1", [box("X", 0, 0, 10, 10), box("Y", 0, 0, 10, 10)], 100, 100)],
        "b": [("f2", [box("Y", 0, 0, 10, 10), box("Z", 0, 0, 10, 10)], 100, 100)],
    }
    ont = build_union_ontology(datasets)
    all_native = {("a", "X"), ("a", "Y"), ("b", "Y"), ("b", "Z")}
    assert set(ont.route_by_native.keys()) == all_native
    assert all(k in ont.native_to_union for k in all_native)


def _save_jpg(path, color):
    Image.fromarray(np.full((20, 20, 3), color, dtype=np.uint8)).save(path)


def test_merge_export_writes_self_consistent_yolo_dataset(tmp_path):
    img_a1 = tmp_path / "a1.jpg"; _save_jpg(img_a1, 10)
    img_a2 = tmp_path / "a2.jpg"; _save_jpg(img_a2, 20)
    img_b1 = tmp_path / "b1.jpg"; _save_jpg(img_b1, 30)
    datasets = {
        "a": [("a1", img_a1, [box("RBC", 10, 10, 20, 20)], 100, 100),
              ("a2", img_a2, [box("WBC", 0, 0, 50, 50)], 100, 100)],
        "b": [("b1", img_b1, [box("RBC", 0, 0, 10, 10)], 100, 100)],
    }
    parse_only = {k: [(fid, objs, w, h) for fid, _, objs, w, h in v] for k, v in datasets.items()}
    ont = build_union_ontology(parse_only)
    out = tmp_path / "merged"
    result = merge_datasets(datasets, ont, out_dir=out)

    assert result.n_images_before_dedup == 3
    assert result.n_images_excluded_as_duplicate == 0
    assert result.n_images_in_merged_export == 3
    assert result.n_union_classes == 2  # RBC, WBC

    img_files = {p.stem for p in (out / "images").glob("*")}
    lbl_files = {p.stem for p in (out / "labels").glob("*")}
    assert img_files == lbl_files == {"a__a1", "a__a2", "b__b1"}
    assert (out / "data.yaml").read_text().count("nc: 2") == 1
    label_text = (out / "labels" / "b__b1.txt").read_text()
    assert label_text.startswith(f"{ont.union_id('RBC')} ")


def test_merge_export_excludes_flagged_cross_dataset_duplicates(tmp_path):
    img_a1 = tmp_path / "a1.jpg"; _save_jpg(img_a1, 50)
    img_b1 = tmp_path / "b1.jpg"; _save_jpg(img_b1, 50)  # same underlying content as a1
    datasets = {
        "a": [("a1", img_a1, [box("RBC", 0, 0, 10, 10)], 100, 100)],
        "b": [("b1", img_b1, [box("RBC", 0, 0, 10, 10)], 100, 100)],
    }
    parse_only = {k: [(fid, objs, w, h) for fid, _, objs, w, h in v] for k, v in datasets.items()}
    ont = build_union_ontology(parse_only)
    duplicate_pairs = [(("a", str(img_a1)), ("b", str(img_b1)))]
    out = tmp_path / "merged2"
    result = merge_datasets(datasets, ont, duplicate_pairs=duplicate_pairs, out_dir=out)

    assert result.n_images_before_dedup == 2
    assert result.n_images_excluded_as_duplicate == 1
    assert result.n_images_in_merged_export == 1
    img_files = {p.stem for p in (out / "images").glob("*")}
    assert img_files == {"a__a1"}  # "a" sorts before "b", so b1 (the later one) is the one dropped


def test_merge_respects_prefer_dataset_order_for_which_side_to_keep(tmp_path):
    img_a1 = tmp_path / "a1.jpg"; _save_jpg(img_a1, 50)
    img_b1 = tmp_path / "b1.jpg"; _save_jpg(img_b1, 50)
    datasets = {
        "bccd": [("a1", img_a1, [box("RBC", 0, 0, 10, 10)], 100, 100)],
        "txl_pbc": [("b1", img_b1, [box("RBC", 0, 0, 10, 10)], 100, 100)],
    }
    parse_only = {k: [(fid, o, w, h) for fid, _, o, w, h in v] for k, v in datasets.items()}
    ont = build_union_ontology(parse_only)
    duplicate_pairs = [(("bccd", str(img_a1)), ("txl_pbc", str(img_b1)))]

    # default (no preference): alphabetical -- "bccd" sorts first and is kept, "txl_pbc" dropped
    out1 = tmp_path / "default"
    r1 = merge_datasets(datasets, ont, duplicate_pairs=duplicate_pairs, out_dir=out1)
    assert {p.stem for p in (out1 / "images").glob("*")} == {"bccd__a1"}

    # explicit preference for txl_pbc over bccd -- txl_pbc kept, bccd dropped
    out2 = tmp_path / "preferred"
    r2 = merge_datasets(datasets, ont, duplicate_pairs=duplicate_pairs, out_dir=out2,
                         prefer_dataset_order=["txl_pbc", "bccd"])
    assert {p.stem for p in (out2 / "images").glob("*")} == {"txl_pbc__b1"}


def test_merge_without_out_dir_still_returns_summary_counts(tmp_path):
    img = tmp_path / "x.jpg"; _save_jpg(img, 5)
    datasets = {"a": [("f1", img, [box("RBC", 0, 0, 10, 10)], 100, 100)]}
    parse_only = {"a": [("f1", [box("RBC", 0, 0, 10, 10)], 100, 100)]}
    ont = build_union_ontology(parse_only)
    result = merge_datasets(datasets, ont)  # no out_dir
    assert result.output_dir is None
    assert result.n_images_in_merged_export == 1


def test_export_class_mapping_audit_and_count_verification(tmp_path):
    datasets = {
        "bccd": [("f1", [box("RBC", 0, 0, 10, 10), box("WBC", 0, 0, 5, 5)], 100, 100)],
        "malaria_polygon": [("f2", [box("Uninfected", 0, 0, 10, 10), box("Parasitized", 0, 0, 5, 5),
                                     box("Parasitized", 10, 10, 15, 15)], 100, 100)],
    }
    overrides = {("malaria_polygon", "Uninfected"): "RBC", ("malaria_polygon", "Parasitized"): "Parasite"}
    ont = build_union_ontology(datasets, overrides=overrides)

    native_counts = {("bccd", "RBC"): 1, ("bccd", "WBC"): 1,
                      ("malaria_polygon", "Uninfected"): 1, ("malaria_polygon", "Parasitized"): 2}
    union_counts = verify_class_counts(ont, native_counts)
    assert union_counts["RBC"] == 2  # bccd's 1 + malaria_polygon's 1
    assert union_counts["Parasite"] == 2
    assert union_counts["WBC"] == 1
    assert sum(union_counts.values()) == sum(native_counts.values())

    out_csv = tmp_path / "mapping.csv"
    export_class_mapping_audit(ont, native_counts, out_csv)
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 4
    by_native = {(r["source_dataset"], r["native_label"]): r for r in rows}
    assert by_native[("bccd", "RBC")]["mapping_type"] == "exact_match"
    assert by_native[("bccd", "RBC")]["information_loss"] == "no"
    assert by_native[("malaria_polygon", "Parasitized")]["mapping_type"] == "semantic_override"
    assert by_native[("malaria_polygon", "Parasitized")]["information_loss"] == "yes"
    assert by_native[("malaria_polygon", "Parasitized")]["native_object_count"] == "2"
    assert all(r["rationale"] for r in rows)  # every row has a real rationale, none blank


def test_verify_class_counts_catches_a_real_mismatch():
    datasets = {"bccd": [("f1", [box("RBC", 0, 0, 10, 10)], 100, 100)]}
    ont = build_union_ontology(datasets)
    # deliberately wrong counts -- a native class present in the ontology but missing from counts
    with pytest.raises(AssertionError):
        verify_class_counts(ont, {("bccd", "RBC"): 5, ("some_other_dataset", "X"): 1})


def test_audit_export_passes_on_a_real_clean_export(tmp_path):
    datasets = {"bccd": [("f1", [box("RBC", 0, 0, 10, 10)], 100, 100)]}
    ont = build_union_ontology(datasets)
    out = tmp_path / "export"
    img_path = tmp_path / "f1.jpg"
    _save_jpg(img_path, 10)
    merge_datasets({"bccd": [("f1", img_path, [box("RBC", 0, 0, 10, 10)], 100, 100)]},
                    ont, out_dir=out)
    result = audit_export(out)
    assert result["passed"] is True
    assert result["images_without_label"] == []
    assert result["labels_without_image"] == []
    assert result["n_bad_class_id_lines_total"] == 0


def test_audit_export_catches_an_out_of_range_class_id(tmp_path):
    datasets = {"bccd": [("f1", [box("RBC", 0, 0, 10, 10)], 100, 100)]}
    ont = build_union_ontology(datasets)
    out = tmp_path / "export"
    img_path = tmp_path / "f1b.jpg"
    _save_jpg(img_path, 20)
    merge_datasets({"bccd": [("f1", img_path, [box("RBC", 0, 0, 10, 10)], 100, 100)]},
                    ont, out_dir=out)
    # corrupt a label file with an out-of-range class id, as if written by a bug
    label_path = next((out / "labels").glob("*.txt"))
    label_path.write_text("99 0.5 0.5 0.1 0.1\n")
    result = audit_export(out)
    assert result["passed"] is False
    assert result["n_bad_class_id_lines_total"] == 1
    assert "out of range" in result["bad_class_id_lines"][0]


def test_provenance_csv_includes_checksums(tmp_path):
    img_path = tmp_path / "f1c.jpg"
    _save_jpg(img_path, 30)
    datasets = {"bccd": [("f1", img_path, [box("RBC", 0, 0, 10, 10)], 50, 50)]}
    ont = build_union_ontology({"bccd": [("f1", [box("RBC", 0, 0, 10, 10)], 50, 50)]})
    out = tmp_path / "export2"
    merge_datasets(datasets, ont, out_dir=out)
    rows = list(csv.DictReader(open(out / "provenance.csv")))
    assert len(rows) == 1
    assert len(rows[0]["image_sha256"]) == 64  # a real sha256 hex digest, not a placeholder
    assert len(rows[0]["label_sha256"]) == 64
