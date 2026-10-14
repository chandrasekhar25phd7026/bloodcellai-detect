import random

import pytest

from bloodcellai_detect.validator import (
    DEFECT_TYPES, STRUCTURALLY_UNDETECTABLE, validate_annotations, inject_defects,
    score_defect_type, run_defect_benchmark, check_real_defects,
)


def box(cls_name, x1, y1, x2, y2):
    return dict(cls_name=cls_name, x1=x1, y1=y1, x2=x2, y2=y2)


def test_validate_flags_zero_area():
    objs = [box("RBC", 10, 10, 10, 30)]  # zero width
    flags = validate_annotations(objs, 100, 100)
    assert flags[0] == {"zero_area"}


def test_validate_flags_out_of_bounds():
    objs = [box("RBC", -5, 10, 20, 30)]
    flags = validate_annotations(objs, 100, 100)
    assert "out_of_bounds" in flags[0]


def test_validate_flags_duplicate_both_sides():
    objs = [box("RBC", 10, 10, 20, 20), box("RBC", 10, 10, 20, 20)]
    flags = validate_annotations(objs, 100, 100)
    assert flags[0] == {"duplicate"} and flags[1] == {"duplicate"}


def test_validate_clean_box_has_no_flags():
    objs = [box("RBC", 10, 10, 20, 20)]
    assert validate_annotations(objs, 100, 100) == [set()]


def test_class_flip_and_shifted_box_are_structurally_undetectable():
    assert STRUCTURALLY_UNDETECTABLE == {"class_flip", "shifted_box"}
    objs = [box("RBC", 10, 10, 20, 20)]
    flipped, log = inject_defects(objs, 100, 100, "class_flip", 1.0, random.Random(0), ["RBC", "WBC"])
    assert log == ["class_flip"]
    assert validate_annotations(flipped, 100, 100) == [set()]  # looks perfectly clean structurally


def test_inject_zero_area_is_recoverable_with_perfect_score():
    objs = [box("RBC", 10, 10, 20, 20) for _ in range(20)]
    rng = random.Random(42)
    defected, log = inject_defects(objs, 100, 100, "zero_area", 0.5, rng, ["RBC"])
    s = score_defect_type(defected, 100, 100, log, "zero_area")
    assert s["n_injected"] == 10
    assert s["precision"] == 1.0 and s["recall"] == 1.0 and s["f1"] == 1.0
    assert s["structurally_undetectable"] is False


def test_inject_class_flip_scores_zero_recall_honestly():
    objs = [box("RBC", 10, 10, 20, 20) for _ in range(20)]
    rng = random.Random(42)
    defected, log = inject_defects(objs, 100, 100, "class_flip", 0.5, rng, ["RBC", "WBC"])
    s = score_defect_type(defected, 100, 100, log, "class_flip")
    assert s["n_injected"] > 0
    assert s["recall"] == 0.0
    assert s["structurally_undetectable"] is True


def test_duplicate_line_injection_adds_an_object():
    objs = [box("RBC", 10, 10, 20, 20)]
    defected, log = inject_defects(objs, 100, 100, "duplicate_line", 1.0, random.Random(0), ["RBC"])
    assert len(defected) == 2 and log == ["duplicate_line", "duplicate_line"]


def test_shifted_box_never_injected_out_of_bounds():
    # a box already hugging the edge: a "shifted" version must stay in-bounds or not count
    objs = [box("RBC", 0, 0, 10, 10)] * 30
    rng = random.Random(1)
    defected, log = inject_defects(objs, 100, 100, "shifted_box", 1.0, rng, ["RBC"])
    for o, t in zip(defected, log):
        if t == "shifted_box":
            assert 0 <= o["x1"] and o["x2"] <= 100 and 0 <= o["y1"] and o["y2"] <= 100


def test_run_defect_benchmark_all_five_types_present():
    per_file = [("img1", [box("RBC", 10, 10, 20, 20), box("WBC", 50, 50, 70, 70)], 100, 100)]
    results = run_defect_benchmark({"toy": per_file}, injection_rates=(0.5,), seed=1)
    rate_summary = results["toy"]["results_by_injection_rate"]["0.5"]
    assert set(rate_summary.keys()) == set(DEFECT_TYPES)
    for dt in STRUCTURALLY_UNDETECTABLE:
        assert rate_summary[dt]["structurally_undetectable"] is True


def test_check_real_defects_finds_preexisting_zero_area_box():
    per_file = [("a.xml", [box("RBC", 10, 10, 10, 20), box("RBC", 30, 30, 40, 40)], 100, 100)]
    result = check_real_defects(per_file)
    assert result["n_raw_objects"] == 2
    assert result["n_real_defects_found"] == 1
    assert result["examples"][0]["file"] == "a.xml"
