import csv
import json
from pathlib import Path

import pytest

from bloodcellai_detect import detect_format_and_role
from bloodcellai_detect.audit import run_audit, wilson_ci
from bloodcellai_detect.lock import lock, validate, verify_lock_or_die
from make_fixtures import build_all

HERE = Path(__file__).parent
EXPECTED_V14 = json.loads((HERE / "expected_v1_4.json").read_text())


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("fixtures"))


def test_v14_matches_original_notebook_outputs(fx):
    """expected_v1_4.json was produced by executing the ORIGINAL notebook function
    (bdqi-v2, cell 299) on the same synthetic fixtures. Frozen means frozen."""
    for name, path in fx.items():
        fmt, role, conf, _ = detect_format_and_role(path, version="1.4")
        e = EXPECTED_V14[name]
        assert (fmt, role, conf) == (e["format"], e["role"], e["confidence"]), name


# Documented limitation of v1.4: only the file-name layout is recognised as segmentation.
V14_SEG_GAPS = {
    "seg_dir_masks": "classification_folders",
    "seg_grth_raabin_like": "classification_folders",
    "seg_same_folder_pairs": "unlabelled",
    "seg_tif_realmask": "classification_folders",
    "seg_voc_labelme": "custom_polygon_json",
    "seg_raabin_grth_jpg": "classification_folders",   # real Raabin layout 1
    "seg_raabin_bmp_gt": "classification_folders",    # real Raabin layout 2
}


@pytest.mark.parametrize("name,wrong_format", sorted(V14_SEG_GAPS.items()))
def test_v14_known_segmentation_gaps(fx, name, wrong_format):
    fmt, role, _, _ = detect_format_and_role(fx[name], version="1.4")
    assert fmt == wrong_format and role != "segmentation"


def test_v14_recognises_filename_masks(fx):
    assert detect_format_and_role(fx["seg_name_mask"], "1.4")[:2] == ("segmentation_masks", "segmentation")


@pytest.mark.parametrize("name", ["seg_name_mask", *sorted(V14_SEG_GAPS)])
def test_v15_finds_segmentation_layouts(fx, name):
    assert detect_format_and_role(fx[name], "1.5")[:2] == ("segmentation_masks", "segmentation")


@pytest.mark.parametrize("name", [k for k in EXPECTED_V14 if not k.startswith("seg_")])
def test_v15_does_not_change_non_segmentation_results(fx, name):
    assert detect_format_and_role(fx[name], "1.5")[:2] == detect_format_and_role(fx[name], "1.4")[:2]


def test_v15_does_not_treat_photos_in_gt_named_folders_as_masks(fx):
    assert detect_format_and_role(fx["class_photos_in_gt_named_dirs"], "1.5")[:2] == ("classification_folders", "classification_only")


def test_missing_path_abstains(fx):
    assert detect_format_and_role(fx["does_not_exist"])[:2] == ("abstain", "abstain")


def test_unknown_version_rejected(fx):
    with pytest.raises(ValueError):
        detect_format_and_role(fx["voc_xml"], version="9.9")


def test_wilson_ci_basic():
    lo, hi = wilson_ci(13, 15)
    assert round(lo, 3) == 0.621 and round(hi, 3) == 0.963   # the value reported in the manuscript (Table 2)


def _gt_csv(path, rows):
    cols = ["dataset_id", "dataset_name", "source_url", "on_kaggle", "kaggle_path_hint", "true_format", "true_role",
            "n_images_expected", "n_classes_expected", "license", "notes", "filled_by", "fill_date"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "x") for c in cols})


def test_lock_rejects_placeholders_and_audit_runs(fx, tmp_path):
    gt = tmp_path / "gt.csv"
    rows = [dict(dataset_id="voc", kaggle_path_hint=str(fx["voc_xml"]), true_format="voc_xml",
                 true_role="detection_benchmark", filled_by="FILL_YOUR_NAME"),
            dict(dataset_id="seg", kaggle_path_hint=str(fx["seg_dir_masks"]), true_format="segmentation_masks",
                 true_role="segmentation", filled_by="Tester")]
    _gt_csv(gt, rows)
    from bloodcellai_detect.lock import read_rows
    assert any("filled_by" in e for e in validate(read_rows(gt)))
    rows[0]["filled_by"] = "Tester"
    _gt_csv(gt, rows)
    lock(gt)
    verify_lock_or_die(gt)
    _, _, stats = run_audit(gt, tmp_path / "out", version="1.4")
    assert stats["n"] == 2 and stats["role_acc"] == 0.5            # v1.4 misses the folder-mask layout
    _, _, stats15 = run_audit(gt, tmp_path / "out15", version="1.5")
    assert stats15["role_acc"] == 1.0
    gt.write_text(gt.read_text() + "\n")                            # tamper after locking
    with pytest.raises(SystemExit):
        verify_lock_or_die(gt)
