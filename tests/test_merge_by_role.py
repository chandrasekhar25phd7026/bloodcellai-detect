import numpy as np
from PIL import Image

from bloodcellai_detect.merge import (
    load_classification_folder, merge_segmentation_datasets, group_by_role,
    build_union_ontology, merge_datasets,
)


def _img(path, color, size=(20, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(path)


def test_classification_loader_reads_folder_as_whole_frame_box(tmp_path):
    ds = tmp_path / "ds"
    _img(ds / "RBC" / "0.jpg", 10)
    _img(ds / "RBC" / "1.jpg", 11)
    _img(ds / "WBC" / "0.jpg", 20)
    out = load_classification_folder(ds)
    assert len(out) == 3
    by_class = {o["cls_name"] for _, _, objs, _, _ in out for o in objs}
    assert by_class == {"RBC", "WBC"}
    _, _, objs, w, h = out[0]
    assert objs[0]["x1"] == 0 and objs[0]["y1"] == 0 and objs[0]["x2"] == w and objs[0]["y2"] == h


def test_classification_merge_reuses_detection_merge_machinery(tmp_path):
    ds1, ds2 = tmp_path / "ds1", tmp_path / "ds2"
    _img(ds1 / "neutrophil" / "0.jpg", 10)
    _img(ds1 / "lymphocyte" / "0.jpg", 20)
    _img(ds2 / "neutrophil" / "0.jpg", 30)
    data1, data2 = load_classification_folder(ds1), load_classification_folder(ds2)
    datasets = {"ds1": data1, "ds2": data2}
    parse_only = {k: [(fid, o, w, h) for fid, _, o, w, h in v] for k, v in datasets.items()}
    ont = build_union_ontology(parse_only)
    assert ont.route_by_native[("ds1", "neutrophil")] == "IDENTICAL_NAME"
    assert ont.route_by_native[("ds2", "neutrophil")] == "IDENTICAL_NAME"
    result = merge_datasets(datasets, ont, out_dir=tmp_path / "merged_cls")
    assert result.n_images_in_merged_export == 3
    assert result.n_union_classes == 2
    lbl = (tmp_path / "merged_cls" / "labels" / "ds1__0.txt").read_text()
    # whole-frame box: xc=yc=0.5, w=h=1.0
    assert "0.500000 0.500000 1.000000 1.000000" in lbl


def test_classification_loader_ignores_images_directly_in_root(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _img(ds / "stray.jpg", 5)
    _img(ds / "A" / "0.jpg", 6)
    out = load_classification_folder(ds)
    assert len(out) == 1
    assert out[0][2][0]["cls_name"] == "A"


def test_segmentation_merge_pools_and_dedups(tmp_path):
    ds1, ds2 = tmp_path / "s1", tmp_path / "s2"
    ds1.mkdir(); ds2.mkdir()
    img_a = ds1 / "a.png"; _img(img_a, 40)
    msk_a = ds1 / "a_mask.png"; _img(msk_a, 255)
    img_b = ds2 / "b.png"; _img(img_b, 40)  # same content as img_a -> cross-dataset duplicate
    msk_b = ds2 / "b_mask.png"; _img(msk_b, 255)
    img_c = ds2 / "c.png"; _img(img_c, 99)
    msk_c = ds2 / "c_mask.png"; _img(msk_c, 128)

    datasets = {"s1": [("a", img_a, msk_a)], "s2": [("b", img_b, msk_b), ("c", img_c, msk_c)]}
    dup_pairs = [(("s1", str(img_a)), ("s2", str(img_b)))]
    result = merge_segmentation_datasets(datasets, duplicate_pairs=dup_pairs, out_dir=tmp_path / "merged_seg")
    assert result.n_pairs_before_dedup == 3
    assert result.n_pairs_excluded_as_duplicate == 1
    assert result.n_pairs_in_merged_export == 2
    assert result.mask_semantics_reconciled is False  # the disclosed limitation, always explicit

    img_stems = {p.stem for p in (tmp_path / "merged_seg" / "images").glob("*")}
    msk_stems = {p.stem for p in (tmp_path / "merged_seg" / "masks").glob("*")}
    assert img_stems == msk_stems == {"s1__a", "s2__c"}


def test_segmentation_merge_without_out_dir_still_counts(tmp_path):
    ds1 = tmp_path / "s1"; ds1.mkdir()
    img_a = ds1 / "a.png"; _img(img_a, 1)
    msk_a = ds1 / "a_mask.png"; _img(msk_a, 2)
    result = merge_segmentation_datasets({"s1": [("a", img_a, msk_a)]})
    assert result.output_dir is None
    assert result.n_pairs_in_merged_export == 1


def test_group_by_role_separates_classification_and_detection(tmp_path):
    cls_ds = tmp_path / "cls_ds"
    _img(cls_ds / "RBC" / "0.jpg", 10)
    _img(cls_ds / "RBC" / "1.jpg", 11)
    _img(cls_ds / "WBC" / "0.jpg", 12)
    _img(cls_ds / "WBC" / "1.jpg", 13)
    unlabelled_ds = tmp_path / "plain_ds"
    _img(unlabelled_ds / "0.jpg", 20)
    _img(unlabelled_ds / "1.jpg", 21)

    groups, evidence = group_by_role({"cls_ds": cls_ds, "plain_ds": unlabelled_ds})
    assert "cls_ds" in groups.get("classification_only", {})
    assert "plain_ds" in groups.get("unlabelled", {})
    assert evidence["cls_ds"]["format"] == "classification_folders"
