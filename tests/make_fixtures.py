"""Builds tiny synthetic dataset folders that reproduce the LAYOUTS discussed in the
manuscript and in the v1.5 notes. Used by the tests; no real data is needed."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _photo(path: Path, seed=0, size=(32, 32)):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _mask(path: Path, values=(0, 255), size=(32, 32)):
    a = np.zeros((size[1], size[0]), dtype=np.uint8)
    a[8:20, 8:20] = values[-1]
    if len(values) == 3:
        a[10:16, 10:16] = values[1]
    Image.fromarray(a).save(path)


def _mk(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_all(base: Path):
    base = Path(base)
    F = {}
    n = 12

    # ---- strong annotation formats -------------------------------------------------
    d = _mk(base / "voc_xml")
    for i in range(n):
        _photo(d / f"img{i}.jpg", i)
        (d / f"img{i}.xml").write_text("<annotation><object><name>RBC</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object></annotation>")
    F["voc_xml"] = d

    d = _mk(base / "coco_json")
    for i in range(n):
        _photo(d / f"img{i}.jpg", i)
    (d / "ann.json").write_text(json.dumps({"images": [{"id": 1}], "categories": [{"id": 1}],
                                           "annotations": [{"id": 1, "bbox": [1, 2, 3, 4]}]}))
    F["coco_json"] = d

    d = _mk(base / "yolo_txt")
    for i in range(n):
        _photo(d / f"img{i}.jpg", i)
        (d / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    F["yolo_txt"] = d

    d = _mk(base / "point_txt")
    for i in range(n):
        _photo(d / f"img{i}.jpg", i)
        (d / f"img{i}.txt").write_text("369 429 4\n100 120 2\n")
    F["point_txt"] = d

    d = _mk(base / "comma_polygon")
    for i in range(n):
        _photo(d / f"1-{i}.jpg", i)
        (d / f"1-{i}.txt").write_text("187,5312,2988\n1-1,Uninfected,No_comment,Polygon,3,10,10,20,10,15,20\n1-2,Parasitized,No_Comment,Polygon,3,30,30,40,30,35,40\n")
    F["comma_polygon"] = d

    d = _mk(base / "comma_point")
    for i in range(n):
        _photo(d / f"1-{i}.jpg", i)
        (d / f"1-{i}.txt").write_text("187,5312,2988\n1-1,Parasitized,No_Comment,Point,1,3833.3,306.6\n1-2,Uninfected,No_comment,Point,1,100.0,200.0\n")
    F["comma_point"] = d

    d = _mk(base / "custom_polygon_json")
    for i in range(n):
        _photo(d / f"img{i}.jpg", i)
    (d / "labels.json").write_text(json.dumps({"annotations": [{"polygon": [[1, 1], [5, 1], [3, 4]]}]}))
    F["custom_polygon_json"] = d

    # ---- classification folders ----------------------------------------------------
    d = _mk(base / "class_flat")
    for c in ("neutrophil", "lymphocyte", "monocyte"):
        _mk(d / c)
        for i in range(n):
            _photo(d / c / f"{i}.jpg", i)
    F["class_flat"] = d

    d = _mk(base / "class_wrapped")      # extra_nesting-like wrapper
    for c in ("a", "b"):
        _mk(d / "extra" / "nested" / "folders" / c)
        for i in range(n):
            _photo(d / "extra" / "nested" / "folders" / c / f"{i}.jpg", i)
    F["class_wrapped"] = d

    d = _mk(base / "class_aml_like")     # sidecar files + data/data/<classes>
    (d / "abbreviations.txt").write_text("BAS basophil\nEOS eosinophil\n")
    (d / "annotations.dat").write_text("x")
    for c in ("BAS", "EOS", "LYT", "MON"):
        _mk(d / "data" / "data" / c)
        for i in range(n):
            _photo(d / "data" / "data" / c / f"{i}.tiff", i)
    F["class_aml_like"] = d

    # ---- segmentation layouts ------------------------------------------------------
    d = _mk(base / "seg_name_mask")      # 'mask' in the FILE name: the only layout v1.4 can see
    for i in range(n):
        _photo(d / f"img{i}.png", i)
        _mask(d / f"img{i}_mask.png")
    F["seg_name_mask"] = d

    d = _mk(base / "seg_dir_masks")      # images/ + masks/ with identical file names
    for i in range(n):
        _photo(_mk(d / "images") / f"{i}.png", i)
        _mask(_mk(d / "masks") / f"{i}.png")
    F["seg_dir_masks"] = d

    d = _mk(base / "seg_grth_raabin_like")   # Original/ + GrTh1/ (+ GrTh2/)
    for i in range(n):
        _photo(_mk(d / "Original") / f"{i}.jpg", i)
        _mask(_mk(d / "GrTh1") / f"{i}.png")
        _mask(_mk(d / "GrTh2") / f"{i}.png", values=(0, 128, 255))
    F["seg_grth_raabin_like"] = d

    d = _mk(base / "seg_same_folder_pairs")  # 001.bmp + 001.png in ONE folder
    for i in range(n):
        _photo(d / "Dataset_2" / f"{i:03d}.bmp" if (d / "Dataset_2").mkdir(parents=True, exist_ok=True) is None else None, i)
        _mask(d / "Dataset_2" / f"{i:03d}.png")
    F["seg_same_folder_pairs"] = d

    d = _mk(base / "seg_tif_realmask")       # Raw/ RealMask/ BinaryMask/ (tif)
    for i in range(n):
        _photo(_mk(d / "WBC" / "Raw") / f"d{1000+i}.tif", i)
        _mask(_mk(d / "WBC" / "RealMask") / f"d{1000+i}.tif", values=(0, 128, 255))
        _mask(_mk(d / "WBC" / "BinaryMask") / f"d{1000+i}.tif", values=(0, 1))
    F["seg_tif_realmask"] = d

    d = _mk(base / "seg_voc_labelme")        # JPEGImages + SegmentationClass + labelme json
    for i in range(n):
        _photo(_mk(d / "JPEGImages") / f"{i}.jpg", i)
        _mask(_mk(d / "SegmentationClass") / f"{i}.png", values=(0, 1, 2))
        _photo(_mk(d / "before") / f"{i}.jpg", i)
        (d / "before" / f"{i}.json").write_text(json.dumps({"version": "5.0", "shapes": [{"label": "lymphocyte", "shape_type": "polygon", "points": [[1, 1], [5, 1], [3, 4]]}]}))
    F["seg_voc_labelme"] = d

    # ---- negative controls and edge cases -----------------------------------------
    d = _mk(base / "yolo_with_labels_dir")   # labels/ dir must NOT be read as masks
    for i in range(n):
        _photo(_mk(d / "images") / f"{i}.jpg", i)
        (_mk(d / "labels") / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    F["yolo_with_labels_dir"] = d

    d = _mk(base / "unlabelled")
    for i in range(n):
        _photo(d / f"{i}.jpg", i)
    F["unlabelled"] = d

    d = _mk(base / "split_class_folders")    # train/val + class folders (a classification layout)
    for s in ("train", "val"):
        for c in ("pos", "neg"):
            for i in range(6):
                _photo(_mk(d / s / c) / f"{i}.jpg", i)
    F["split_class_folders"] = d


    # ---- layouts copied from the REAL Raabin-WBC ground-truth downloads (structure only) ----
    d = _mk(base / "seg_raabin_grth_jpg")        # GrTh/GrTh/{Original,Ground Truth}/<class>/*.jpg, masks saved as JPEG
    for c in ("Basophil", "Lymphocyte", "Neutrophil"):
        for i in range(4):
            _photo(_mk(d / "GrTh" / "Original" / c) / f"{c}_{i}.jpg", i, size=(160, 160))
            a = np.zeros((160, 160), dtype=np.uint8)
            a[40:120, 40:120] = 255
            a[60:100, 60:100] = 128
            Image.fromarray(a).convert("RGB").save(_mk(d / "GrTh" / "Ground Truth" / c) / f"{c}_{i}.jpg", quality=90)
    F["seg_raabin_grth_jpg"] = d

    d = _mk(base / "seg_raabin_bmp_gt")          # Train_bmp / Train_GT_bmp / Test_bmp / Test_GT_bmp, same file names
    for split, n_ in (("Train", 8), ("Test", 4)):
        for i in range(n_):
            _photo(_mk(d / f"{split}_bmp") / f"20190526_{i}.bmp", i, size=(48, 48))
            _mask(_mk(d / f"{split}_GT_bmp") / f"20190526_{i}.bmp", size=(48, 48))
    F["seg_raabin_bmp_gt"] = d

    d = _mk(base / "class_photos_in_gt_named_dirs")   # negative control: 'GT' in a folder name but real photographs
    for c in ("GT_a", "GT_b"):
        for i in range(8):
            _photo(_mk(d / c) / f"{i}.png", i, size=(48, 48))
    F["class_photos_in_gt_named_dirs"] = d

    F["does_not_exist"] = base / "nope"
    return F
