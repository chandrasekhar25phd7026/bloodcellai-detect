"""
Reproduce the manuscript's Section 4.7 (union-ontology validation + reliable merged-dataset
export) on your own copies of the four detection-capable datasets: bccd, malaria_polygon,
bbbc041, txl_pbc.

bccd and malaria_polygon use the parsers already in bloodcellai_detect.validator (the same
ones Section 4.2's conversion-fidelity experiment uses). bbbc041 and txl_pbc need their own
small loader below -- fill in LOAD_BBBC041 / LOAD_TXL_PBC with a parser for each dataset's
real annotation format (their exact native schema was not re-verified for this script; do
not assume the placeholders below are correct for your attached copies). Each loader must
return a list of (file_id, image_path, objects, width, height), where each object is
{"cls_name": str, "x1": float, "y1": float, "x2": float, "y2": float} in absolute pixels.

Usage:
    python reproduce_merge.py /path/to/bccd /path/to/malaria_polygon /path/to/bbbc041 /path/to/txl_pbc
"""
import json
import sys
from pathlib import Path

from bloodcellai_detect.validator import parse_voc_xml, parse_malaria_polygon
from bloodcellai_detect.merge import build_union_ontology, merge_datasets
from bloodcellai_detect.duplicates import scan_datasets

# Disclosed interpretive choice, matching Section 2.2's existing example for malaria_polygon.
# Extend this if you decide bbbc041's own life-cycle-stage classes (e.g. "trophozoite", "ring",
# "schizont", "gametocyte") should also map onto "Parasite" -- that is a scientific judgement
# call for you to make explicitly, not something this script should assume on its own.
OVERRIDES = {
    ("malaria_polygon", "Uninfected"): "RBC",
    ("malaria_polygon", "White_Blood_Cell"): "WBC",
    ("malaria_polygon", "Parasitized"): "Parasite",
    ("bbbc041", "red blood cell"): "RBC",
    ("bbbc041", "leukocyte"): "WBC",  # confirmed present in training.json (not in every test.json sample)
    # bbbc041's four infected-stage classes: mapped to one "Parasite" class to match
    # malaria_polygon's granularity. If you want per-stage detail instead, remove these four
    # lines and the stage names (trophozoite, ring, schizont, gametocyte) will each be kept as
    # their own union class (UNMATCHED_KEPT_AS_NEW) -- a real choice for you to make, not a
    # default this script should pick quietly.
    ("bbbc041", "trophozoite"): "Parasite",
    ("bbbc041", "ring"): "Parasite",
    ("bbbc041", "schizont"): "Parasite",
    ("bbbc041", "gametocyte"): "Parasite",
}


def load_bccd(root: Path):
    root = Path(root)
    ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
    out = []
    for p in sorted(ann_dir.glob("*.xml")):
        img_path = img_dir / f"{p.stem}.jpg"
        if not img_path.exists():
            continue
        objs, (w, h) = parse_voc_xml(p)
        objs = [o for o in objs if o["x2"] > o["x1"] and o["y2"] > o["y1"]]
        if objs:
            out.append((p.stem, img_path, objs, w, h))
    return out


# Verified against the official ReadMe.pdf (NIH/NLM, 2026-10-16): GT text files and their
# images are NOT in the same folder -- they sit in sibling folders per patient:
#   .../Polygon Set/<Patient ID>/GT/<ImageName>.txt
#   .../Polygon Set/<Patient ID>/img/<ImageName>.jpg   (seen capitalized "Img" on the NLM file
#                                                        index too, so matched case-insensitively)
# An earlier version of this loader assumed txt and jpg shared one folder (same stem, swapped
# extension) and would have silently returned zero images on the real data -- fixed here.
IMAGE_EXT_CANDIDATES = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def _find_sibling_image(gt_txt_path: Path) -> "Path | None":
    patient_dir = gt_txt_path.parent.parent  # .../<Patient ID>/GT/<file>.txt -> .../<Patient ID>
    img_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name.lower() == "img"]
    for img_dir in img_dirs:
        for ext in IMAGE_EXT_CANDIDATES:
            candidate = img_dir / f"{gt_txt_path.stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_malaria_polygon(root: Path):
    out, no_image_found = [], []
    gt_txt_files = [p for p in sorted(Path(root).rglob("*.txt")) if p.parent.name.upper() == "GT"]
    for p in gt_txt_files:
        objs, (w, h) = parse_malaria_polygon(p)
        if not objs:
            continue
        img_path = _find_sibling_image(p)
        if img_path is None:
            no_image_found.append(str(p))
            continue
        out.append((p.stem, img_path, objs, w, h))
    if no_image_found:
        print(f"load_malaria_polygon: {len(no_image_found)} GT files had no matching image "
              f"in a sibling img/ folder, skipped (not silently merged in):")
        for f in no_image_found[:10]:
            print("  ", f)
    return out


# Verified against a real test.json (2026-10-16): the kmader/malaria-bounding-boxes Kaggle
# mirror of BBBC041. 120 test images, 5,922 raw objects across 6 categories; excluding
# "difficult" (5 objects -- an ambiguity flag, not a real cell type, so not passed through as
# a detection class) leaves 5,917 -- matching independently published BBBC041 test-split
# counts exactly. Point this at BOTH training.json and test.json to get the full dataset;
# call it once per file and merge the two lists.
def load_bbbc041(json_path: Path, images_root: Path = None):
    """json_path: training.json or test.json from the Kaggle mirror above. images_root: the
    folder containing the referenced images (if None, uses the pathname as given, relative to
    the current directory).

    Kaggle's extraction of this dataset has been observed to duplicate the top-level folder
    one level deep (an "images/" folder sitting both at images_root and at
    images_root/malaria/images/, with identical contents) -- a packaging artifact, not a
    meaningful structural difference. Rather than guess which exact prefix the JSON's
    `pathname` field assumes, this loader tries the direct path first and, if that file isn't
    there, falls back to finding any file under images_root with the same basename. This is
    reported, not silent: `n_found_by_fallback` in the printed summary tells you how often the
    fallback was needed.
    """
    with open(json_path) as f:
        records = json.load(f)

    basename_index = None  # built lazily, only if the direct path ever misses

    def build_index():
        idx = {}
        for p in Path(images_root).rglob("*"):
            if p.is_file():
                idx.setdefault(p.name, p)
        return idx

    out, n_fallback, n_missing = [], 0, 0
    for rec in records:
        img_info = rec['image']
        h, w = img_info['shape']['r'], img_info['shape']['c']
        objs = []
        for o in rec['objects']:
            if o['category'] == 'difficult':  # an ambiguity flag, not a cell type -- excluded, not silently kept
                continue
            bb = o['bounding_box']
            objs.append(dict(cls_name=o['category'], x1=bb['minimum']['c'], y1=bb['minimum']['r'],
                              x2=bb['maximum']['c'], y2=bb['maximum']['r']))
        if not objs:
            continue
        pathname = img_info['pathname'].lstrip('/')
        direct = (Path(images_root) / pathname) if images_root else Path(pathname)
        if direct.exists():
            image_path = direct
        elif images_root:
            if basename_index is None:
                basename_index = build_index()
            image_path = basename_index.get(Path(pathname).name)
            if image_path is None:
                n_missing += 1
                continue
            n_fallback += 1
        else:
            n_missing += 1
            continue
        out.append((Path(pathname).stem, image_path, objs, float(w), float(h)))
    print(f"load_bbbc041({Path(json_path).name}): {len(out)} loaded, "
          f"{n_fallback} found via basename fallback, {n_missing} not found at all")
    return out


# Verified against a real TXL-PBC download (2026-10-15): data.yaml reads
# names: ['WBC', 'RBC','Platelets'], nc: 3; labels are standard YOLO-normalized
# "class_id xc yc w h" with \r\n line endings; images/labels/ each split into train (882),
# val (252), test (126). Running this loader reproduces exactly the 18,143-object total
# already reported for txl_pbc in Table 6 (12,510 + 3,752 + 1,881).
TXL_PBC_CLASS_NAMES = ["WBC", "RBC", "Platelets"]


def load_txl_pbc(root: Path, splits=("train", "val", "test")):
    from PIL import Image
    root = Path(root)
    out = []
    for split in splits:
        images_root, labels_root = root / "images" / split, root / "labels" / split
        for lbl_path in sorted(labels_root.glob("*.txt")):
            img_path = images_root / f"{lbl_path.stem}.png"
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            objs = []
            for raw in lbl_path.read_text().splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                objs.append(dict(cls_name=TXL_PBC_CLASS_NAMES[cid], x1=x1, y1=y1, x2=x2, y2=y2))
            if objs:
                out.append((f"{split}/{lbl_path.stem}", img_path, objs, float(w), float(h)))
    return out


def main(bccd_root, malaria_root, bbbc041_root, txl_pbc_root, out_dir="merged_export"):
    datasets = {
        "bccd": load_bccd(bccd_root),
        "malaria_polygon": load_malaria_polygon(malaria_root),
        "bbbc041": load_bbbc041(Path(bbbc041_root) / "training.json", bbbc041_root)
                   + load_bbbc041(Path(bbbc041_root) / "test.json", bbbc041_root),
        "txl_pbc": load_txl_pbc(txl_pbc_root),
    }

    parse_only = {k: [(fid, objs, w, h) for fid, _, objs, w, h in v] for k, v in datasets.items()}
    ontology = build_union_ontology(parse_only, overrides=OVERRIDES)

    print("=== Union ontology ===")
    print(f"Native classes (dataset, name) -> union class, and how each was decided:")
    for key in sorted(ontology.route_by_native):
        print(f"  {key} -> {ontology.native_to_union[key]}  [{ontology.route_by_native[key]}]")
    print(f"\nUnion classes ({len(ontology.union_classes)}): {ontology.union_classes}")

    print("\n=== Cross-dataset duplicate scan (same roots) ===")
    roots = {"bccd": bccd_root, "malaria_polygon": malaria_root, "bbbc041": bbbc041_root, "txl_pbc": txl_pbc_root}
    dup_scan = scan_datasets(roots)
    print(f"Candidate near-duplicate pairs found: {dup_scan['n_candidate_pairs']} "
          f"({dup_scan['n_within_dataset']} within-dataset, {dup_scan['n_cross_dataset']} cross-dataset)")
    print("Look at a SAMPLE of dup_scan['review_sample'] BY EYE for false positives (not every")
    print("pair -- a bulk visual check isn't practical). For the actual exclusion list, use")
    print("dup_scan['all_pairs'] -- NOT review_sample, which is capped for eyeballing and can")
    print("under-count real duplicates when one dataset pair has many true candidates (this")
    print("project's own earlier guidance made exactly this mistake: see docs/PROVENANCE.md).")

    # Only bccd vs txl_pbc were ever confirmed as real duplicates in this project's own review
    # (bbbc041's apparent self-duplication was a folder-copy artifact; malaria_polygon's
    # apparent near-duplicates were a coarse-hash collision -- see Section 4.7). Adjust this
    # filter if your own visual review finds genuine duplicates elsewhere.
    confirmed_duplicates = [
        ((row["dataset_a"], row["path_a"]), (row["dataset_b"], row["path_b"]))
        for row in dup_scan["all_pairs"]
        if {row["dataset_a"], row["dataset_b"]} == {"bccd", "txl_pbc"} or row["dataset_a"] == row["dataset_b"] == "bccd"
    ]
    print(f"\nConfirmed duplicates to exclude: {len(confirmed_duplicates)}")

    # txl_pbc's annotations are a documented re-labeling of (part of) bccd (Section 4.6.3) --
    # prefer keeping txl_pbc's version of a duplicated image, not an uninformed alphabetical
    # tiebreak, for exactly the pairs where that relationship is known to apply.
    result = merge_datasets(datasets, ontology, duplicate_pairs=confirmed_duplicates, out_dir=Path(out_dir),
                             prefer_dataset_order=["txl_pbc", "bccd"])
    print("\n=== Merge result ===")
    print(json.dumps(result.__dict__, indent=2, default=str))
    print(f"\nWrote merged export to {out_dir}/ (images/, labels/, data.yaml, provenance.csv)")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
