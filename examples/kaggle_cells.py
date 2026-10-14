# ======================================================================================
# KAGGLE CELLS: format + role detection on NEW segmentation / classification datasets
# Run the cells in order. Each "# %% CELL n" block is one notebook cell.
# ======================================================================================

# %% CELL 1 : make the package importable  (no internet needed)
# Upload the folder `bloodcellai-detect` as a Kaggle Dataset (any name) and add it to this notebook.
import glob, sys, zipfile, os
hits = glob.glob("/kaggle/input/**/bloodcellai_detect/__init__.py", recursive=True)
if not hits:   # the package may still be a .zip: unpack it into /kaggle/working
    zips = glob.glob("/kaggle/input/**/bloodcellai-detect*.zip", recursive=True)
    assert zips, "package not found: add the uploaded bloodcellai-detect dataset to this notebook"
    zipfile.ZipFile(zips[0]).extractall("/kaggle/working/pkg")
    hits = glob.glob("/kaggle/working/pkg/**/bloodcellai_detect/__init__.py", recursive=True)
assert hits, "bloodcellai_detect/__init__.py not found inside the package"
sys.path.insert(0, hits[0].rsplit("/bloodcellai_detect/", 1)[0])
import bloodcellai_detect as B
print("bloodcellai_detect", B.__version__, "| detector versions:", list(B.VERSIONS))


# %% CELL 1b : ONLY if you uploaded .rar / .zip files (not folders): unpack them into /kaggle/working/data
# Needs Internet ON once (notebook Settings, right panel) to install the unpacker. Safe to re-run; ~1 minute.
import glob, os, subprocess, shutil, zipfile

def find_one(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits[0] if hits else None

if shutil.which("bsdtar") is None:
    subprocess.run("apt-get update -qq && apt-get install -y -qq libarchive-tools", shell=True, check=True)

def unpack(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run(["bsdtar", "-xf", archive, "-C", out_dir], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:300]

DATA = "/kaggle/working/data"
grth = find_one("GrTh.rar")
bmp = find_one("GroundTruths_bmpformat.rar")
d1 = find_one("Dataset 1.rar")
d2 = find_one("Dataset 2.rar")
if d1 is None:   # the Zheng zip was NOT unpacked by Kaggle: unpack it here (Result 1/2 are never used)
    zheng_zip = find_one("w7cvnmn4c5*.zip")
    assert zheng_zip, "Zheng zip not found"
    zipfile.ZipFile(zheng_zip).extractall(f"{DATA}/zheng_zip")
    d1 = glob.glob(f"{DATA}/zheng_zip/**/Dataset 1.rar", recursive=True)[0]
    d2 = glob.glob(f"{DATA}/zheng_zip/**/Dataset 2.rar", recursive=True)[0]
for name, arc, out in (("Raabin JPG layout", grth, "raabin_grth"), ("Raabin BMP layout", bmp, "raabin_bmp"),
                       ("Zheng Dataset 1", d1, "zheng_d1"), ("Zheng Dataset 2", d2, "zheng_d2")):
    assert arc, f"{name}: archive not found"
    if not os.path.isdir(f"{DATA}/{out}"):
        unpack(arc, f"{DATA}/{out}")

def count(p, ext):
    return sum(1 for f in glob.glob(f"{p}/**/*", recursive=True) if f.lower().endswith(ext))
print("Raabin JPG :", count(f"{DATA}/raabin_grth", ".jpg"), "jpg   (expected 2290)")
print("Raabin BMP :", count(f"{DATA}/raabin_bmp", ".bmp"), "bmp   (expected 2290)")
print("Zheng D1   :", count(f"{DATA}/zheng_d1", ".bmp"), "bmp,", count(f"{DATA}/zheng_d1", ".png"), "png (expected 300, 300)")
print("Zheng D2   :", count(f"{DATA}/zheng_d2", ".bmp"), "bmp,", count(f"{DATA}/zheng_d2", ".png"), "png (expected 100, 100)")
print("Use in cell 3:", f"{DATA}/raabin_grth", f"{DATA}/raabin_bmp", f"{DATA}/zheng_d1", f"{DATA}/zheng_d2", sep="\n  ")


# %% CELL 2 : find the real folder of every attached dataset (fills the PATHS dict in cell 3)
import os, collections
IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
for root in sorted(set(glob.glob("/kaggle/input/datasets/*/*") + glob.glob("/kaggle/input/*"))):
    if not os.path.isdir(root) or os.path.basename(root) == "datasets":
        continue
    ext = collections.Counter()
    n_files = 0
    for dp, dn, fn in os.walk(root):
        for f in fn:
            n_files += 1
            ext[os.path.splitext(f)[1].lower()] += 1
        if n_files > 50000:
            break
    top = sorted(os.listdir(root))[:6]
    print(f"{root}\n    files>={n_files}  top-level={top}  ext={dict(ext.most_common(5))}")
# Look inside each dataset (ls two levels down) and choose the folder that CONTAINS the images/masks/labels.


# %% CELL 2b : look inside ONE dataset (change the path, run again for each dataset)
def tree(path, depth=3, max_items=8, _lvl=0):
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return
    dirs = [n for n in names if os.path.isdir(os.path.join(path, n))]
    files = [n for n in names if not os.path.isdir(os.path.join(path, n))]
    pad = "    " * _lvl
    for d in dirs[:max_items]:
        print(f"{pad}[{d}/]")
        if _lvl + 1 < depth:
            tree(os.path.join(path, d), depth, max_items, _lvl + 1)
    if len(dirs) > max_items:
        print(f"{pad}... +{len(dirs)-max_items} more folders")
    for f in files[:max_items]:
        print(f"{pad}{f}")
    if len(files) > max_items:
        print(f"{pad}... +{len(files)-max_items} more files")
tree("/kaggle/input/datasets/<user>/<slug>")


# %% CELL 3 : write + LOCK the ground truth BEFORE running the detector
import csv, datetime, shutil
GT_SRC = (glob.glob("/kaggle/input/**/ground_truth_new_datasets_PROPOSED.csv", recursive=True)
          + glob.glob("/kaggle/working/pkg/**/ground_truth_new_datasets_PROPOSED.csv", recursive=True))[0]
GT = "/kaggle/working/e1_new_gt.csv"
YOUR_NAME = "MUTTANGI CHANDRASEKHAR"          # your name = you checked every row

# dataset_id -> real folder, or "NOT_YET_ATTACHED"  (edit using the output of cell 2)
PATHS = {
    "raabin_wbc_seg_grth":            "NOT_YET_ATTACHED",   # folder that CONTAINS the extracted GrTh folder
    "raabin_wbc_seg_bmp":             "NOT_YET_ATTACHED",   # folder that CONTAINS Train_bmp, Train_GT_bmp, Test_bmp, Test_GT_bmp
    "zheng_wbc_seg_d1":               "NOT_YET_ATTACHED",   # the folder "Dataset 1" (bmp + png pairs)
    "zheng_wbc_seg_d2":               "NOT_YET_ATTACHED",   # the folder "Dataset 2"
    "erythrocytesidb":                "NOT_YET_ATTACHED",
    "ieee_leukocyte_seg5":            "NOT_YET_ATTACHED",
    "ieee_leukocyte_dataset2_pairs":  "NOT_YET_ATTACHED",
    "leukemia_image_6778":            "NOT_YET_ATTACHED",
    "raabin_wbc_cls_official":        "NOT_YET_ATTACHED",
}
shutil.copy(GT_SRC, GT)
rows = list(csv.DictReader(open(GT, newline="", encoding="utf-8")))
cols = list(rows[0].keys())
for r in rows:
    r["kaggle_path_hint"] = PATHS.get(r["dataset_id"], "NOT_YET_ATTACHED")
    r["filled_by"] = YOUR_NAME
    r["fill_date"] = datetime.date.today().isoformat()
with open(GT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# STOP: open each real folder and confirm true_format / true_role in the CSV are right.
# Only after that run the lock (it refuses to overwrite an existing lock).
from bloodcellai_detect.lock import lock
lock(GT)


# %% CELL 4 : run the FROZEN detector v1.4 (this is the result you report as the held-out test)
from bloodcellai_detect.audit import run_audit
results14, mut14, stats14 = run_audit(GT, "/kaggle/working/e1_new_v1_4", version="1.4",
                                      mutation_ids=[])          # e.g. ["raabin_wbc_seg"] to add the mutation test
# Every failure is printed with its evidence. Do not edit the ground truth to improve the score.


# %% CELL 5 : (optional, POST-HOC) experimental v1.5 on the same locked ground truth
results15, mut15, stats15 = run_audit(GT, "/kaggle/working/e1_new_v1_5", version="1.5", mutation_ids=[])
# Report v1.5 separately and label it post-hoc. It was written after seeing v1.4's segmentation gap.


# %% CELL 6 : quick look at one folder (any time)
fmt, role, conf, ev = B.detect_format_and_role("/kaggle/input/datasets/<user>/<slug>/<folder>", version="1.4")
print(fmt, role, conf, ev)
