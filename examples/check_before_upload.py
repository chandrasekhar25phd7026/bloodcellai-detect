"""Run this on YOUR computer BEFORE uploading to Kaggle. Edit the four paths, then: python check_before_upload.py
It checks that the extracted folders have the structure the E1 ground truth was written for."""
from pathlib import Path

GRTH_FOLDER   = r"D:\datasets\raabin\grth_layout"        # must CONTAIN the extracted GrTh folder
BMP_FOLDER    = r"D:\datasets\raabin\bmp_layout"         # must CONTAIN Train_bmp, Train_GT_bmp, Test_bmp, Test_GT_bmp
ZHENG_D1      = r"D:\datasets\zheng\Dataset 1"           # the folder with the .bmp and .png pairs
ZHENG_D2      = r"D:\datasets\zheng\Dataset 2"

def count(p, ext):
    return sum(1 for f in Path(p).rglob("*") if f.is_file() and f.suffix.lower() == ext)

def check(label, ok, detail):
    print(("OK   " if ok else "FIX  ") + label + ": " + detail)

g = Path(GRTH_FOLDER)
check("Raabin JPG layout", count(g / "GrTh" / "Original", ".jpg") == 1145 and count(g / "GrTh" / "Ground Truth", ".jpg") == 1145,
      f"Original={count(g/'GrTh'/'Original', '.jpg')}  Ground Truth={count(g/'GrTh'/'Ground Truth', '.jpg')}  (expected 1145 and 1145 under GrTh/Original and GrTh/Ground Truth)")
b = Path(BMP_FOLDER)
counts = {d: count(b / d, ".bmp") for d in ("Train_bmp", "Train_GT_bmp", "Test_bmp", "Test_GT_bmp")}
check("Raabin BMP layout", counts == {"Train_bmp": 912, "Train_GT_bmp": 912, "Test_bmp": 233, "Test_GT_bmp": 233}, str(counts) + "  (expected 912, 912, 233, 233)")
for label, folder, n in (("Zheng Dataset 1", ZHENG_D1, 300), ("Zheng Dataset 2", ZHENG_D2, 100)):
    f = Path(folder)
    check(label, count(f, ".bmp") == n and count(f, ".png") == n,
          f"bmp={count(f, '.bmp')}  png={count(f, '.png')}  (expected {n} and {n}; the folder must NOT contain Result files)")
print("\nAlso check: no .rar files inside any folder you upload.")
