# First contact with real mask-bearing datasets (2026-10-13)

Three real downloads were inspected and run in a sandbox: Raabin-WBC ground truths (JPG layout and BMP layout of the same
data) and Zheng WBC Dataset 1 and Dataset 2. The true format/role of each was written down and hash-locked BEFORE the detector ran
(`segmentation_masks` / `segmentation` for all four rows).

| detector | result on the 4 rows | note |
|---|---|---|
| frozen v1.4 | 0/4 format, 0/4 role | HELD-OUT result. Raabin x2 read as `classification_folders`, Zheng x2 as `unlabelled` |
| v1.5 first draft (written from synthetic layouts) | 2/4 | Zheng found; both Raabin layouts missed |
| v1.5 revised (post-hoc) | 4/4 | revised after seeing why: JPEG masks have about 35 colours; the folder `Train_GT_bmp` was not matched; a wrapper folder called `GrTh` hid the real image/mask split |

The revised v1.5 was tuned on these exact files, so they are DEVELOPMENT data for v1.5 and say nothing about its accuracy on unseen datasets.
Mask-likeness measured on the real files (share of pixels near the 3 most common grey levels): masks 1.000, photographs 0.18 to 0.86.
The frozen v1.4 numbers remain a valid held-out result.
