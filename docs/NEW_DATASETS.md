# Datasets proposed for the segmentation / classification extension of E1

See `examples/ground_truth_new_datasets_PROPOSED.csv` (the `true_*` values are PROPOSALS taken from the dataset
descriptions; confirm each one against the real folder before locking).

| id | role (proposed) | why it is useful | access |
|---|---|---|---|
| raabin_wbc_seg_grth, raabin_wbc_seg_bmp | segmentation | nucleus/cytoplasm ground truths for 1,145 cells in two packagings (JPG folders and BMP train/test folders): inspected on the real files | raabindata.com |
| erythrocytesidb | segmentation | RBC masks for sickle-cell smears | gated agreement |
| zheng_wbc_seg_d1, zheng_wbc_seg_d2 | segmentation | expert masks (nuclei white, cytoplasm gray, background black); a reported Zenodo copy puts masks in `RealMask/` and `BinaryMask/` folders whose file names do not contain "mask" | Mendeley Data 10.17632/w7cvnmn4c5.1 |
| ieee_leukocyte_seg5 | segmentation | LabelMe JSON next to PNG label maps: tests rule order | IEEE DataPort |
| ieee_leukocyte_dataset2_pairs | segmentation | image and mask side by side with the same stem (`001.bmp`, `001.png`) | IEEE DataPort |
| leukemia_image_6778 | classification | normal vs leukemic cells (reported 6,778 images) | verify on Kaggle |
| raabin_wbc_cls_official | classification | official layout of a dataset that earlier appeared as a flat-image + CSV export | raabindata.com |

Rules for using them: lock the ground truth first; run frozen v1.4; report every failure; run v1.5 only afterwards and label it post-hoc;
check each licence before redistributing anything.
