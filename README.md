# bloodcellai-detect

Rule-based detection of the **annotation format** and the **role** (detection benchmark, classification only,
segmentation, unlabelled) of blood-cell imaging dataset folders. It uses structural rules and file contents only.
**No trained model is involved.** The detector can abstain instead of guessing.

This is the format/role-detection part of BloodCellAI. It is the code behind the E1 experiment of the
accompanying manuscript (format accuracy 13/15, role accuracy 14/15, abstain rate 0/15 on the 15-dataset zoo).
No dataset images are included in this repository.

## What's in this repository

| Paper section | Code here? |
|---|---|
| 4.1 Format and role detection | Yes — `bloodcellai_detect` (Python), tested, parity-checked against the original notebook |
| 2.1–2.6 Universal data model, harmonization, YOLO export | Yes — `web/` (JavaScript, runs in a browser, tested) |
| 4.3 Structural defect validator | Yes — `bloodcellai_detect.validator` (Python), tested |
| 4.4 Near-duplicate and leakage screen | Yes — `bloodcellai_detect.duplicates` (Python), tested |

### The structural validator (Section 4.3)

```python
from bloodcellai_detect.validator import validate_annotations, run_defect_benchmark

flags = validate_annotations(objects, width, height)   # one flag-set per object: zero_area / out_of_bounds / duplicate
```
`validate_annotations` is geometry- and duplication-only, by design: it never looks at image content, so a wrong
class label or a small in-bounds box shift looks exactly like a correct one. `run_defect_benchmark` reproduces the
manuscript's injection-and-scoring methodology on any objects you parse yourself (`parse_voc_xml` and
`parse_malaria_polygon` are included as convenience readers). `STRUCTURALLY_UNDETECTABLE` names the defect types the
validator cannot catch, reported as a disclosed limitation rather than hidden.

### The near-duplicate and leakage screen (Section 4.4)

```python
from bloodcellai_detect.duplicates import scan_datasets   # needs: pip install "bloodcellai-detect[validate]"

result = scan_datasets({"dataset_a": "/path/a", "dataset_b": "/path/b"})
```
Average-hash + difference-hash (128 bits combined), with banded (LSH-style) candidate search so it stays tractable
on tens of thousands of images, plus an exact filename-stem overlap check across datasets. Both the banding and the
hash choice are approximate by design, stated in the module's own docstring rather than hidden. Patient/slide-ID
metadata is never inferred from image bytes; `scan_datasets` does not attempt it.

### Reproduce the manuscript's Tables 7–10

`examples/reproduce_e4_e7.py` reproduces the structural-validator and duplicate-screen results on your own copies of
the real datasets:
```bash
python examples/reproduce_e4_e7.py defects    /path/to/bccd /path/to/malaria_polygon
python examples/reproduce_e4_e7.py duplicates /path/to/locked_ground_truth.csv   # from `bloodcellai-detect lock`
```

## Install

```bash
pip install git+https://github.com/chandrasekhar25phd7026/bloodcellai-detect.git     # core: standard library only
pip install "bloodcellai-detect[masks]"                                    # adds Pillow and NumPy (mask pixel checks in v1.5)
```

## Quick start

```python
from bloodcellai_detect import detect_format_and_role

fmt, role, confidence, evidence = detect_format_and_role("/path/to/dataset")                 # frozen v1.4
fmt, role, confidence, evidence = detect_format_and_role("/path/to/dataset", version="1.5")  # experimental
```

```bash
bloodcellai-detect detect /path/to/dataset --detector 1.4
```

## Formats and roles

| format | role |
|---|---|
| voc_xml, coco_json, yolo_txt, custom_bbox_json, custom_polygon_json | detection_benchmark |
| point_labels, classification_folders | classification_only |
| segmentation_masks | segmentation |
| unlabelled | unlabelled |
| abstain | abstain (a valid, scoreable answer) |

## Two detector versions (read before reporting numbers)

* **v1.4 (default, FROZEN).** The exact rule set behind the published numbers, extracted verbatim from the
  development notebook. A parity test checks that it reproduces the original notebook function's output.
  Do not edit `rules_v1_4.py`.
* **v1.5 (EXPERIMENTAL, post-hoc).** Adds directory-name and same-stem-pair evidence for segmentation datasets.
  v1.4 only recognises segmentation when a **file name** contains "mask"/"segmentation", so it misses the usual
  layouts (`masks/` folders, `GrTh1/`, `SegmentationClass/`, `RealMask/`, `001.bmp` + `001.png`). v1.5 never overrides a
  strong VOC/COCO/YOLO/point/comma-delimited result. It was developed on synthetic layouts and then revised after failing on two real
  Raabin-WBC layouts; those real files are therefore development data (see `docs/REAL_DATA_PREVIEW.md`).
  Report any v1.5 number as post-hoc, separately from frozen-v1.4 results.

## Reproduce the E1 audit on your own datasets

1. Copy `examples/ground_truth_*.csv`. Fill `kaggle_path_hint` with the real folder (or `NOT_YET_ATTACHED`),
   check `true_format` and `true_role` against the real folder or its documentation, and put your own name in `filled_by`.
2. `bloodcellai-detect lock --gt my_ground_truth.csv` (validates the taxonomy and writes a sha256 lock; refuses to overwrite).
3. `bloodcellai-detect audit --gt my_ground_truth.csv --out e1_out --detector 1.4 --mutation-ids bccd nih_malaria`
   The audit refuses to run if the CSV changed after locking. Outputs: accuracy with Wilson 95% CIs, the role
   confusion matrix, every failure with evidence, a six-type mutation stress test, and `table1`/`table2` CSVs.

Write the true format and role **before** running the detector. Do not edit the ground truth to make the detector look better.

## Known limitations

* Heuristic, file-content based; not tested outside the datasets used so far. The v1.4 rules were revised four times
  in response to failures on the same 15 datasets later used for evaluation, so the reported accuracy is a
  development-set figure, not a held-out estimate.
* The segmentation rule was never exercised on a real mask-bearing dataset in the 15-dataset run (the attached LISC copy had no masks).
* Segmentation datasets that also ship polygon/LabelMe JSON can be read as polygon detection by v1.4.
* Formats outside the taxonomy (for example CIFAR-style binary batches, flat images with CSV labels) are not recognised and should abstain or be misread.
* `max_files_scanned=20000` caps the scan; very large datasets are judged on a sample of files.
* `validate_annotations` cannot detect a wrong class label or a small in-bounds box shift — both look structurally identical to a correct box; this is reported as `STRUCTURALLY_UNDETECTABLE`, not fixed around.
* `duplicates.ahash_bits`/`dhash_bits` cannot distinguish two perfectly uniform (flat-color) images of different colours from each other — a disclosed property of these two specific hash algorithms, exercised directly in `tests/test_duplicates.py`.
* The near-duplicate search is approximate (banded/LSH-style): a true near-duplicate pair whose bits differ across every single band is missed; more bands trade recall for more candidate pairs to review.
* Neither module infers patient or slide identifiers from image bytes; that remains a by-hand documentation task.

## Tests

```bash
pip install -e ".[test]" && pytest -q
```

The tests build tiny synthetic folders that reproduce the layouts discussed above. They are not real data.

## The browser tool (detection + harmonization, no install)

`web/blood-cell-dataset-detector.html` is a single, self-contained page: open it directly, no Python
install needed. It runs the same detector as this package (ported to JavaScript, parity-tested against
it) and, for five formats, builds a harmonized YOLO export you download as a `.zip` — entirely
client-side; nothing is uploaded anywhere. See `web/README.md` for what harmonization covers and what it
deliberately changed from the original reference code.

## Provenance

See `docs/PROVENANCE.md` (where each file comes from in the original Kaggle notebooks).

## Data licences

This repository contains no dataset images. Each public dataset keeps its own licence and terms.
Check them before redistributing any converted data.

## Citation

See `CITATION.cff` (a DOI is minted by Zenodo when a GitHub release is archived).

## License

MIT (code). Confirm this choice with your supervisor and institution before publishing.
