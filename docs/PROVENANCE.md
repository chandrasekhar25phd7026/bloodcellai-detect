# Provenance: what came from where

Source notebooks (Kaggle): `bdqi-v2` (430 cells) and `bloodcellai2` (197 cells).

| file in this repository | origin |
|---|---|
| `src/bloodcellai_detect/rules_v1_4.py` | `bdqi-v2` cell 299 (`%writefile paperA_e1_audit.py`, v1.4): `_is_float`, `_looks_like_voc_xml`, `_find_classification_root`, `detect_format_and_role`, `_safe_read`, `_safe_json`, copied verbatim. Earlier versions of the same script are in cells 226, 231, 237/250, 254/265. |
| `src/bloodcellai_detect/lock.py` | `bdqi-v2` cell 225 (`paperA_e1_lock_ground_truth.py`). Adds rejection of placeholder `filled_by` values and a lock path derived from the CSV name. |
| `src/bloodcellai_detect/audit.py` | `bdqi-v2` cell 299, steps 2-5 (`wilson_ci`, `run_main_audit`, `summarize`, `build_subsample`, `apply_mutation`, `run_mutation_stress_test`, `write_tables`), with Kaggle paths removed and the detector version made selectable. |
| `src/bloodcellai_detect/rules_v1_5.py` | New. Written after reviewing v1.4's segmentation rule; design informed by the `MASK_DIRS` idea in the capability-card generator (`bdqi-v2` cell 132). |
| `tests/expected_v1_4.json` | Produced by executing the original notebook function on the synthetic fixtures. |
| `examples/ground_truth_e1_15_datasets_RECONSTRUCTED.csv` | Reconstructed from the draft ground truth in `bdqi-v2` cell 232 (15 of 21 rows). Replace with your own locked CSV. |

## Structural validator (src/bloodcellai_detect/validator.py)

| file | origin |
|---|---|
| `validator.py`: all functions | Ported from `bdqi-v2` cell 279 (`%writefile paperA_e4_defect_detection.py`), the script behind Section 4.3 (Tables 7-8). Cells 276 and 278 are byte-identical re-executions of the same script, not separate revisions. `validate_annotations`, `inject_defects`, `score_defect_type`, `parse_voc_xml`, `parse_malaria_polygon` are unchanged. `run_dataset` (hardcoded to BCCD + malaria_polygon, with Kaggle paths baked in) is generalized into `run_defect_benchmark`, which takes pre-parsed `{name: [(file_id, objects, width, height), ...]}` instead; `check_real_defects_bccd` is generalized into `check_real_defects` the same way. |

## Near-duplicate and leakage screen (src/bloodcellai_detect/duplicates.py)

| file | origin |
|---|---|
| `duplicates.py`: hashing, banding, stem-overlap functions | Ported from `bdqi-v2` cell 292 (`%writefile paperA_e7_duplicates_leakage.py`), the **third and final** of three revisions in the notebook (cells 282, 288, 292 — 11,756, 16,299 and 19,607 characters respectively); v288 added the exact filename-stem overlap check over v282, and v292 added per-dataset-pair-key review sampling (so one dataset's internal redundancy cannot starve every other pair's saved sample) and the stem-vs-hash content cross-check. `ahash_bits`, `dhash_bits`, `combined_fingerprint`, `hamming`, `band_key`, `find_near_duplicates`, `filename_stem_overlap`, `discover_images` are unchanged. `run` (which read dataset paths from one specific locked ground-truth CSV path and wrote fixed output filenames) is generalized into `scan_datasets`, which takes a plain `{dataset_id: path}` mapping; `scan_datasets_from_locked_gt` restores the original's CSV-reading convenience as an optional wrapper. |

Not packaged here, on purpose:

* `bdqi-v2` cells 130-140, the *dataset capability card generator* (`detect_annotation_type`, `dataset_card.py`):
  an earlier, richer tool (pairing coverage, box validity, class-name ontology, fingerprint) used for BCCD, Chula-RBC and the malaria
  subsets. It is a different code path from the E1 detector.
* `bdqi-v2` E2/E5/E6/E8 scripts (conversion fidelity, BDQI validation, naive-merge causes, runtime) and the Paper-B
  detection training code — out of scope for this repository (E2's round-trip fidelity numbers are reported in the
  manuscript but reproducing them is not yet packaged; the BDQI scripts belong to a separate companion paper).
* `bloodcellai2`: the original `bloodcell` package workflow (`UniversalDatasetBuilder.auto_register`, `UniversalBuilder`,
  `DatasetQualityGate`) and the preliminary six-dataset benchmark. `export_yolo` and the format-specific parsers from
  this package were ported separately, to `web/harmonize.js` — see that file's own provenance section below.

## Harmonization (web/harmonize.js)

| file | origin |
|---|---|
| `web/harmonize.js`: `parseVocXml`, `parseYoloTxt`, `parseCocoJson`, `cocoRecordToObjects` | Ported from `bloodcell/parsers.py` (`parse_pascal_xml`, `parse_yolo`, `parse_coco_json`) and `bloodcell/adapters.py` (`CocoAdapter.convert`'s bbox-to-normalized-box math), in the user-supplied `bloodcell` package (an earlier, six-dataset prototype this project superseded — see the caveat in `web/README.md`). Logic ported verbatim; hardcoded `CLASS_MAP` / `DATASET_REGISTRY` class lookups deliberately NOT ported (see `web/README.md`). |
| `web/harmonize.js`: `parsePointTxt` | Generalizes `bloodcell/parsers.py`'s `parse_chula_txt` / `parse_rbc_morphology_txt` (x, y, class with a fixed-size box) to use each image's real decoded size instead of a hardcoded 640×480 canvas. |
| `web/harmonize.js`: `buildClassRegistry`, `buildYoloExport` | Ported from `bloodcell/dataset_export.py`'s `export_yolo()` (contiguous class-id remap, `data.yaml`, per-split label files), with classes discovered from data rather than supplied by a registry. |
| Not ported | `bloodcell/quality_gate.py`, `quality_assessment.py`, `annotation_intelligence.py`, and the `preprocessing/`/`training/`/`transforms/` top-level packages from the same upload: BDQI/quality-related, out of scope for this repository (tracked as a separate companion paper). |
