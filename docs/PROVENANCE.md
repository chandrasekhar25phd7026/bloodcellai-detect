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

## Union ontology and merged-dataset export (src/bloodcellai_detect/merge.py)

New code, not ported from an existing script. Written to close two gaps named in the
manuscript itself: Section 2.2 describes mapping native class names onto a union ontology but
only gives one worked example (Paper 2's own two-dataset case), never independently validated
within this paper; and the manuscript's own motivation (pooling datasets is "a manual,
error-prone job") was never demonstrated end-to-end as an actual produced result. `merge.py`
reuses `validator.py`'s `Obj`/`PerFile` types and `duplicates.py`'s `scan_datasets` output
(candidate pairs, reviewed and confirmed by a human before being passed in as
`duplicate_pairs`) rather than duplicating either. The never-silently-drop class-handling
policy is the same one already used in `web/harmonize.js`, applied here across datasets
instead of within one.

## txl_pbc loader (examples/reproduce_merge.py)

`load_txl_pbc` was written and verified against a real TXL-PBC download (2026-10-15):
`data.yaml` reads `names: ['WBC', 'RBC','Platelets']`, `nc: 3`; labels are standard
YOLO-normalized `class_id xc yc w h` with `\r\n` line endings; 882/252/126 images in
train/val/test. Running it reproduces exactly the 18,143-object total already reported for
txl_pbc in Table 6 (12,510 + 3,752 + 1,881), confirming it is the same dataset. `load_bbbc041`
is still a placeholder: a "malaria" dataset checked the same day contained 716 raw images and
no annotation file of any kind, so it cannot be used as-is.

## Role-aware merging (src/bloodcellai_detect/merge.py, extended)

New code. `group_by_role` reuses the existing detector (`detector.detect_format_and_role`,
Section 4.1) with no changes, purely to group datasets before merging. `load_classification_folder`
and the classification path through `merge_datasets` need no new core logic -- a classification
image is represented as one whole-frame object, the same convention already used in
`web/harmonize.js`. `merge_segmentation_datasets` is new, deliberately scoped to pooling and
deduplication only: reconciling different datasets' mask pixel-value semantics is out of scope
and named as a limitation in the function's own docstring, not attempted silently.

## malaria_polygon source, confirmed (2026-10-16)

Dataset: NIH-NLM-ThinBloodSmearsPf (National Library of Medicine, National Institutes of
Health, Bethesda, MD, USA), 193 patients, images acquired at Chittagong Medical College
Hospital, Bangladesh, under IRB#12972. Requested citation (from the dataset's own ReadMe.pdf):
Kassim, Y. M., Palaniappan, K., Yang, F., Poostchi, M., Palaniappan, N., Maude, R. J., Antani,
S., and Jaeger, S. "Clustering-Based Dual Deep Learning Architecture for Detecting Red Blood
Cells in Malaria Diagnostic Smears." IEEE Journal of Biomedical and Health Informatics, 2020.

`load_malaria_polygon` in `examples/reproduce_merge.py` was corrected after reading the
official ReadMe.pdf: GT `.txt` files and their images are NOT in the same folder (an earlier
version assumed a shared-folder, swapped-extension layout and would have silently returned
zero images on the real data). The real layout is `<Patient ID>/GT/<name>.txt` next to a
sibling `<Patient ID>/img/<name>.jpg` (seen capitalized "Img" on the NLM file index too,
matched case-insensitively). `parse_malaria_polygon` itself (the per-file parser) needed no
changes -- verified directly against the ReadMe's own worked example line.

## Real bug found and fixed (2026-10-16): overrides applied even when the data wasn't there

A real Kaggle run reported three of four datasets loading zero objects (`bccd`, `txl_pbc`,
`malaria_polygon` all 0 -- almost certainly a stale path/variable issue in that session), yet
`build_union_ontology`'s printed ontology still showed `malaria_polygon`'s three overridden
classes as if real. Root cause: Pass 1 applied every entry in `overrides` unconditionally,
never checking whether that `(dataset, native_name)` pair actually appeared in the dataset's
own discovered classes. Fixed: an override now only applies if the native class is genuinely
present; otherwise it is reported in the new `UnionOntology.unused_overrides` field and printed
loudly, rather than silently making a failed dataset look like it contributed classes. Covered
by two new regression tests in `tests/test_merge.py`.

## Real bug found and fixed (2026-10-17): review_sample under-counted real duplicates

A more exhaustive, independent audit of the bccd/txl_pbc merged export found 160 real
duplicate pairs where this project's own guidance (building the exclusion list from
`scan_datasets`'s `review_sample`) had only found 151. Root cause: `review_sample` is
deliberately capped per dataset-pair-key for human reviewability (see its own docstring), and
with few pair-keys present, that cap can be smaller than the true number of real candidates,
silently leaving some out. `scan_datasets` now also returns `all_pairs`, the complete,
uncapped candidate list, which is what the exclusion step should filter and use --
`review_sample` remains for visual spot-checking only. `examples/reproduce_merge.py` is
updated accordingly. Covered by a new regression test in `tests/test_duplicates.py`.

## Real design gap closed (2026-10-17): which side of a duplicate to keep

`merge_datasets` had only one tiebreak for a confirmed duplicate pair -- alphabetical by
dataset name -- which is an uninformed default, not a considered choice. For bccd/txl_pbc
specifically, txl_pbc's annotations are a documented re-labeling of (part of) bccd's images
(Section 4.6.3), so keeping bccd's original over txl_pbc's re-annotation whenever they
duplicate was the wrong way round. Added `prefer_dataset_order` to `merge_datasets` so a
real, stated preference can be supplied; the alphabetical rule remains only as the explicit,
disclosed fallback when no preference is given. Covered by a new test in `tests/test_merge.py`.

## New audit tooling added in response to external review (2026-10-17)

`export_class_mapping_audit`, `verify_class_counts`, and `audit_export` (all in merge.py)
are new code, not ported from anywhere. They answer three specific external review asks:
a machine-readable class-mapping table (source dataset, native label, canonical label,
mapping type, information loss, rationale, plus the real object count per native label),
a check that class counts are preserved exactly at the mapping step (before the separate,
already-disclosed duplicate-image exclusion), and an automated check that every exported
image has exactly one label file with valid class ids. `merge_datasets`'s provenance.csv
now also includes a sha256 checksum per exported image and label file. Found and fixed one
real bug while building this: the mapping-audit's rationale lookup did not originally
handle the UNMATCHED_KEPT_AS_NEW route at all (a native class with no identical-name match
and no override), silently mislabeling it as a "synonym" mapping with a blank rationale --
fixed, with a dedicated rationale entry and a regression test.

Also added: four schema-integrity edge-case tests requested by the same review (an empty
label file, a zero-height as well as zero-width box, a box exactly on the image boundary
correctly NOT flagged, and duplicate-line detection shown robust to whitespace differences
in the source text).

## README rewritten in narrative style, and dataset-packaging script added (2026-10-17)

README.md was rewritten from a reference-manual style into a narrative one, at the user's
request, with no change to any technical content, number, or limitation -- every fact in
the old version is still here, in the same words where precision mattered (version
numbers, accuracy figures, limitation wording).

`examples/package_dataset_for_zenodo.py` is new: it packages a real merge_datasets()
export into an upload-ready folder for a *separate* Zenodo dataset record (not this code
repository) -- running the real export-integrity audit first and refusing to package an
export that fails it, copying the real files, and writing a dataset-specific README and
CITATION.cff from the export's own real provenance.csv (counts are re-derived from the
file, never hardcoded). Tested end to end against a real synthetic export built with
merge_datasets() itself.

## Packaging script made robust to older exports without checksum columns (2026-10-17)

A real provenance.csv from the user's own Kaggle session (the Section 4.7 merge, 2,956
images, verified internally consistent: 203 bccd + 1,260 txl_pbc + 1,328 bbbc041 + 165
malaria_polygon) had only the original 5 columns -- it was produced by a package version
from before checksums were added. Rather than require re-running the merge again,
`package_dataset_for_zenodo.py` now computes image and label SHA-256 checksums directly
from the real files on disk at packaging time, regardless of what provenance.csv already
contains, and writes a fresh 7-column provenance.csv into the packaged output. Tested
against a synthetic reproduction of exactly this scenario (an old-format 5-column input).

## Release hygiene: removed .pytest_cache from the shipped archive (2026-10-17)

A build artifact (`.pytest_cache/`, left over from running the test suite locally) had been
included in the packaged zip -- harmless, but not something that belongs in a public source
release. Removed, and the repository was swept for any other build caches (__pycache__,
.egg-info, .DS_Store, .pyc, .ipynb_checkpoints -- none found beyond the one), any local
sandbox file paths leaked into shipped files (none found), and anything resembling an
embedded credential or secret (none found; the only "token"/"secret" matches were ordinary
parsing-token code and an unrelated npm package name in package-lock.json).
