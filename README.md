# bloodcellai-detect

## The problem this started from

Blood-cell datasets are published all over the place — Kaggle, GitHub, institutional
archives — and almost none of them agree on how to describe the same thing. One calls a
box a `<bndbox>` in an XML file. Another writes five numbers on a line of plain text.
A third hands you a folder per class with no boxes at all. A fourth marks each cell with
a single point instead of a rectangle. If you want to use more than one of these together,
somebody has to sit down, open each dataset, figure out what it actually is, and write
one-off code to convert it — every time, for every new dataset.

`bloodcellai-detect` is the part of BloodCellAI that does that first step automatically:
it looks at a folder and tells you, from its structure and file contents alone — never
from a filename or a dataset's reputation — what annotation format it's in and what role
it plays (a real detection benchmark, a classification-only collection, a segmentation
set, or genuinely unlabelled). **No trained model is involved.** It's a cascade of rules,
and when nothing matches confidently, it says so instead of guessing.

This is the code behind the manuscript's E1 experiment: 86.7% format accuracy, 93.3% role
accuracy, 0% abstain rate, across a real 15-dataset zoo — development-set figures, not a
claim about datasets never seen before (more on that honesty below). No dataset images
ship in this repository; it works on data you already have.

## What's actually in here

| Paper section | Where the code lives |
|---|---|
| 4.1 Format and role detection | `bloodcellai_detect` (Python) — tested, parity-checked against the original research notebook |
| 2.1–2.6 Universal data model, harmonization, YOLO export | `web/` (JavaScript, runs in a browser) — tested |
| 4.3 Structural defect validator | `bloodcellai_detect.validator` — tested |
| 4.4 Near-duplicate and leakage screen | `bloodcellai_detect.duplicates` — tested |
| 2.2 Cross-dataset class harmonization + 4.7 Merged-dataset export | `bloodcellai_detect.merge` — tested |
| Role-aware merging across any datasets you point it at | `merge.group_by_role` + the role-specific mergers below — tested |

Everything below is a walkthrough, not a spec sheet — read it the way you'd read someone
explaining their own tool to you.

## Getting it running

```bash
pip install git+https://github.com/chandrasekhar25phd7026/bloodcellai-detect.git     # core: standard library only
pip install "bloodcellai-detect[masks]"                                              # adds Pillow and NumPy, for v1.5's mask pixel checks
```

And then the smallest possible example — point it at a folder and ask what it is:

```python
from bloodcellai_detect import detect_format_and_role

fmt, role, confidence, evidence = detect_format_and_role("/path/to/dataset")                 # frozen v1.4
fmt, role, confidence, evidence = detect_format_and_role("/path/to/dataset", version="1.5")  # experimental
```

or from the command line:

```bash
bloodcellai-detect detect /path/to/dataset --detector 1.4
```

`evidence` isn't decoration — it's why the detector reached that conclusion, so a wrong
answer is debuggable instead of a black box.

### The formats and roles it knows about

| format | role |
|---|---|
| voc_xml, coco_json, yolo_txt, custom_bbox_json, custom_polygon_json | detection_benchmark |
| point_labels, classification_folders | classification_only |
| segmentation_masks | segmentation |
| unlabelled | unlabelled |
| abstain | abstain — a valid, scoreable answer, not a failure |

## Two detector versions, and why that matters for reading any number here

* **v1.4 (default, FROZEN).** The exact rule set behind the published manuscript numbers,
  extracted verbatim from the development notebook, with a parity test to prove it. Nobody
  edits `rules_v1_4.py` — that's the point of calling it frozen.
* **v1.5 (EXPERIMENTAL, post-hoc).** v1.4 only recognises segmentation data when a
  **file name** literally contains "mask" or "segmentation", which misses most real
  layouts (`masks/` folders, `GrTh1/`, `SegmentationClass/`, a `001.bmp` sitting next to a
  `001.png`). v1.5 adds directory-name and same-stem-pairing evidence to catch these,
  without ever overriding a strong VOC/COCO/YOLO/point/polygon result underneath. It was
  built after watching v1.4 fail on two real datasets, then fixed in direct response to
  that failure — which makes any number from v1.5 on those same two datasets a regression
  check, not independent evidence of generalizing to new data. Report it as post-hoc,
  always separately from v1.4's frozen numbers.

## The structural validator (Section 4.3)

```python
from bloodcellai_detect.validator import validate_annotations, run_defect_benchmark

flags = validate_annotations(objects, width, height)   # one flag-set per object: zero_area / out_of_bounds / duplicate
```

This one is deliberately narrow: geometry and duplication only, never image content. A box
with the wrong class label, or one shifted by a few pixels but still in bounds, looks
exactly like a correct box to this code — and the manuscript reports that as
`STRUCTURALLY_UNDETECTABLE`, not something quietly patched over. `run_defect_benchmark`
reproduces the manuscript's own injection-and-scoring method on objects you parse
yourself (`parse_voc_xml` and `parse_malaria_polygon` are included as convenience readers).

## The near-duplicate and leakage screen (Section 4.4)

```python
from bloodcellai_detect.duplicates import scan_datasets   # needs: pip install "bloodcellai-detect[validate]"

result = scan_datasets({"dataset_a": "/path/a", "dataset_b": "/path/b"})
```

Average-hash plus difference-hash (128 bits combined), with banded candidate search so it
stays usable on tens of thousands of images, plus an exact filename-stem check across
datasets. A real lesson from using this on our own data: the result carries **two**
different lists, and they're not interchangeable. `review_sample` is capped on purpose,
so a person can actually look at a representative handful by eye — it is *not* the
complete set of real candidates. `all_pairs` is the complete, uncapped list, and it's the
one to filter and pass on for anything you intend to exclude. We found this out the hard
way: an earlier pass of our own used `review_sample` for exclusion and silently
under-counted real duplicates by about 6%. Lesson kept in the code's own docstring, not
just in our notes.

Neither this module nor anything else here infers patient or slide identifiers from image
bytes — that stays a by-hand documentation task, stated plainly rather than quietly assumed
away.

## Union ontology and a reliable merged-dataset export (Sections 2.2 and 4.7)

```python
from bloodcellai_detect.merge import build_union_ontology, merge_datasets

ontology = build_union_ontology(datasets, overrides={("malaria_polygon", "Uninfected"): "RBC"})
result = merge_datasets(datasets_with_images, ontology, duplicate_pairs=confirmed_duplicates,
                         out_dir="merged", prefer_dataset_order=["txl_pbc", "bccd"])
```

The rule this whole module is built around: a native class name is never silently dropped
and never silently guessed into the wrong bucket. Each one is placed by exactly one of
three disclosed routes — `IDENTICAL_NAME` when two datasets already use the same label,
an `overrides` entry you stated yourself, or `UNMATCHED_KEPT_AS_NEW` when neither applies —
and every single placement shows up in the returned report, not just the ones that went
smoothly.

`duplicate_pairs` only excludes what you've actually confirmed are real duplicates (by eye,
on a sample) — this function never auto-excludes a candidate on its own. And when two
datasets have a documented, known relationship (one is a re-annotation of the other, say),
`prefer_dataset_order` lets you say which side should be kept, instead of falling back to
an alphabetical tiebreak that has no idea one annotation is better than the other.

Three more things came out of using this for real, and they're built in now, not left as
an afterthought:

```python
from bloodcellai_detect.merge import export_class_mapping_audit, verify_class_counts, audit_export

export_class_mapping_audit(ontology, native_object_counts, "mapping.csv")   # every native->union
                                                                             # decision, machine-readable
verify_class_counts(ontology, native_object_counts)                        # asserts nothing was
                                                                             # lost or gained at the mapping step
audit_export(out_dir)                                                      # every exported image has
                                                                             # exactly one valid label file
```

`merge_datasets` also writes a `provenance.csv` tracing every merged file back to its
source dataset and original filename, now with a SHA-256 checksum for both the image and
the label file — so anyone can verify, byte for byte, that what they downloaded is what was
actually reported.

## Role-aware merging: point it at any folder of datasets, not just ours

```python
from bloodcellai_detect.merge import group_by_role

groups, evidence = group_by_role({"a": "/path/a", "b": "/path/b", "c": "/path/c"})
# groups = {"classification_only": {...}, "detection_benchmark": {...}, "segmentation": {...}, ...}
```

This runs the frozen detector on every dataset you hand it and sorts them by role
automatically. Each role then merges the way it actually needs to:

- **Classification** needs no new machinery at all. `load_classification_folder`
  represents a classification image as one object, box = the whole frame — which is
  exactly what `merge_datasets` already expects, so the same two functions above just work.
- **Detection** needs a parser for its specific format first (VOC XML, YOLO text, COCO
  JSON, or your own), then the same merge functions — `examples/reproduce_merge.py` is a
  full worked example.
- **Segmentation** pools image/mask pairs and runs the duplicate screen across them
  (`merge_segmentation_datasets`), but it does **not** try to reconcile different
  datasets' mask pixel-value conventions on your behalf — that's named as a real
  limitation in the code, not quietly guessed at.

`examples/merge_by_role.py` walks through all three roles end to end.

## Reproducing the manuscript's own tables

```bash
python examples/reproduce_e4_e7.py defects    /path/to/bccd /path/to/malaria_polygon
python examples/reproduce_e4_e7.py duplicates /path/to/locked_ground_truth.csv   # from `bloodcellai-detect lock`
```

And the full cross-dataset merge from Section 4.7 — four real datasets, one harmonized
export, every exclusion disclosed — is in `examples/reproduce_merge.py`.

## Running the E1 audit on your own datasets

1. Copy `examples/ground_truth_*.csv`. Fill in `kaggle_path_hint` with the real folder (or
   `NOT_YET_ATTACHED`), check `true_format` and `true_role` against the real folder or its
   documentation, and put your own name in `filled_by`.
2. `bloodcellai-detect lock --gt my_ground_truth.csv` — validates the taxonomy and writes a
   SHA-256 lock; refuses to silently overwrite one that already exists.
3. `bloodcellai-detect audit --gt my_ground_truth.csv --out e1_out --detector 1.4 --mutation-ids bccd nih_malaria`
   — refuses to run at all if the CSV changed after locking. You get accuracy with Wilson
   95% confidence intervals, the full role confusion matrix, every failure with its
   evidence, a six-type mutation stress test, and `table1`/`table2` CSVs ready to drop into
   a manuscript.

Write down the true format and role **before** you run the detector. Don't edit the ground
truth afterward to make the detector look better — that's the entire point of locking it.

## What this doesn't do, said plainly

- It's heuristic and file-content based, and it hasn't been tested outside the datasets
  used so far. The v1.4 rules were revised four times in direct response to failures on the
  same 15 datasets later used for evaluation — so the headline accuracy is a development-set
  figure, not an estimate of performance on datasets it has never seen.
- The segmentation rule was never exercised on a real mask-bearing dataset in the original
  15-dataset run (the attached LISC copy had no masks at all).
- A segmentation dataset that also ships polygon/LabelMe JSON can get read as polygon
  detection by v1.4 instead.
- Formats outside this taxonomy — CIFAR-style binary batches, flat images with a CSV of
  labels, and so on — aren't recognised, and should abstain rather than being silently
  misread (if you find one that isn't abstaining correctly, that's a real bug; please file it).
- `max_files_scanned=20000` caps the scan; a very large dataset is judged on a sample, not
  every file.
- `validate_annotations` cannot catch a wrong class label or a small in-bounds shift —
  both look structurally identical to a correct box. Reported as
  `STRUCTURALLY_UNDETECTABLE`, not patched around.
- The duplicate hashes can't tell two perfectly flat, uniform-color images of different
  colors apart from each other — a known property of these two specific hash algorithms,
  with a test that exercises it directly rather than hiding it.
- The near-duplicate search is approximate by design (banded/LSH-style): a genuine
  near-duplicate whose bits happen to differ across every single band gets missed. More
  bands trade recall for more candidates to review by eye.
- Nothing here infers patient or slide identifiers from image bytes.
- `merge_segmentation_datasets` pools masks; it does not reconcile different mask
  pixel-value conventions across datasets.

## Running the tests yourself

```bash
pip install -e ".[test]" && pytest -q
```

The tests build small synthetic folders reproducing the layouts discussed above — they are
not real data, and nothing here requires downloading anything to run.

## The browser tool — no install at all

`web/blood-cell-dataset-detector.html` is one self-contained page. Open it directly; there's
no Python to install. It runs the same detector as this package, ported to JavaScript and
parity-tested against it, and for five formats it builds a harmonized YOLO export you
download as a `.zip` — entirely client-side, nothing ever leaves your browser. See
`web/README.md` for exactly what harmonization covers and what it deliberately changed from
the reference implementation.

## Where each file actually comes from

See `docs/PROVENANCE.md` — which Kaggle notebook each module was extracted from, and every
real bug found and fixed since the first release, in the order we found them. Nothing here
is retouched after the fact to look cleaner than the process actually was.

## Data licenses

No dataset images ship in this repository. Every public dataset keeps its own license and
terms — check them yourself before redistributing anything derived from them. The merged,
harmonized dataset described in Section 4.7 is archived and licensed separately; see its own
Zenodo record, linked from `CITATION.cff`.

## Citing this

See `CITATION.cff` — Zenodo mints a DOI automatically whenever a GitHub release here is
archived.

## License

MIT, for the code. Confirm this choice with your supervisor and institution before
publishing, if you haven't already.
