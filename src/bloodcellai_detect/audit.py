"""E1 audit: accuracy with Wilson 95% CIs, role confusion matrix, every failure listed,
and the six-type mutation stress test. Ported from `paperA_e1_audit.py` (notebook cell 299, v1.4).
"""
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .detector import detect_format_and_role
from .rules_v1_4 import IMAGE_EXTS
from .lock import verify_lock_or_die, read_rows

MUTATION_SEED = 42
N_MUTATIONS_PER_DATASET = 10
MUTATION_SUBSAMPLE_SIZE = 40
MUTATION_TYPES = ["renamed_folders", "extra_nesting", "mixed_image_formats",
                  "missing_labels", "stray_files", "empty_classes"]
UNSET = ("", "FILL_ME_IN", "NOT_YET_ATTACHED")


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def run_main_audit(gt_rows, version="1.4"):
    results = []
    for row in gt_rows:
        hint = row["kaggle_path_hint"].strip()
        if hint in UNSET:
            results.append({**row, "pred_format": "SKIPPED", "pred_role": "SKIPPED",
                            "correct_format": None, "correct_role": None,
                            "evidence": {"reason": "path not provided / NOT_YET_ATTACHED"}})
            continue
        fmt, role, conf, evidence = detect_format_and_role(Path(hint), version=version)
        results.append({**row, "pred_format": fmt, "pred_role": role, "confidence": conf,
                        "correct_format": fmt == row["true_format"],
                        "correct_role": role == row["true_role"], "evidence": evidence})
    return results


def summarize(results, label=""):
    scored = [r for r in results if r["correct_format"] is not None]
    if not scored:
        print(f"[{label}] nothing was scored; check the paths.")
        return None
    n = len(scored)
    fc = sum(1 for r in scored if r["correct_format"])
    rc = sum(1 for r in scored if r["correct_role"])
    fl, fh = wilson_ci(fc, n)
    rl, rh = wilson_ci(rc, n)
    ab = sum(1 for r in scored if r["pred_format"] == "abstain") / n
    print(f"\n=== E1 audit summary [{label}] (n={n}) ===")
    print(f"Format accuracy: {fc}/{n} = {fc/n:.3f}  (95% CI {fl:.3f}-{fh:.3f})")
    print(f"Role accuracy:   {rc}/{n} = {rc/n:.3f}  (95% CI {rl:.3f}-{rh:.3f})")
    print(f"Abstain rate:    {ab:.3f}")
    print("\n--- Every failure, with evidence ---")
    for r in scored:
        if not r["correct_format"] or not r["correct_role"]:
            print(f"  {r['dataset_id']:24s} true=({r['true_format']}, {r['true_role']})  "
                  f"pred=({r['pred_format']}, {r['pred_role']})  evidence={r['evidence']}")
    roles = sorted({r["true_role"] for r in scored} | {r["pred_role"] for r in scored})
    print("\n--- Role confusion matrix (true \\ pred) ---")
    corner = "true\\pred"
    print(f"{corner:22s}" + "".join(f"{x:22s}" for x in roles))
    for t in roles:
        print(f"{t:22s}" + "".join(f"{sum(1 for r in scored if r['true_role']==t and r['pred_role']==p):<22d}" for p in roles))
    return {"n": n, "format_acc": fc / n, "format_ci": (fl, fh), "role_acc": rc / n,
            "role_ci": (rl, rh), "abstain_rate": ab}


def build_subsample(src_root: Path, dst_root: Path, n=MUTATION_SUBSAMPLE_SIZE, seed=0):
    rng = random.Random(seed)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    all_files = [p for p in src_root.rglob("*") if p.is_file()]
    if not all_files:
        return dst_root
    for f in rng.sample(all_files, min(n, len(all_files))):
        dest = dst_root / f.relative_to(src_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
    return dst_root


def apply_mutation(root: Path, mutation_type: str, seed: int):
    rng = random.Random(seed)
    if mutation_type == "renamed_folders":
        for d in list(root.iterdir()):
            if d.is_dir():
                d.rename(root / ("x" + hashlib.md5(d.name.encode()).hexdigest()[:8]))
    elif mutation_type == "extra_nesting":
        wrapper = root / "extra" / "nested" / "folders"
        wrapper.mkdir(parents=True)
        for item in list(root.iterdir()):
            if item != root / "extra":
                shutil.move(str(item), str(wrapper / item.name))
    elif mutation_type == "mixed_image_formats":
        try:
            from PIL import Image
        except ImportError:
            return
        imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        for p in rng.sample(imgs, min(len(imgs), max(1, len(imgs) // 2))):
            try:
                im = Image.open(p).convert("RGB")
                im.save(p.with_suffix(".bmp" if p.suffix.lower() != ".bmp" else ".jpg"))
                p.unlink()
            except Exception:
                pass
    elif mutation_type == "missing_labels":
        labels = [p for p in root.rglob("*") if p.suffix.lower() in {".xml", ".json", ".txt"}]
        for p in rng.sample(labels, min(len(labels), max(1, len(labels) // 3))):
            p.unlink()
    elif mutation_type == "stray_files":
        for name in [".DS_Store", "Thumbs.db", "readme_junk.tmp"]:
            (root / name).write_text("stray file for mutation robustness test")
    elif mutation_type == "empty_classes":
        (root / "empty_class_zzz").mkdir(exist_ok=True)
    else:
        raise ValueError(mutation_type)


def run_mutation_stress_test(gt_rows, work_dir: Path, dataset_ids, version="1.4"):
    by_id = {r["dataset_id"]: r for r in gt_rows}
    out = []
    for ds in dataset_ids:
        row = by_id.get(ds)
        if row is None or row["kaggle_path_hint"].strip() in UNSET or not Path(row["kaggle_path_hint"]).exists():
            print(f"[mutation] skipping '{ds}' (not in ground truth, no path, or path missing)")
            continue
        src = Path(row["kaggle_path_hint"])
        for i in range(N_MUTATIONS_PER_DATASET):
            mtype = MUTATION_TYPES[i % len(MUTATION_TYPES)]
            seed = MUTATION_SEED + i
            copy_root = Path(work_dir) / ds / f"mut{i:02d}_{mtype}"
            build_subsample(src, copy_root, seed=seed)
            apply_mutation(copy_root, mtype, seed=seed)
            fmt, role, _, _ = detect_format_and_role(copy_root, version=version)
            out.append({"dataset_id": ds, "mutation_index": i, "mutation_type": mtype,
                        "true_format": row["true_format"], "true_role": row["true_role"],
                        "pred_format": fmt, "pred_role": role,
                        "correct_format": fmt == row["true_format"], "correct_role": role == row["true_role"]})
    return out


def summarize_mutation_robustness(res):
    if not res:
        print("\nNo mutation results.")
        return
    by = defaultdict(list)
    for r in res:
        by[r["mutation_type"]].append(r["correct_role"])
    print("\n=== Mutation robustness (role accuracy by mutation type) ===")
    for m, o in sorted(by.items()):
        lo, hi = wilson_ci(sum(o), len(o))
        print(f"  {m:22s} n={len(o):3d}  acc={sum(o)/len(o):.3f}  (95% CI {lo:.3f}-{hi:.3f})")


def write_tables(results, mutation_results, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table1_dataset_zoo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "true_role", "true_format", "n_images_expected", "n_classes_expected", "license", "auto_detected"])
        for r in results:
            auto = ("SKIPPED" if r["pred_format"] == "SKIPPED" else
                    "correct" if (r["correct_format"] and r["correct_role"]) else
                    "abstain" if r["pred_format"] == "abstain" else "wrong")
            w.writerow([r["dataset_id"], r["true_role"], r["true_format"], r["n_images_expected"],
                        r["n_classes_expected"], r["license"], auto])
    with open(out_dir / "table2_audit_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "true_format", "pred_format", "correct_format", "true_role",
                    "pred_role", "correct_role", "confidence", "evidence"])
        for r in results:
            if r["pred_format"] == "SKIPPED":
                continue
            w.writerow([r["dataset_id"], r["true_format"], r["pred_format"], r["correct_format"],
                        r["true_role"], r["pred_role"], r["correct_role"], r.get("confidence"), json.dumps(r["evidence"])])
        for r in mutation_results:
            w.writerow([f"{r['dataset_id']}__mut{r['mutation_index']:02d}_{r['mutation_type']}",
                        r["true_format"], r["pred_format"], r["correct_format"], r["true_role"],
                        r["pred_role"], r["correct_role"], "", "mutation_test"])
    print(f"Wrote tables to {out_dir}")


def run_audit(gt_csv, out_dir, version="1.4", mutation_ids=(), skip_mutation=False):
    lock = verify_lock_or_die(Path(gt_csv))
    rows = read_rows(Path(gt_csv))
    print(f"Ground truth verified against lock (sha256={lock['sha256'][:16]}...), {len(rows)} rows. Detector v{version}.")
    results = run_main_audit(rows, version=version)
    stats = summarize(results, label=f"main audit, non-mutated, detector v{version}")
    mres = []
    if not skip_mutation and mutation_ids:
        mres = run_mutation_stress_test(rows, Path(out_dir) / "mutation_copies", mutation_ids, version=version)
        summarize_mutation_robustness(mres)
    write_tables(results, mres, Path(out_dir))
    return results, mres, stats
