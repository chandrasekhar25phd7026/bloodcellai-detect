import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .detector import detect_format_and_role, VERSIONS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bloodcellai-detect",
                                 description="Rule-based annotation-format and role detection for blood-cell imaging datasets.")
    ap.add_argument("--version", action="version", version=f"bloodcellai-detect {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect format and role of one dataset folder")
    d.add_argument("path")
    d.add_argument("--detector", default="1.4", choices=sorted(VERSIONS), help="detector rule version (default: frozen 1.4)")
    d.add_argument("--json", action="store_true")

    l = sub.add_parser("lock", help="validate and hash-lock a ground-truth CSV before running the audit")
    l.add_argument("--gt", required=True)

    a = sub.add_parser("audit", help="run the E1 audit against a locked ground-truth CSV")
    a.add_argument("--gt", required=True)
    a.add_argument("--out", default="e1_audit_out")
    a.add_argument("--detector", default="1.4", choices=sorted(VERSIONS))
    a.add_argument("--mutation-ids", nargs="*", default=[], help="dataset_ids to include in the mutation stress test")

    args = ap.parse_args(argv)
    if args.cmd == "detect":
        fmt, role, conf, ev = detect_format_and_role(Path(args.path), version=args.detector)
        if args.json:
            print(json.dumps({"format": fmt, "role": role, "confidence": conf, "evidence": ev, "detector": args.detector}, indent=2))
        else:
            print(f"format={fmt}  role={role}  confidence={conf}  detector=v{args.detector}")
            print(f"evidence={ev}")
        return 0
    if args.cmd == "lock":
        from .lock import lock
        lock(Path(args.gt))
        return 0
    if args.cmd == "audit":
        from .audit import run_audit
        run_audit(args.gt, args.out, version=args.detector, mutation_ids=args.mutation_ids)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
