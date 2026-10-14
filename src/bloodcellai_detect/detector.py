"""Public entry point: choose the detector version explicitly.

    from bloodcellai_detect import detect_format_and_role
    fmt, role, confidence, evidence = detect_format_and_role("path/to/dataset")            # frozen v1.4
    fmt, role, confidence, evidence = detect_format_and_role("path/to/dataset", version="1.5")  # experimental

The default is the frozen v1.4 so that results reproduce the published numbers.
"""
from pathlib import Path

from . import rules_v1_4, rules_v1_5

VERSIONS = {"1.4": rules_v1_4.detect_format_and_role, "1.5": rules_v1_5.detect_format_and_role}


def detect_format_and_role(root, version="1.4", max_files_scanned=20000):
    """Returns (format, role, confidence, evidence). 'abstain' is a valid answer."""
    if version not in VERSIONS:
        raise ValueError(f"unknown detector version {version!r}; choose from {sorted(VERSIONS)}")
    return VERSIONS[version](Path(root), max_files_scanned=max_files_scanned)
