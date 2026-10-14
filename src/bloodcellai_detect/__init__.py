"""bloodcellai-detect: rule-based annotation-format and role detection, a structural
annotation validator, and a near-duplicate/leakage screen, for blood-cell imaging datasets.

- `detect_format_and_role` (this module): format/role detection (Section 4.1).
- `bloodcellai_detect.validator`: structural defect validator (Section 4.3). Pure
  standard library; always importable.
- `bloodcellai_detect.duplicates`: near-duplicate and leakage screen (Section 4.4).
  Needs Pillow and NumPy: pip install "bloodcellai-detect[validate]".
"""
from .detector import detect_format_and_role, VERSIONS

__version__ = "1.1.0"
__all__ = ["detect_format_and_role", "VERSIONS", "__version__"]
