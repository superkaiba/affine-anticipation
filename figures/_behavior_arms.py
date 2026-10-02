"""Behavior and projection-arm names the combined behavior-transfer figure needs.

The EPS figure script ``scripts/issue1739_old_combined_figure.py`` imports these
from its sibling ``scripts/issue1739_four_method_figure.py``, which also carries
the fitting machinery (scipy, the issue-1739 arms module, absolute machine-local
input roots) that this release does not ship.  Only the four
name tuples and the hashing helper the figure actually uses are copied here,
verbatim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

METHODS = (
    "answer_direction_on_context",
    "mapped_answer",
    "context_native",
    "real_answer",
)
LABELS = (
    "Answer → context",
    "Answer → mapped answer",
    "Context → context",
    "Answer → real answer",
)
BEHAVIORS = ("sycophancy", "hallucination", "evil")
HEADINGS = ("Sycophancy", "Hallucination", "Harmful compliance")


def sha(path: Path) -> str:
    """Hash exact cached bytes for provenance and completion checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
