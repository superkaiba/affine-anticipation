"""Post-training stage constants and the JSON writer the k5 figure needs.

The EPS figure step ``scripts/issue1902_k5_length_control.py::run_figure``
imports these from two sibling pipeline modules that are not part of this
release: ``STAGES`` / ``LAYER`` / ``N_FOLDS`` / ``_jsonable`` / ``_write_json``
from ``scripts/issue1902_lasttoken_comparison.py``, and ``STAGE_NAMES`` /
``STAGE_COLORS`` / ``STAGE_MARKERS`` from ``scripts/issue1902_k5_fits.py``.  The
definitions are copied here verbatim so the ported figure script keeps its
original values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

STAGES = ("B", "S", "D", "R")
LAYER = 31
N_FOLDS = 6

STAGE_NAMES = {"B": "Base", "S": "SFT", "D": "DPO", "R": "RLVR"}
STAGE_COLORS = {"B": "#C98A1B", "S": "#176B87", "D": "#7B3294", "R": "#5AAE61"}
STAGE_MARKERS = {"B": "s", "S": "o", "D": "D", "R": "^"}


def jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars and arrays to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write sorted, indented JSON atomically via a same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
