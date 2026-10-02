"""Helpers ``figures/rollout_count.py`` inherits from its sibling scripts.

The rollout-count generator in the research repository imported three sibling
scripts: ``issue1901_singleturn_retrieval_final`` (the retrieval scorer),
``issue1901_figure2_five_rollout_scaling`` (the scaling rescore), and
``issue779_fitter_fair_comparison`` (the fitter). Only the first contributes to
the figure path ported here, and only through the four names below, copied
unchanged from ``scripts/issue1901_singleturn_retrieval_final.py`` (commit
``94d6f8838f3``) of the ``explore-persona-space`` research repository.

The scoring path those scripts serve -- enumerating all 31 subsets of five
on-policy answer draws and bootstrapping their metrics -- consumed GPU-side
activation banks and is not part of this figure package; the summaries it
produced are committed under ``records/`` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

BOOT_N = 2_000
BOOT_SEED = 190_141


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()
