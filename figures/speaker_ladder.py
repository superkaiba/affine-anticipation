"""Speaker/framing specialization-ladder reads, shared by the Section 4.4 figures.

Indexes the banked ladder cells and the leave-one-cell-out pooled fits by
(character, framing, model), and applies the cap-excluded substitution at the
one truncation-contaminated cell.

Ported from EPS ``scripts/issue2054_paper_r2_figs.py`` at commit a66fa4d3c1c
(EPS issue 2054) -- only the objects :mod:`figures.shared_speakers` uses.  The
rest of that script draws two figures that are not in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = ROOT / "records"

# Reader-facing display names live here and nowhere else: one label map, so a
# rename lands once. Internal cell ids never reach an axis label.
# (character key, framing key, two-line tick label)
POSITIONS: list[tuple[str, str, str]] = [
    ("assistant", "chat", "Assistant\nchat template"),
    ("assistant", "bare_text", "Assistant\nplain text"),
    ("helios", "attrib_quoted", "HELIOS\nship AI"),
    ("wren", "attrib_quoted", "Wren\nhelpful"),
    ("dana", "attrib_quoted", "Dana\nordinary"),
    ("vex", "attrib_quoted", "Vex\nvillain"),
]
PROVENANCE = "on_policy"
ARM = "context"
MODELS = ("base", "instruct")
MODEL_KEY = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}

# Cap-excluded refits of the one truncation-contaminated cell, keyed by
# (character, framing, model). Own-map value: the banked within-cell exclusion
# refit (bare text / capped removed). Pooled value: the companion refit, read
# from its artifact rather than pasted in, so the number in the figure is the
# number whose validation gate passed.
CAP_EXCLUDED = {("assistant", "bare_text", "instruct"): 0.38985271917718867}

LADDER: Path
LOCO: Path
CAP_POOLED_ARTIFACT: Path


def set_records(records: Path) -> None:
    """Point the ladder inputs at ``records``."""
    global LADDER, LOCO, CAP_POOLED_ARTIFACT
    ladder_dir = records / "issue_2054/specialization_ladder"
    LADDER = ladder_dir / "ladder.json"
    LOCO = ladder_dir / "loco_pooled.units.jsonl"
    CAP_POOLED_ARTIFACT = records / "issue_2054/caphit_pooled_refit/caphit_pooled_refit.json"


set_records(DEFAULT_RECORDS)


def load_cells() -> dict[tuple[str, str, str], dict]:
    """Index the ladder's context-arm on-policy units by (character, framing, model)."""
    payload = json.loads(LADDER.read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for unit in payload["units"]:
        if unit["arm"] != ARM or unit["provenance"] != PROVENANCE:
            continue
        out[(unit["character"], unit["framing"], unit["model"])] = unit
    return out


def load_loco() -> dict[tuple[str, str, str], float]:
    """Fold-mean held-out R^2 of the pooled map fit WITHOUT this speaker's cells."""
    out: dict[tuple[str, str, str], float] = {}
    for line in LOCO.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["arm"] != ARM or row["condition"] != PROVENANCE:
            continue
        key = (row["speaker"], row["framing"], MODEL_KEY[row["model"]])
        out[key] = float(np.mean([f["loco"]["r2"] for f in row["per_fold"]]))
    return out


def build_series(cells: dict[tuple[str, str, str], dict], loco: dict) -> dict:
    """Pull the six positions x two models out of the indexed cells, fail loud on a miss."""
    series: dict = {"labels": [label for _, _, label in POSITIONS], "n": [], "null": []}
    for model in MODELS:
        series[model] = {
            "ceiling": [],
            "pooled": [],
            "bias": [],
            "gain": [],
            "loco": [],
            "identity": [],
        }
    for character, framing, _ in POSITIONS:
        for model in MODELS:
            key = (character, framing, model)
            if key not in cells:
                raise KeyError(f"no {ARM}/{PROVENANCE} cell for {key}; have {sorted(cells)}")
            unit = cells[key]
            series[model]["ceiling"].append(unit["ceiling_r2"])
            series[model]["pooled"].append(unit["r2"]["pooled"])
            # The two minimal per-cell corrections of the specialization
            # ladder, CUMULATIVE: `bias` is z + mean(y - z) (shift only),
            # while `gain` is y_bar + alpha * (z - z_bar) -- a recentre onto
            # the target answer mean PLUS a global gain, so it already
            # contains the shift. Hence the two bars sit near-identical.
            series[model]["bias"].append(unit["r2"]["bias"])
            series[model]["gain"].append(unit["r2"]["gain"])
            series[model]["identity"].append(unit["aux_r2"]["identity_cell"])
            if key not in loco:
                raise KeyError(f"no LOCO row for {key}; have {sorted(loco)}")
            series[model]["loco"].append(loco[key])
            series["n"].append(unit["n_join"])
            series["null"].append(unit["banked_null_r2_pooled_p95"])
    return series


def _substituted_indices() -> list[int]:
    """Positions whose instruct bar is plotted at the cap-excluded refit."""
    return [
        idx
        for idx, (character, framing, _) in enumerate(POSITIONS)
        if (character, framing, "instruct") in CAP_EXCLUDED
    ]


def _apply_cap_excluded(
    values: list[float], indices: list[int], table: dict[tuple[str, str, str], float]
) -> list[float]:
    """Swap in the cap-excluded refit at the substituted positions."""
    out = list(values)
    for idx in indices:
        character, framing, _ = POSITIONS[idx]
        out[idx] = table[(character, framing, "instruct")]
    return out
