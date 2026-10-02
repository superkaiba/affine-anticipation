#!/usr/bin/env python3
"""Paper figure ``c1_cot_capability`` (section 4.5, chain of thought and capability).

Combines the manuscript's CoT and capability panels from frozen plotted data.
Panel A: answer prediction under four thinking conditions (thinking off at the
context, thinking on at the context, the best interior thinking-span state, and
the end of the chain of thought), as held-out R^2 and top-1 retrieval.
Panel B: held-out R^2 for questions answerable only with thinking versus in both
modes, read at the context and at the end of thought.
Panel C: held-out R^2 against the AA Intelligence Index across ten models.

No inference, fitting, aggregation, or statistical tests run here.  The data
snapshot records the original producer commits and verified source hashes.

Ported from EPS ``scripts/paper_fig_cot_capability.py``, the copy in the
``cot-fig-midcot`` working tree at ``.claude/worktrees/cot-fig-midcot`` (EPS
issue 1901 line).  That copy is the only one carrying BOTH the 2026-09-14
four-bar panel A (the ``p7_traj_t20`` best mid-CoT state the submitted figure
shows) and the 2026-09-16 shortened ``ASPECT = 0.295`` canvas the submitted PDF
measures at (942.857 x 279.725 pt = 13.095 x 3.885 in); the two 2026-09-11
copies under ``/mnt/eps-data`` predate both and draw three bars.  The input is
that working tree's ``figures/paper/c1_cot_capability_data.json`` snapshot
(sha256 7adf9ef9aa9206032054203855ed4b66c27a573cbc403410f45b27656d6b7ed7),
copied to ``records/figure_data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from context_answer.plotting.style import (
    INK,
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
    style_score_axis,
)

STEM = "c1_cot_capability"
SOURCE_REL = Path("figure_data/c1_cot_capability_data.json")

MID_COT = "#E1AA9E"  # 50% white tint of the CoT-end red: same family, earlier state
ASPECT = 0.295
"""Canvas height over width.  The export prints 1.63 in tall at the 5.5 in text width."""

TOP_MARGIN_IN = 0.61
"""Inches above the plot boxes: the one-line panel headings and the legend row."""

BOTTOM_MARGIN_IN = 0.67
"""Inches below the plot boxes: two-line tick labels and panel C's axis label."""

PANEL_COLUMNS = ((0.080, 0.190), (0.345, 0.195), (0.625, 0.355))
"""Left edge and width of each panel as a fraction of the canvas width."""

CELL_ORDER = ["p7_Aoff", "p7_A", "p7_traj_t20", "p7_D"]
CONDITION_COLORS = dict(
    zip(
        CELL_ORDER,
        [MUTED, ROLES["linear"].color, MID_COT, ROLES["nonlinear"].color],
        strict=True,
    )
)
BORDERS = dict(zip(CELL_ORDER, [":", "-", "-.", "--"], strict=True))
OFFSETS = {
    "q35_0p8b": (40, 65),
    "q35_2b": (12, 5),
    "q35_4b": (40, -24),
    "q35_9b": (12, 28),
    "q35_27b": (12, 0),
    "q36_27b": (12, -28),
    "q38_27b": (-12, 9),
    "o3_7b_i": (12, -5),
    "o31_32b_i": (12, -10),
    "q3_32b": (-12, 4),
}


def bar_with_interval(ax, x, value, bounds, width, cell, *, retrieval=False):
    """Draw a saved estimate and its endpoints, including non-enclosing intervals."""
    lo, hi = bounds
    if not np.isfinite([value, lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid estimate/interval: {value}, {bounds}")
    color = CONDITION_COLORS[cell]
    ax.bar(
        x,
        value,
        width=width * 0.86,
        facecolor="white" if retrieval else color,
        edgecolor=color,
        linestyle=BORDERS[cell],
        linewidth=1.8,
        hatch="///" if retrieval else None,
        zorder=2,
    )
    ax.vlines(x, lo, hi, color=MUTED, linewidth=1.8, zorder=4)


def draw_prediction(ax, data):
    """Panel A: the three shipped conditions plus the best interior thinking-span state."""
    conditions = data["panels"]["A"]["conditions"]
    assert [row["cell"] for row in conditions] == CELL_ORDER
    assert data["panels"]["A"]["n_evaluated"] == 33810
    width = 0.27
    for j, key in enumerate(["r2_corpus", "acc1"]):
        for i, row in enumerate(conditions):
            scores = row["metrics"]
            assert scores["n"] == 33810
            bar_with_interval(
                ax,
                j * 1.35 + (i - 1.5) * width,
                scores[key],
                scores[f"{key}_ci"],
                width,
                row["cell"],
                retrieval=j == 1,
            )
    ax.set_xticks([0, 1.35], [r"$R^2$", "Top-1\nretrieval"])
    ax.set_xlim(-0.72, 2.07)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(better_label("Score"))
    style_axis(ax)
    panel_header(ax, "", "", "A Answer prediction", kicker_y=1.025, title_y=1.025)


def draw_correctness(ax, data):
    """Preserve equal-dataset-weight estimates and intervals in panel B."""
    pooled = data["panels"]["B"]["readouts"]
    counts = {"necessary": 4522, "both_correct": 17693}
    width = 0.30
    readout_cells = {"context": "p7_A", "end_of_thought": "p7_D"}
    for i, readout in enumerate(["context", "end_of_thought"]):
        for j, (group, count) in enumerate(counts.items()):
            row = pooled[readout][group]
            assert row["n"] == sum(row["n_by_corpus"].values()) == count
            assert len(row["weights"]) == 7
            assert np.allclose(list(row["weights"].values()), 1 / 7)
            bar_with_interval(
                ax,
                j + (i - 0.5) * width,
                row["r2_corpus_mean"],
                row["r2_corpus_mean_ci"],
                width,
                readout_cells[readout],
            )
    ax.set_xticks([0, 1], ["Only with\nthinking", "Both\nmodes"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.61)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    style_axis(ax)
    panel_header(ax, "", "", "B Correctness groups", kicker_y=1.025, title_y=1.025)


def draw_capability(ax, data):
    """Retain all ten model labels, coordinates, and saved statistics in panel C."""
    rows = data["rows"]
    assert len(rows) == len(OFFSETS) == data["plotted_panel"]["n"] == 10
    assert {row["model_key"] for row in rows} == set(OFFSETS)
    assert np.isfinite([[r["aa_index"], r["test_r2"]] for r in rows]).all()
    style_score_axis(ax, y_min=0.59, y_max=0.756, y_step=0.04)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75])
    ax.set_xlim(-3, 64)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel(better_label("AA Intelligence Index"))
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    role = ROLES["linear"]
    ax.scatter(
        [r["aa_index"] for r in rows],
        [r["test_r2"] for r in rows],
        s=110,
        marker=role.marker,
        color=role.color,
        linewidths=1.9,
        zorder=3,
    )
    for row in rows:
        dx, dy = OFFSETS[row["model_key"]]
        key = row["model_key"]
        arrow = None
        if key in {"q35_0p8b", "q35_4b", "q35_9b", "q36_27b"}:
            arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.9}
        ax.annotate(
            row["label"],
            (row["aa_index"], row["test_r2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center" if key == "q35_0p8b" else ("left" if dx > 0 else "right"),
            va="center",
            color=INK,
            arrowprops=arrow,
            fontsize=15,
        )
    stats = data["plotted_panel"]
    ax.text(
        0.025,
        0.95,
        rf"Spearman $\rho = {stats['rho']:.3f}$, $p = {stats['p_uncorrected']:.3f}$",
        transform=ax.transAxes,
        va="top",
    )
    panel_header(ax, "", "", "C Model capability", kicker_y=1.025, title_y=1.025)


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Export the combined figure with unchanged values and complete provenance."""
    source = records_dir / SOURCE_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / STEM
    raw = source.read_bytes()
    snapshot = json.loads(raw)
    cot, capability = [snapshot[key]["data"] for key in ("cot", "capability")]
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=ASPECT)
    height_in = fig.get_figheight()
    # Margins are fixed in inches because the furniture they hold (heading line,
    # legend row, two-line tick labels) does not scale with the canvas.  Every
    # inch saved on the canvas therefore comes out of the plot boxes, not out of
    # the space the labels need.
    bottom = BOTTOM_MARGIN_IN / height_in
    height = 1.0 - (TOP_MARGIN_IN + BOTTOM_MARGIN_IN) / height_in
    if height <= 0:
        raise ValueError(f"margins {TOP_MARGIN_IN} + {BOTTOM_MARGIN_IN} exceed {height_in:.3f} in")
    axes = [fig.add_axes([left, bottom, width, height]) for left, width in PANEL_COLUMNS]
    fig.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor=color,
                linestyle=border,
                linewidth=2,
                label=row["label"],
            )
            for row, color, border in (
                (row, CONDITION_COLORS[row["cell"]], BORDERS[row["cell"]])
                for row in cot["panels"]["A"]["conditions"]
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        handlelength=1.1,
        columnspacing=0.7,
        handletextpad=0.5,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    draw_prediction(axes[0], cot)
    draw_correctness(axes[1], cot)
    draw_capability(axes[2], capability)
    exported = save_c2a_figure(
        fig,
        stem,
        include_width=frac,
        title="Answer predictability, chain of thought, and model capability",
        subject="Existing Qwen3-8B CoT comparisons and ten-model capability association.",
        creator=f"figures/{Path(__file__).name}",
    )
    record = {
        "source": str(SOURCE_REL),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_panels": snapshot,
        "panel_mapping": {"A": "cot.A", "B": "cot.B", "C": "capability"},
        "layout": "One horizontal row: CoT prediction, correctness groups, model capability",
        "changes": (
            "Single-row layout with one-line panel headings and tighter vertical spacing. "
            "2026-09-16: shortened to aspect 0.295 with inch-fixed vertical margins and an "
            "unpadded legend row. No estimate, interval, label, series, or font size changed. "
            "Panel C uses AA Intelligence Index, expanded in the caption. "
            "All estimates, intervals, model labels, and font sizes preserved. "
            "2026-09-14: panel A gains a fourth bar, the best interior thinking-span state "
            "(t=0.2, cell p7_traj_t20 from p7_traj__a3.json); no map refit."
        ),
        "maps_refit": False,
        "render": exported["record"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            Path(save_c2a_figure.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "output_sha256": {
            key: hashlib.sha256(Path(exported[key]).read_bytes()).hexdigest()
            for key in ("pdf", "png", "grayscale")
        },
    }
    meta_path = stem.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    plt.close(fig)
    written = [exported["pdf"], exported["png"], exported["grayscale"], meta_path]
    for path in written:
        print(f"  {path}")
    return written


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("records"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    main(args.records, args.out)


if __name__ == "__main__":
    _cli()
