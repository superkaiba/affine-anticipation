"""The answer-PCA useful-directions spectrum panel, drawn onto a caller's axes.

Shared panel drawer behind three figures in :mod:`figures.section42`, including
the paper stem ``c3_direction_r2_spectrum``.  Everything here is plot-only: it
reads the banked ``records/issue_779/plot3_redesign/plot3_redesign.json``
summary that the host script loads.

Ported from EPS ``scripts/issue779_plot3_redesign.py`` at commit 1cf55f86103
(EPS issue 779) -- only the three objects the host needs (``ROSTER``,
``PAPER_OFFSETS``, ``draw_spectrum_panel``).  The rest of that script
recomputes the spectrum from activation stores and persona-vector tensors,
which this release does not ship; the banked summary is the input instead.
"""

from __future__ import annotations

import numpy as np

from context_answer.plotting.style import (
    INK,
    METRIC_LABELS,
    MUTED,
    ROLES,
    better_label,
    style_axis,
)

# (artifact key, plotted label, provenance of the direction in the source bank).
ROSTER = [
    # The Chen et al. trait the paper calls harmful compliance; the source
    # artifact keeps its extraction-time name "evil".
    (
        "evil",
        "harmful compliance",
        "#779 persona vector r_B (HF issue779_monitoring/r_b/evil.pt)",
    ),
    (
        "sycophancy",
        "sycophancy",
        "#779 persona vector r_B (HF issue779_monitoring/r_b/sycophancy.pt)",
    ),
    (
        "hallucination",
        "hallucination",
        "#779 persona vector r_B (HF issue779_monitoring/r_b/hallucination.pt)",
    ),
    (
        "refusal",
        "refusal",
        "diff-of-means refuse−engage over #2356 armA consolidated v_C at block 19 "
        "(HF issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy)",
    ),
    (
        "assistant_axis",
        "assistant axis",
        "#2203 assistant axis, default-assistant − role-play diff-of-means "
        "(HF issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt)",
    ),
    (
        "casualness",
        "casualness",
        "#1434 writing-style persona vector (casual-vs-formal; "
        "HF issue1434_writingstyle/analysis_tensors/rb_writing_style.pt)",
    ),
    (
        "impoliteness",
        "impoliteness",
        "#1482 rb4 persona vector (local data/issue_779/r_b/impolite.pt, #779-recipe extraction)",
    ),
    # Distinct from the persona vector above: a diff-of-means over answer-side
    # activations, AdvBench harmful requests minus the neutral probe pool, so
    # the label names the corpus that built it.
    (
        "harmful_compliance",
        "AdvBench harmful requests",
        "#658 persona-vector primary diffmeans recipe "
        "(HF issue658_theory_assumptions/store/r_b.pt)",
    ),
    (
        "correctness_math",
        "correctness (math)",
        "#2388 matched correct−incorrect t1 direction over math spread contexts",
    ),
    (
        "correctness_mcq",
        "correctness (MMLU-Pro)",
        "#2388 matched correct−incorrect t1 direction over MMLU-Pro spread contexts",
    ),
    (
        "correctness_code",
        "correctness (code)",
        "#2388 matched correct−incorrect t1 direction over code spread contexts",
    ),
]

PAPER_OFFSETS = {
    # "harmful compliance" and "AdvBench harmful requests" are long labels, so
    # the two rows below them are staggered to keep every label clear of its
    # neighbours and of the correctness leader lines.
    "harmful compliance": (-14, -14, "center", "top"),
    "sycophancy": (0, 12, "center", "bottom"),
    "hallucination": (-12, -2, "right", "center"),
    "refusal": (12, -2, "left", "center"),
    "assistant axis": (128, 18, "left", "bottom"),
    "casualness": (30, 16, "left", "bottom"),
    "impoliteness": (-30, -26, "right", "top"),
    "AdvBench harmful requests": (-10, -28, "right", "top"),
    "correctness (math)": (20, 10, "left", "bottom"),
    "correctness (MMLU-Pro)": (20, -10, "left", "top"),
    "correctness (code)": (16, -12, "left", "top"),
}


def draw_spectrum_panel(
    ax, res: dict, *, offsets: dict | None = None, legend_frame: bool = False
) -> None:
    """Draw the useful-directions spectrum panel (c2a-v2) onto ``ax``.

    Everything inside the axes: the random-direction band, the answer-PCA
    spectrum curve, the labeled direction points with leader lines, log x
    scale, axis labels, and the in-axes legend.  Panel furniture (kicker and
    title) stays with the caller, so the same panel serves the standalone
    spectrum figure and the combined layouts in :mod:`figures.section42`.
    ``offsets`` overrides :data:`PAPER_OFFSETS` per plot label for hosts whose
    panel geometry differs from the standalone wide canvas.  ``legend_frame``
    puts a white knockout behind the legend for hosts whose narrower axes let
    the curve descend through the legend area.
    """
    ranks1 = np.array(res["ranks_evaluated"]) + 1
    r2 = np.array(res["r2_by_rank"])
    rd = res["random_directions"]
    ax.axhspan(
        rd["r2_mean"] - rd["r2_sd"],
        rd["r2_mean"] + rd["r2_sd"],
        alpha=0.3,
        color=ROLES["control"].color,
        lw=0,
        label="random directions (mean ± SD)",
    )
    ax.plot(ranks1, r2, lw=1.8, color=ROLES["linear"].color, label="answer-PCA direction")
    ax.axhline(0.0, lw=0.8, color=MUTED)

    label_offsets = dict(PAPER_OFFSETS)
    label_offsets.update(offsets or {})
    # Headroom for the top label row (the labeled cluster sits at R2 0.84-0.91).
    ax.set_ylim(-0.32, 1.05)
    for name, lab, _ in ROSTER:
        e = res["directions"][name]
        x, y = e["plotted_rank_1based"], e["heldout_r2"]
        ax.scatter([x], [y], s=64, zorder=7, color=INK, marker="o", linewidths=0)
        dx, dy, ha, va = label_offsets.get(lab, (8, 6, "left", "bottom"))
        # A long offset (the crowded upper-right cluster) gets a thin leader line.
        arrow = (
            dict(arrowstyle="-", lw=0.8, color=MUTED, shrinkA=1, shrinkB=3)
            if abs(dx) + abs(dy) > 27
            else None
        )
        ax.annotate(
            lab,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=13,
            color=INK,
            ha=ha,
            va=va,
            zorder=6,
            arrowprops=arrow,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Variance rank (log)")
    ax.set_ylabel(better_label(METRIC_LABELS["r2"]))
    style_axis(ax)
    legend = ax.legend(loc="lower left")
    if legend_frame:
        legend.set_frame_on(True)
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.85)
        frame.set_edgecolor("none")
