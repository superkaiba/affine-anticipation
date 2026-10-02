#!/usr/bin/env python3
"""Paper figure ``c1_k5_length_control`` (method details, answer-length control).

Panel A: held-out pooled R^2 of each post-training stage's own map, over all
contexts versus the primary length-matched design, with row-bootstrap 95%
intervals.
Panel B: mean per-context total and residual sum of squares within five quantile
bins of the stage's own K=5 mean answer length, from the full-data fits.

Ported from the figure step of EPS ``scripts/issue1902_k5_length_control.py``
(``run_figure``), the copy in the ``wt-1902-k5`` working tree at
``/mnt/eps-data/thomasjiralerspong/wt-1902-k5``.  Version choice: that copy is
the only one of the five on disk whose panel A kicker is the post-migration
descriptive "own map" form the submitted figure carries; the other four still
render "OWN MAP · LAYER 31".

Only the figure step is ported.  The upstream script's other stages
(``lengths`` / ``match`` / ``parity`` / ``fit`` / ``summarize``) are the compute
pipeline that produced the summary JSON: they need the issue-1902 activation
stores, the rollout banks and GPU ridge fits, none of which ship with this
release.  Their committed output, ``eval_results/issue_1902/k5_targets/
length_matched/summary.json``, is the figure step's only input and is copied to
``records/issue_1902/k5_targets/length_matched/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from context_answer.plotting.style import (
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from context_answer.provenance import git_short_sha
from figures._k5_stage_style import (
    STAGE_COLORS,
    STAGE_MARKERS,
    STAGE_NAMES,
    STAGES,
    write_json,
)

FIG_STEM = "c1_k5_length_control"
SOURCE_REL = Path("issue_1902/k5_targets/length_matched/summary.json")


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Render the figure from ``records_dir`` into ``out_dir``; return written paths."""
    import matplotlib.pyplot as plt

    source = records_dir / SOURCE_REL
    summary = json.loads(source.read_text())
    prim = summary["designs"]["primary"]
    unm = summary["unmatched_reference"]
    strat = summary["length_stratified"]["per_stage"]

    set_c2a_style()
    fig, include_frac = c2a_figure("full", aspect=0.42)
    ax_a = fig.add_axes([0.075, 0.14, 0.36, 0.68])
    ax_b = fig.add_axes([0.56, 0.14, 0.41, 0.68])

    xs = np.arange(len(STAGES))
    series = (
        ("All contexts", unm, "#176B87", "o", -0.13),
        ("Length-matched", {s: prim["stages"][s] for s in STAGES}, "#C4553D", "D", 0.13),
    )
    for label, blocks, color, marker, dx in series:
        vals = [blocks[s]["r2"] for s in STAGES]
        cis = np.asarray([blocks[s]["r2_row_ci95"] for s in STAGES])
        err = np.abs(cis.T - np.asarray(vals))
        ax_a.errorbar(
            xs + dx,
            vals,
            yerr=err,
            fmt=marker,
            color=color,
            markersize=5,
            capsize=2.2,
            linewidth=1.1,
            linestyle="none",
            label=label,
        )
    ax_a.set_xticks(xs, [STAGE_NAMES[s] for s in STAGES])
    ax_a.set_ylabel(better_label("Held-out $R^2$"))
    style_axis(ax_a)
    # The read layer used to sit in this kicker. The manuscript caption and the
    # surrounding appendix text carry it now ("Context and answer vectors are
    # taken at layer 31"), so the canvas keeps only the descriptive data-slice
    # label, matching panel B. No row is freed, so the layout is unchanged.
    panel_header(ax_a, "A", "own map", "Answer-length control")
    ax_a.legend(frameon=False, loc="lower left", handletextpad=0.4)

    for s in STAGES:
        med = [b["len_median"] for b in strat[s]]
        ax_b.plot(
            med,
            [b["mean_ss_tot"] for b in strat[s]],
            color=STAGE_COLORS[s],
            marker=STAGE_MARKERS[s],
            markersize=4,
            linewidth=1.4,
            label=STAGE_NAMES[s],
        )
        ax_b.plot(
            med,
            [b["mean_ss_res"] for b in strat[s]],
            color=STAGE_COLORS[s],
            marker=STAGE_MARKERS[s],
            markersize=4,
            linewidth=1.4,
            linestyle="--",
            markerfacecolor="none",
        )
    ax_b.set_xscale("log")
    ax_b.set_xlim(22, 820)
    ax_b.set_xticks([30, 100, 300, 700], ["30", "100", "300", "700"])
    ax_b.xaxis.set_minor_formatter(NullFormatter())
    ax_b.set_xlabel("Answer length (tokens, bin median)")
    ax_b.set_ylabel("Mean SS per context")
    style_axis(ax_b)
    panel_header(ax_b, "B", "FULL DATA · 5 LENGTH BINS", "Variance vs residual by length")
    handles, labels = ax_b.get_legend_handles_labels()

    handles += [
        Line2D([], [], color="#687078", linewidth=1.4, label="Total SS"),
        Line2D([], [], color="#687078", linewidth=1.4, linestyle="--", label="Residual SS"),
    ]
    ax_b.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.4, ncols=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / FIG_STEM
    outputs = save_c2a_figure(
        fig,
        stem,
        title="K=5 diagonal R^2 under an answer-length control",
        subject="Length-matched own-map R^2 plus a length-stratified SS decomposition",
        creator=f"figures/{Path(__file__).name}",
        include_width=include_frac,
    )
    plt.close(fig)
    sidecar = {
        "metadata": {
            "script": f"figures/{Path(__file__).name}",
            "git_commit": git_short_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_summary": str(SOURCE_REL),
            "panel_a": (
                "held-out pooled R^2 per stage, all contexts vs the primary "
                f"length-matched design (band {prim['band']['lo']:.1f}-"
                f"{prim['band']['hi']:.1f} tokens, n={prim['matched_n']}/stage), "
                "row-bootstrap 95% CIs"
            ),
            "panel_b": (
                "mean per-context SS_tot (solid) and SS_res (dashed) per stage "
                "within 5 quantile bins of the stage's own K=5 mean answer length, "
                "from the full-data fits"
            ),
        },
        "panel_a": {
            "stages": [STAGE_NAMES[s] for s in STAGES],
            "unmatched_r2": [unm[s]["r2"] for s in STAGES],
            "unmatched_ci": [unm[s]["r2_row_ci95"] for s in STAGES],
            "matched_r2": [prim["stages"][s]["r2"] for s in STAGES],
            "matched_ci": [prim["stages"][s]["r2_row_ci95"] for s in STAGES],
        },
        "panel_b": strat,
        "render": outputs["record"],
    }
    data_path = stem.parent / f"{FIG_STEM}_data.json"
    write_json(data_path, sidecar)
    written = [outputs["pdf"], outputs["png"], outputs["grayscale"], data_path]
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
