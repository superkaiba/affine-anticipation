#!/usr/bin/env python3
"""Paper figure ``c5_behavior_transfer_original_combined`` (section 4.6).

Condenses the original four-method results into the paper's three-panel style.
Reads the verified historical summary only; all 36 values and OOD standard errors
remain unchanged.  Exports a full-width vector figure and its review/provenance
files.

Ported from EPS ``scripts/issue1739_old_combined_figure.py`` at commit
6a4e20115ee4d4fd2d8f1e9e6dd1ba6ff9a3cbd4 (EPS issue 1739),
the commit that introduced both the script and the figure.  The script is on no other branch and
in no working tree, so it is extracted with ``git show``.

Input: ``records/issue_1739/old_four_method_figure_20260917/summary.json``
(sha256 ba6dda3b..., the value the script's own pin asserts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from context_answer.plotting.style import (
    INK,
    MUTED,
    PAPER,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from figures._behavior_arms import BEHAVIORS, HEADINGS, LABELS, METHODS, sha

SOURCE_REL = Path("issue_1739/old_four_method_figure_20260917/summary.json")
SOURCE_SHA = "ba6dda3bd3ca206e65725085532e40b277e0cbd1b4a2f1d0b2a72c0158cb2384"
STEM = "c5_behavior_transfer_original_combined"
CAPTION = (
    "**Mapped answers usually improve prediction from fixed answer directions.** "
    "(A) Sycophancy, (B) hallucination, and (C) harmful compliance across generic chat, "
    "in-distribution (ID), and out-of-distribution (OOD) evaluations. Bars show Spearman "
    "correlations. OOD bars average datasets equally; whiskers show one standard error "
    "across datasets. The original maps, whitening, and method-specific layers are "
    "retained. ID rows were included in map fitting and therefore do not measure "
    "map-held-out generalization. The context-direction baseline uses the matching "
    "cached extraction. Sample counts appear below each evaluation group."
)


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Render the existing estimates as three grouped-bar panels without refitting."""
    source = records_dir / SOURCE_REL
    if sha(source) != SOURCE_SHA:
        raise ValueError("Historical summary changed; verify it before updating the pin")
    data = json.loads(source.read_text())
    if len(data["cells"]) != 9 or any(len(c["arms"]) != 4 for c in data["cells"]):
        raise ValueError("Expected nine complete evaluation groups")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.38)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.36, wspace=0.15)
    faces = (MUTED, ROLES["linear"].color, PAPER, INK)
    edges = (INK, ROLES["linear"].color, MUTED, INK)
    offsets = (-0.285, -0.095, 0.095, 0.285)
    for panel, (behavior, heading) in enumerate(zip(BEHAVIORS, HEADINGS, strict=True)):
        ax = axes[panel]
        ticklabels = []
        for group, regime in enumerate(("Generic chat", "ID", "OOD")):
            matches = [
                c for c in data["cells"] if c["behavior"] == behavior and c["regime"] == regime
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicated cell: {behavior}/{regime}")
            cell = matches[0]
            label = "Generic\nchat" if regime == "Generic chat" else regime
            ticklabels.append(f"{label}\nn={cell['n']:,}")
            for index, method in enumerate(METHODS):
                estimate = cell["arms"][method]
                value, interval = estimate["rho"], estimate["interval"]
                visible = [value] if interval is None else [value, *interval]
                if not np.isfinite(visible).all() or min(visible) < -0.2 or max(visible) > 0.8:
                    raise ValueError(f"Nonfinite or clipped estimate: {estimate}")
                x = group + offsets[index]
                ax.bar(
                    x,
                    value,
                    width=0.16,
                    color=faces[index],
                    edgecolor=edges[index],
                    linewidth=1.3,
                    hatch="///" if index == 0 else None,
                    label=LABELS[index] if group == 0 else None,
                    zorder=2,
                )
                if interval is not None:
                    lo, hi = interval
                    if lo > hi:
                        raise ValueError(f"Reversed interval: {interval}")
                    ax.vlines(x, lo, hi, color=INK, linewidth=1.1, zorder=3)
                    ax.hlines([lo, hi], x - 0.035, x + 0.035, color=INK, linewidth=1.1, zorder=3)
        ax.set_xlim(-0.52, 2.52)
        ax.set_ylim(-0.2, 0.8)
        ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticks([0, 1, 2], ticklabels)
        style_axis(ax, grid_axis="none")
        ax.tick_params(axis="x", length=0, pad=8)
        ax.axhline(0, color=MUTED, linewidth=0.7, zorder=0)
        panel_header(ax, chr(65 + panel), heading, kicker_y=1.06)
        if panel:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel(better_label(r"Spearman $\rho$"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.01),
        ncol=2,
        labelspacing=0.25,
        handlelength=1.2,
        columnspacing=1.7,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = save_c2a_figure(
        fig,
        out_dir / STEM,
        title="Fixed behavior directions across evaluation regimes",
        subject="Original results, three behavior panels, four fixed projection methods",
        creator=f"figures/{Path(__file__).name}",
        include_width=fraction,
    )
    metadata = {
        **data,
        "summary_sha256": SOURCE_SHA,
        "renderer_sha256": sha(Path(__file__)),
        "render": result["record"],
        "caption": CAPTION,
        "layout": (
            "Three behavior panels; generic, ID and OOD groups; unchanged historical values"
        ),
    }
    for key in ("pdf", "png", "grayscale"):
        metadata[f"{key}_sha256"] = sha(result[key])
    meta_path = out_dir / f"{STEM}.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    caption_path = out_dir / f"{STEM}.caption.md"
    caption_path.write_text(CAPTION + "\n")
    plt.close(fig)
    written = [result["pdf"], result["png"], result["grayscale"], meta_path, caption_path]
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
