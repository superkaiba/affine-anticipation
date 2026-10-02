#!/usr/bin/env python3
"""Paper figure ``c4_shared_speakers`` (section 4.4, the three-panel speaker figure).

Panel A: held-out R^2 of each speaker setting's own map, for the base and the
instruction-tuned checkpoint, beside the instruction-tuned seven-setting shared
map, with the base map applied frozen to the instruction-tuned model as diamonds.
Panel B: transfer between speaker settings on the instruction-tuned checkpoint,
frozen versus target-trained, error bars giving the minimum and maximum across
five fold means, negative R^2 shown at one quarter scale.
Panel C: turn-transfer heatmaps (train turn x evaluation turn) for both
checkpoints.

Ported from the archived producer the submission shipped beside the figure,
``figures/paper/source/c4_shared_speakers.py`` in the paper's Overleaf project
(an archive of EPS ``scripts/issue2054_story_manuscript_figure.py``; EPS issue
2054).  That archived file's sha256 is exactly the ``script_sha256`` recorded in
the submitted figure's sidecar, so it is the producer of the shipped PDF.

Only the archived producer's RENDER half is ported.  Its reduction half
(``summarize`` / ``add_stage_transfer`` / ``add_character_transfers`` /
``add_joint_character_transfers``) consumed eight inputs, three of which --
``figures/paper/inputs/speaker_character_transfer/{qwen2.5-7b.json,
qwen2.5-7b-instruct.json,assistant_sources.json}`` -- were never committed to
EPS or to the Overleaf project, so that half cannot be rerun anywhere.  Its
exact output was archived instead, as the figure's own
``c4_shared_speakers.data.json`` sidecar, and the submission's shipped
``c4_shared_speakers.render.py`` wrapper reproduces the figure from precisely
that file.  This port is that wrapper's route: it reads the archived reduction
output from ``records/overleaf/c4_shared_speakers.data.json`` (byte-identical to
the ``data`` block of the submitted sidecar) and calls the ported ``render``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from context_answer.plotting.style import (  # noqa: E402
    INCLUDE_WIDTHS,
    INK,
    ROLES,
    better_label,
    c2a_figure,
    canvas_width_in,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

STEM = "c4_shared_speakers"
SOURCE_REL = Path("overleaf/c4_shared_speakers.data.json")

MODELS = [("qwen2.5-7b", "Base", "base_model"), ("qwen2.5-7b-instruct", "Instruct", "post_trained")]
NAMES = [
    "Assistant (chat)",
    "Assistant (plain)",
    "Helios",
    "Wren",
    "Dana",
    "Vex",
    "Assistant (story)",
]
METHODS = [
    ("frozen", "Frozen", "o"),
    ("own", "Target-trained", "s"),
]
PAIRS = [("story", "characters"), ("chat", "story"), ("plain", "story"), ("story", "chat")]
# Plain-text comparisons are archived but omitted from the manuscript. The
# shared map retains its original seven-setting training provenance.
DISPLAY_INDICES = (0, 2, 3, 4, 5, 6)
# k5 stage-transfer summary labels, keyed to NAMES positions. The plain-text
# setting (index 1) has no stage-transfer cell and is not displayed.
STAGE_LABELS = {"Chat": 0, "HELIOS": 2, "Wren": 3, "Dana": 4, "Vex": 5, "Assistant-story": 6}
DISPLAY_PAIRS = [
    ("character", "other_characters"),
    ("chat+character", "other_characters"),
    ("story", "characters"),
    ("chat", "story"),
    ("story", "chat"),
]
# Bar geometry in row units. The row pitch shrinks with the shorter canvas, so
# the bars and the within-row offset shrink with it and the gutter between rows
# stays readable.
BAR_HEIGHT = 0.20
BAR_OFFSET = 0.23
# Two-line speaker labels set at the default 1.2 line spacing leave only about
# 2 pt between the adjacent "Assistant (chat)" and "Assistant (plain text)"
# rows, which is what pinned the old panel height. Single line spacing frees
# roughly 0.1 in per label and lets the row pitch come down without a collision.
TICK_LINESPACING = 1.0
# Vertical budget in canvas inches. The legend column above the bar panels and
# the tick-and-label band below them are set by type size, so they stay fixed
# and a shorter figure comes out of the plotted body alone. PANEL_IN is bounded
# below by panel B, whose five two-line row labels need about 0.53 in of pitch.
PANEL_IN = 2.70
TOP_BAND_IN = 1.68
BOTTOM_BAND_IN = 0.70
# Room for the "Instruct" title between the two turn-transfer heatmaps.
HEAT_GAP_IN = 0.44
# Heatmap geometry as published, expressed against the bar-panel band: the top
# of the upper heatmap sits at 0.934 of the band height, each map at 0.410.
HEAT_TOP_FRAC = 0.93435
COLORBAR_FRAC = 0.86


def compress_negative(values):
    """Keep positive R2 linear and show negative values at one quarter scale."""
    values = np.asarray(values)
    return np.where(values < 0, values / 4, values)


def expand_negative(values):
    """Invert the continuous piecewise display transform for Matplotlib ticks."""
    values = np.asarray(values)
    return np.where(values < 0, values * 4, values)


def render(data: dict, output: Path) -> dict:
    """Place joint/separate bars, Instruct transfer bars and turn heatmaps in one row."""
    set_c2a_style()
    # Reserve a separate legend column above each bar panel, then spend the rest
    # of the canvas on the panels themselves.
    width_in = canvas_width_in(INCLUDE_WIDTHS["full"])
    height_in = TOP_BAND_IN + PANEL_IN + BOTTOM_BAND_IN
    fig, fraction = c2a_figure("full", aspect=height_in / width_in)
    bottom = BOTTOM_BAND_IN / height_in
    panel = PANEL_IN / height_in
    heat_height = (HEAT_TOP_FRAC * PANEL_IN - HEAT_GAP_IN) / 2 / height_in
    a = fig.add_axes([0.105, bottom, 0.215, panel])
    b = fig.add_axes([0.510, bottom, 0.200, panel])
    for ax, letter, title in [(a, "A", "Separate and shared"), (b, "B", "Transfer (Instruct)")]:
        style_axis(ax, grid_axis="none")
        panel_header(ax, letter, title, kicker_y=1.045)
    base_color, instruct_color = ROLES["base_model"].color, ROLES["post_trained"].color
    base = data["models"]["qwen2.5-7b"]
    instruct = data["models"]["qwen2.5-7b-instruct"]
    bars = [
        (base["own"], base_color, None, "Base: separate"),
        (instruct["own"], instruct_color, None, "Instruct: separate"),
        (instruct["shared"], "white", "////", "Instruct: shared"),
    ]
    for j, (values, color, hatch, label) in enumerate(bars):
        a.barh(
            np.arange(len(DISPLAY_INDICES)) + (j - 1) * BAR_OFFSET,
            [values[i] for i in DISPLAY_INDICES],
            height=BAR_HEIGHT,
            color=color,
            edgecolor=instruct_color if hatch else color,
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
    # Base map applied frozen to Instruct: one diamond per displayed setting,
    # drawn on the Instruct-own row so the gap to that bar reads as retention.
    frozen = instruct.get("base_frozen")
    if frozen is not None:
        rows = [(k, frozen[i]) for k, i in enumerate(DISPLAY_INDICES) if frozen[i] is not None]
        a.plot(
            [v for _, v in rows],
            [k for k, _ in rows],
            linestyle="none",
            marker="D",
            markersize=5,
            markerfacecolor=base_color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=4,
            label="Base → Instruct",
        )
    a.set_yticks(
        range(6),
        [
            "Assistant\n(chat)",
            "Helios",
            "Wren",
            "Dana",
            "Vex",
            "Assistant\n(story)",
        ],
    )
    for tick in a.get_yticklabels():
        tick.set_linespacing(TICK_LINESPACING)
    a.set_ylim(5.5, -0.5)
    a.set_xlim(0, 0.75)
    a.set_xticks([0, 0.3, 0.6])
    a.set_xlabel(better_label("Held-out $R^2$"))
    fit_handles, fit_labels = a.get_legend_handles_labels()
    # get_legend_handles_labels lists line artists before bars; keep the bars
    # first and the frozen-transfer diamond last.
    order = sorted(range(len(fit_labels)), key=lambda k: fit_labels[k] == "Base → Instruct")
    fit_handles = [fit_handles[k] for k in order]
    fit_labels = [fit_labels[k] for k in order]
    # Panel A uses hue for model identity. Panel B uses neutral fill/outline for
    # transfer method, avoiding a second meaning for the same teal bars.
    transfer_color = INK
    method_style = {
        "frozen": (transfer_color, None),
        "own": ("white", None),
    }
    expected_pairs = [
        ("character", "other_characters"),
        ("chat+character", "other_characters"),
        PAIRS[0],
        ("chat", "plain"),
        *PAIRS[1:],
    ]
    if [(r["source"], r["target"]) for r in instruct["transfers"]] != expected_pairs:
        raise ValueError("Unexpected transfer-panel rows")
    displayed_transfers = [
        r for r in instruct["transfers"] if (r["source"], r["target"]) in DISPLAY_PAIRS
    ]
    assert [(r["source"], r["target"]) for r in displayed_transfers] == DISPLAY_PAIRS
    for j, pair in enumerate(displayed_transfers):
        for k, (method, _, _) in enumerate(METHODS):
            stat = pair["metrics"][method]
            mean, folds = stat["mean"], stat["folds"]
            color, hatch = method_style[method]
            b.barh(
                j + (k - 0.5) * BAR_OFFSET,
                mean,
                height=BAR_HEIGHT,
                color=color,
                edgecolor=transfer_color,
                linewidth=0.8,
                hatch=hatch,
                xerr=[[max(0, mean - min(folds))], [max(0, max(folds) - mean)]],
                error_kw={"ecolor": INK, "capsize": 2, "elinewidth": 0.8, "capthick": 0.8},
            )
    b.set_yticks(
        range(5),
        [
            "One character\n→ other characters",
            "Chat + one character\n→ other characters",
            "Assistant (story)\n→ characters",
            "Assistant (chat)\n→ assistant (story)",
            "Assistant (story)\n→ assistant (chat)",
        ],
    )
    for tick in b.get_yticklabels():
        tick.set_linespacing(TICK_LINESPACING)
    b.set_ylim(4.5, -0.5)
    endpoints = [
        v
        for pair in displayed_transfers
        for method, _, _ in METHODS
        for v in pair["metrics"][method]["folds"]
    ]
    assert min(endpoints) >= -1.0 and max(endpoints) <= 0.75
    b.set_xscale("function", functions=(compress_negative, expand_negative))
    b.set_xlim(-1.0, 0.75)
    b.set_xticks([-0.8, 0, 0.3, 0.6])
    b.axvline(0, color=ROLES["control"].color, linewidth=0.8, zorder=0)
    b.set_xlabel(better_label("Held-out $R^2$"))
    method_handles = [
        Patch(
            facecolor=method_style[method][0],
            edgecolor=transfer_color,
            hatch=method_style[method][1],
            label=label,
            linewidth=0.8,
        )
        for method, label, _ in METHODS
    ]
    # Each legend stays above its own panel, including the longest label.
    for x0, heading, handles, labels, ncol in [
        (0.105, "Model and fit", fit_handles, fit_labels, 2),
        (0.510, "Transfer method", method_handles, [m[1] for m in METHODS], 1),
    ]:
        legend_kicker(fig, x0, 0.978, heading)
        fig.legend(
            handles=handles,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(x0 - 0.001, 0.945),
            ncol=ncol,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.4,
            columnspacing=0.9,
            borderaxespad=0.0,
            labelspacing=0.3,
        )
    conditions = ["1", "2", "3", "12", "1+2+3"]
    matrices = {}
    for i, (key, label, role) in enumerate(
        [("pretrained", "Base", "base_model"), ("instruct", "Instruct", "post_trained")]
    ):
        # Both maps keep their published share of the panel band, so the shorter
        # canvas shrinks them in step with panels A and B.
        top = bottom + HEAT_TOP_FRAC * panel
        ax = fig.add_axes(
            [0.790, top - (i + 1) * heat_height - i * HEAT_GAP_IN / height_in, 0.140, heat_height]
        )
        if i == 0:
            # Align the C kicker with the A and B kickers at 1.045 of panel height.
            panel_header(ax, "C", "Turn transfer", kicker_y=1 + 0.11065 * panel / heat_height)
        cells = data["turns"]["results"]["models"][key]["cells"]
        matrix = []
        for source in conditions:
            selected = sorted(
                [v for v in cells if v["source"] == source and v["method"] == "raw"],
                key=lambda v: v["target_turn"],
            )
            assert [v["target_turn"] for v in selected] == list(range(1, 13))
            matrix.append([v["r2"] for v in selected])
        matrices[key] = matrix
        im = ax.imshow(
            matrix, aspect="auto", vmin=0, vmax=0.6, cmap="cividis", interpolation="nearest"
        )
        ax.set_title(label, loc="left", color=ROLES[role].color, pad=6)
        ax.set_yticks(range(5), ["1", "2", "3", "12", "1–3"])  # noqa: RUF001 - numeric range
        ax.set_ylabel("Train turn")
        ax.set_xticks([0, 5, 11], ["1", "6", "12"])
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Evaluation turn")
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([0.950, bottom, 0.010, COLORBAR_FRAC * panel])
    cb = fig.colorbar(im, cax=color_ax, ticks=[0, 0.3, 0.6])
    cb.ax.set_title(better_label("$R^2$"), pad=12)
    rendered = save_c2a_figure(
        fig,
        output / STEM,
        include_width=fraction,
        title="Maps across speakers, framings and turns",
        subject="K5 seven-setting speaker maps and matched-count K1 turn transfer",
        creator=f"figures/{Path(__file__).name}",
    )
    plt.close(fig)
    return {"render": rendered["record"], "turn_matrices": matrices, "outputs": rendered}


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Render the figure from the archived plotting inputs; return written paths."""
    source = records_dir / SOURCE_REL
    data = json.loads(source.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    result = render(data, out_dir)
    rendered = result.pop("outputs")
    result["data"] = data
    result["sources"] = {
        "archived_plotting_inputs": {
            "path": str(SOURCE_REL),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    }
    result["visual_encodings"] = {
        "panel_a": (
            "Base own (amber), Instruct own (teal), Instruct seven-setting shared "
            "(hatched teal), base map frozen onto Instruct (amber diamonds)"
        ),
        "panel_b": "Instruct only: frozen (ink fill), target-trained (outline)",
        "panel_b_error_bars": "Minimum and maximum across five fold means",
        "panel_b_x_scale": "Negative held-out R^2 shown at one quarter scale",
        "gridlines": "Removed; panel B retains a zero reference",
        "character_aggregation": "Equal average of four characters within each fold",
        "turn_heatmaps": {"colormap": "cividis", "limits": [0, 0.6]},
    }
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["style_sha256"] = hashlib.sha256(
        Path(save_c2a_figure.__code__.co_filename).read_bytes()
    ).hexdigest()
    result["output_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.glob(f"{STEM}*"))
        if path.suffix in {".pdf", ".png"}
    }
    meta_path = out_dir / f"{STEM}.meta.json"
    meta_path.write_text(json.dumps(result, indent=2) + "\n")
    written = [rendered["pdf"], rendered["png"], rendered["grayscale"], meta_path]
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
