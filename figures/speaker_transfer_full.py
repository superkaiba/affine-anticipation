#!/usr/bin/env python3
"""Paper figure ``c4_speaker_transfer_full`` (method details, speaker transfer).

Renders the chat/story appendix comparisons from the archived transfer summary:
four panels (base and instruction-tuned checkpoints x transfer direction), each
row a chat or character setting, each marker one of four map-fitting methods,
with x error bars giving the minimum and maximum across five held-out folds.
No fits are performed; everything is read from the archived summary JSON.

Ported from the archived producer the submission shipped alongside the figure,
``figures/paper/source/c4_speaker_transfer_full.py`` in the paper's Overleaf
project (itself an archive of EPS ``scripts/issue2054_story_transfer_appendix.py``;
EPS issue 2054).  The input is that figure's own
``c4_speaker_transfer_full.data.json`` sidecar, copied to
``records/overleaf/c4_speaker_transfer_full.data.json``.
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
from matplotlib.lines import Line2D  # noqa: E402

from context_answer.plotting.style import (  # noqa: E402
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

STEM = "c4_speaker_transfer_full"
SOURCE_REL = Path("overleaf/c4_speaker_transfer_full.data.json")

STORY = "conversation_paired_stories_assistant__on_policy__attrib_quoted"
SETTINGS = [("Chat", "conversation_paired_stories_assistant__on_policy__chat")] + [
    (name.title(), f"char_{name}__on_policy__attrib_quoted")
    for name in ("helios", "wren", "dana", "vex")
]

METHODS = (
    ("frozen", "Frozen", ROLES["linear"].color, "o"),
    ("bias", "+ Bias", ROLES["nonlinear"].color, "D"),
    ("bias_scale", "+ Bias + scale", ROLES["other_source"].color, "^"),
    ("own", "Target's own map", ROLES["control"].color, "s"),
)


def transfer_figure(summary, out):
    """Render the archived chat/story transfer comparisons and fold ranges.

    Returns the ``save_c2a_figure`` result plus a ``plotted`` record of every
    value the figure draws: one entry per panel, row and method, holding the
    held-out R^2 at the marker and the fold range the error bar spans. The
    submitted sidecar records only which settings were displayed, so without
    this block nothing the figure actually shows is checkable.
    """
    fig, fraction = c2a_figure("full", 0.85)
    axes = fig.subplots(2, 2)
    plotted: dict = {"methods": [key for key, *_ in METHODS], "panels": []}
    # Display names only; the data keys in the summary keep their own spelling.
    labels = ["Assistant, chat", "Helios", "Wren", "Dana", "Vex"]
    endpoints = [
        endpoint
        for result in summary.values()
        for pair in result["transfers"]
        if "__bare_text" not in pair["source"] and "__bare_text" not in pair["target"]
        for method in pair["methods"].values()
        for endpoint in method["fold_range"]
    ]
    low, high = min(0, min(endpoints)), max(0, max(endpoints))
    padding = 0.05 * (high - low)
    for row, (model, result) in enumerate(summary.items()):
        checkpoint = "Instruction-tuned" if model.endswith("-instruct") else "Base"
        for col, direction in enumerate(("out", "in")):
            ax = axes[row, col]
            panel: dict = {
                "letter": "ABCD"[row * 2 + col],
                "checkpoint": checkpoint,
                "direction": direction,
                "rows": [],
            }
            plotted["panels"].append(panel)
            for j, (_, prefix) in enumerate(SETTINGS):
                source, target = (STORY, prefix) if direction == "out" else (prefix, STORY)
                pair = next(
                    r
                    for r in result["transfers"]
                    if r["source"] == source and r["target"] == target
                )
                drawn: dict = {"label": labels[j], "from": source, "to": target, "methods": {}}
                panel["rows"].append(drawn)
                for offset, (key, _, color, marker) in zip(np.linspace(-0.24, 0.24, 4), METHODS):
                    value = pair["methods"][key]
                    lo, hi = value["fold_range"]
                    drawn["methods"][key] = {
                        "r2": float(value["r2"]),
                        "fold_range": [float(lo), float(hi)],
                    }
                    ax.errorbar(
                        value["r2"],
                        j + offset,
                        xerr=[[value["r2"] - lo], [hi - value["r2"]]],
                        fmt=marker,
                        color=color,
                        markersize=6,
                        capsize=2,
                        markerfacecolor="white" if key == "own" else color,
                    )
            ax.set_yticks(range(5), labels)
            ax.invert_yaxis()
            ax.axvline(0, color=ROLES["control"].color, linewidth=0.8)
            ax.grid(False, axis="y")
            ax.grid(True, axis="x", alpha=0.2)
            style_axis(ax, grid_axis="x")
            ax.set_xlim(low - padding, high + padding)
            ax.set_xticks([-0.5, 0.0, 0.5])
            ax.set_xlabel(better_label("Held-out $R^2$"))
            title = "Story assistant → target" if direction == "out" else "Source → story assistant"
            panel_header(ax, "ABCD"[row * 2 + col], checkpoint, title, kicker_y=1.14)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            marker=m,
            linestyle="none",
            label=label,
            markerfacecolor="white" if key == "own" else c,
        )
        for key, label, c, m in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.11, top=0.90, hspace=0.60, wspace=0.70)
    plotted["x_limits"] = [float(low - padding), float(high + padding)]
    plotted["x_ticks"] = [-0.5, 0.0, 0.5]
    rendered = save_c2a_figure(
        fig,
        out / STEM,
        include_width=fraction,
        title="Assistant-in-story map transfer",
        subject="Held-out map transfer; five-fold range",
        creator=f"figures/{Path(__file__).name}",
    )
    plt.close(fig)
    return rendered, plotted


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Reproduce the appendix figure without new fits or evaluations."""
    source = records_dir / SOURCE_REL
    data = json.loads(source.read_text())
    expected = {"qwen2.5-7b", "qwen2.5-7b-instruct"}
    if set(data["models"]) != expected:
        raise ValueError("Unexpected model coverage")
    for result in data["models"].values():
        if len(result["transfers"]) != 12:
            raise ValueError("Expected twelve archived source-target pairs per model")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_c2a_style()
    rendered, plotted = transfer_figure(data["models"], out_dir)
    meta = {
        "render": rendered["record"],
        "plotted": plotted,
        "plotting_script": f"figures/{Path(__file__).name}",
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            Path(save_c2a_figure.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "source_results_sha256": data["source_sha256"],
        "displayed_settings": [name for name, _ in SETTINGS],
        "omitted_comparison": "Plain-text assistant; original data preserved",
        "transfer_errorbars": "Minimum and maximum across five held-out folds",
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out_dir.glob(f"{STEM}*"))
            if path.suffix in {".pdf", ".png"}
        },
    }
    meta_path = out_dir / f"{STEM}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
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
