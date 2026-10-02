#!/usr/bin/env python3
"""Paper figure ``c4_turn_transfer_full`` (method details, matched-budget turn transfer).

Plots verified matched-budget turn-transfer curves using the manuscript style:
held-out R^2 against evaluation turn, one curve per map-training-turn condition,
for the instruction-tuned and the pretrained checkpoint, with pointwise
conditional 95% paired-conversation bootstrap bands.

Ported from EPS ``scripts/plot_issue825_turn_matched.py`` in the Codex working
tree of the research repository (EPS issue 825).  That
tree is where the submitted figure came from: its sidecar's
``latex_include_line`` names ``turn_matched_transfer_20260915.pdf``, its
``source_path`` is that tree's
``eval_results/issue_825/turn12_matched_20260915/results_with_turn12.json``, and
its title, subject and five-row ``encodings`` block are exactly what this
script's ``transfer`` branch emits.  The submission renamed the stem
``c4_turn_transfer_full``.

The upstream script renders three figures from one input.  Only the first,
``transfer``, is a paper figure; ``rotation_sensitivity`` (pooled-assignment
sensitivity) and ``copy_baselines`` (source-trained copy-plus-bias controls) are
kept because they are the same loop over the same data, and keep their upstream
names.  Pass ``--kind`` to render one.

Input: ``records/issue_825/turn12_matched_20260915/results_with_turn12.json``
(sha256 306b7e251a552a77aa3578a67467d14d72d809dd1431fa3e73053ac8a8bb0ddc, the
value the submitted sidecar records).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb

from context_answer.plotting.style import (
    INK,
    ROLES,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

SOURCE_REL = Path("issue_825/turn12_matched_20260915/results_with_turn12.json")
KINDS = ("transfer", "rotation_sensitivity", "copy_baselines")
# Upstream wrote every panel as turn_matched_<kind>_20260915; the submission ships
# only the transfer panel, under the paper stem.
STEMS = {
    "transfer": "c4_turn_transfer_full",
    "rotation_sensitivity": "turn_matched_rotation_sensitivity",
    "copy_baselines": "turn_matched_copy_baselines",
}
TITLES = {
    "transfer": "Transfer at a matched training budget",
    "rotation_sensitivity": "Sensitivity to pooled turn assignment",
    "copy_baselines": "Source-trained copy-plus-bias baselines",
}


def main(records_dir: Path, out_dir: Path, kinds: tuple[str, ...] = KINDS) -> list[Path]:
    """Render the requested panels from ``records_dir`` into ``out_dir``."""
    source = records_dir / SOURCE_REL
    result = json.loads(source.read_text())
    if result["n_conversations"] != 4975 or result["answer_draws"] != 1:
        raise ValueError("wrong matched-panel result")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_c2a_style()
    turns = np.arange(1, 13)
    control = np.array(to_rgb(ROLES["control"].color))
    encodings = [
        ("1", "Turn 1", to_hex(0.2 + 0.8 * control), "s"),
        ("2", "Turn 2", ROLES["control"].color, "^"),
        ("3", "Turn 3", to_hex(0.55 * control), "D"),
        ("1+2+3", "Turns 1 + 2 + 3", ROLES["linear"].color, "o"),
    ]
    if "12" in result["source_conditions"]:
        encodings.insert(3, ("12", "Turn 12", INK, "v"))
    written: list[Path] = []
    for kind in kinds:
        fig, fraction = c2a_figure("full", aspect=0.48)
        axes = fig.subplots(1, 2, sharey=True)
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.28, top=0.80, wspace=0.15)
        all_limits = []
        # Every value this panel draws, so a sidecar comparison can catch a moved
        # curve. The submitted sidecar records only the encodings and the input
        # checksum, so none of the plotted R^2 or bootstrap bands was checkable.
        plotted: dict = {
            "turns": turns.tolist(),
            "bands_drawn": kind != "rotation_sensitivity",
            "panels": [],
        }
        for ax, model, letter in zip(axes, ("instruct", "pretrained"), ("A", "B"), strict=True):
            cells = result["models"][model]["cells"]
            panel: dict = {"letter": letter, "model": model, "curves": []}
            plotted["panels"].append(panel)
            ax.axvspan(0.7, 3.5, color=ROLES["control"].color, alpha=0.07, zorder=-2)
            ax.axvline(3.5, color=ROLES["control"].color, alpha=0.6, linewidth=1, linestyle=":")
            for src, label, color, marker in encodings:
                method = "source_identity_bias" if kind == "copy_baselines" else "raw"
                rows = sorted(
                    (r for r in cells if r["source"] == src and r["method"] == method),
                    key=lambda r: r["target_turn"],
                )
                if [r["target_turn"] for r in rows] != turns.tolist():
                    raise ValueError("incomplete curve")
                values = np.array([r["r2"] for r in rows])
                ci = np.array([r["r2_ci95"] for r in rows])
                all_limits.extend(ci.ravel())
                panel["curves"].append(
                    {
                        "source": src,
                        "label": label,
                        "method": method,
                        "r2": values.tolist(),
                        "r2_ci95": ci.tolist(),
                    }
                )
                ax.plot(
                    turns,
                    values,
                    label=label,
                    color=color,
                    marker=marker,
                    markerfacecolor=color if src == "1+2+3" else "white",
                    markersize=6,
                    linewidth=2.4 if src == "1+2+3" else 1.8,
                    linestyle="--" if src == "12" else "-",
                )
                if kind != "rotation_sensitivity":
                    ax.fill_between(turns, ci[:, 0], ci[:, 1], color=color, alpha=0.13, linewidth=0)
            if kind == "rotation_sensitivity":
                rotations = np.array(result["models"][model]["rotation_r2"])
                panel["rotation_r2"] = rotations.tolist()
                for r in rotations:
                    ax.plot(
                        turns,
                        r,
                        color=ROLES["linear"].color,
                        alpha=0.45,
                        linewidth=1,
                        linestyle=":",
                    )
                all_limits.extend(rotations.ravel())
            ax.set_xlim(0.7, 12.3)
            ax.set_xticks([1, 2, 3, 6, 9, 12])
            ax.set_xlabel("Evaluation turn")
            panel_header(ax, letter, model.title(), kicker_y=1.06)
            style_axis(ax)
        lo, hi = min(all_limits), max(all_limits)
        axes[0].set_ylim(np.floor((lo - 0.02) * 10) / 10, min(1, np.ceil((hi + 0.02) * 10) / 10))
        plotted["y_limits"] = [float(v) for v in axes[0].get_ylim()]
        axes[0].set_ylabel(better_label(r"Held-out $R^2$"))
        legend_kicker(
            fig,
            0.085,
            0.165,
            "Map training turns" if kind != "copy_baselines" else "Copy-bias training turns",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.08, 0.07),
            ncol=len(encodings),
            frameon=False,
            columnspacing=1.25,
        )
        subject = (
            "Three counterbalanced pooled fits shown as dotted curves; solid curve "
            "averages performance."
            if kind == "rotation_sensitivity"
            else "Bands are pointwise conditional 95% paired-conversation bootstrap intervals."
        )
        title = TITLES[kind]
        stem = STEMS[kind]
        saved = save_c2a_figure(
            fig,
            out_dir / stem,
            title=title,
            subject=subject,
            creator=f"figures/{Path(__file__).name}",
            include_width=fraction,
        )
        plt.close(fig)
        meta = dict(
            title=title,
            subject=subject,
            source_path=str(SOURCE_REL),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            n_conversations=4975,
            answer_draws=1,
            render=saved["record"],
            plotted=plotted,
            encodings=[dict(source=s, label=la, color=c, marker=m) for s, la, c, m in encodings],
            caveat=(
                "Turns1–3 overlap pooled source support; forward transfer is turns4–12. "
                "Each pooled rotation is a distinct matched-N fit; average losses, not "
                "predictions."
            ),
        )
        meta_path = out_dir / f"{stem}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        written.extend([saved["pdf"], saved["png"], saved["grayscale"], meta_path])
        for key in ("pdf", "png", "grayscale"):
            print(f"  {saved[key]}")
        print(f"  {meta_path}")
    return written


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("records"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument(
        "--kind",
        choices=KINDS,
        action="append",
        help="Render only this panel; repeatable. Default: all three.",
    )
    args = parser.parse_args()
    main(args.records, args.out, tuple(args.kind) if args.kind else KINDS)


if __name__ == "__main__":
    _cli()
