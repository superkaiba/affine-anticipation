#!/usr/bin/env python3
"""Paper figure ``c3_ctxans_projection`` (appendix A.4, PCA of both spaces).

Three PCA panels: do context-space clusters survive into answer space?  The same
9,058 points appear in every panel.  A shows the clusters where they were fit.
B recolors answer space by the CONTEXT clusters (do they stay together?).
C recolors answer space by its OWN clusters, which is what stops the figure
being misread as "answer space is unstructured".

Ported from EPS ``scripts/issue1901_ctxans_projection_figure.py`` at commit
e78f95c91a8c3cfc2f29010f98d56272dc8f8b89 (EPS issue 1901).  The coordinates and
summary are written by the upstream ``scripts/issue1901_ctxans_projection.py``
analysis step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from context_answer.plotting.style import (
    INK,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)

STEM = "c3_ctxans_projection"
SOURCE_REL = Path("issue_1901/ctxans_projection")

# Okabe-Ito, colorblind-safe; black swapped for the paper's ink.
PALETTE = [INK, "#E69F00", "#56B4E9", "#009E73", "#D8C525", "#0072B2", "#D55E00", "#CC79A7"]


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Render the figure from ``records_dir`` into ``out_dir``; return written paths."""
    source = records_dir / SOURCE_REL
    z = np.load(source / "coords.npz")
    s = json.loads((source / "summary.json").read_text())
    pc_c, pc_a = z["pc_context"], z["pc_answer"]
    lab_c, lab_a = z["label_context"], z["label_answer"]
    vc, va = s["var_explained"]["context"], s["var_explained"]["answer_centered"]

    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.36)
    axes = [fig.add_axes([0.085 + i * 0.315, 0.16, 0.225, 0.58]) for i in range(3)]

    def scatter(ax, xy, lab):
        for g in range(int(lab.max()) + 1):
            m = lab == g
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=1.6,
                c=PALETTE[g % len(PALETTE)],
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )
        ax.set_aspect("equal")
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    scatter(axes[0], pc_c, lab_c)
    axes[0].set_xlabel(f"Context PC1-2 ({100 * sum(vc):.0f}%)")
    # Context count removed from the canvas: the caption states "the 9,058 distractor
    # contexts". The eyebrow keeps only the panel letter (panel_header renders a dangling
    # "A  ·  " for an empty kicker, so the letter goes in the kicker slot), which
    # leaves the row occupied and opens no whitespace band. Panels B and C keep their
    # kickers: they decode the marks rather than state figure-global metadata.
    panel_header(axes[0], "", "A", "Context space")

    scatter(axes[1], pc_a, lab_c)
    axes[1].set_xlabel(f"Answer PC1-2 ({100 * sum(va):.0f}%)")
    panel_header(axes[1], "B", "same colors as A", "In answer space")

    scatter(axes[2], pc_a, lab_a)
    axes[2].set_xlabel(f"Answer PC1-2 ({100 * sum(va):.0f}%)")
    panel_header(axes[2], "C", f"ARI {s['ari_centered']:.2f} vs A", "Answer clusters")

    out_dir.mkdir(parents=True, exist_ok=True)
    res = save_c2a_figure(
        fig,
        out_dir / STEM,
        title="Context and answer spaces, first two principal components",
        subject="Affine Anticipation, appendix A.4",
        creator=f"figures/{Path(__file__).name}",
        include_width=frac,
    )
    meta_path = out_dir / f"{STEM}.meta.json"
    meta_path.write_text(
        json.dumps(
            {"source": str(SOURCE_REL), "stats": s, "render": res["record"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    written = [res["pdf"], res["png"], res["grayscale"], meta_path]
    for path in written:
        print(f"  {path}")
    return written


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=Path("records"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    main(a.records, a.out)


if __name__ == "__main__":
    _cli()
