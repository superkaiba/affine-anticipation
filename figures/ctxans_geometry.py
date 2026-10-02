#!/usr/bin/env python3
"""Paper figure ``c3_ctxans_geometry`` (appendix A.4, context-vs-answer geometry).

Renders the context-vs-answer geometry figure from the banked summary JSON (no
compute).  Panel A is model-free geometry, drawn in ink; panel B is the linear
map's retrieval, drawn in the paper's linear-predictor teal.

Ported from EPS ``scripts/issue1901_ctxans_geometry_figure.py`` at commit
e78f95c91a8c3cfc2f29010f98d56272dc8f8b89 (EPS issue 1901).  The summary JSON is
written by the upstream ``scripts/issue1901_ctxans_geometry.py`` analysis step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from context_answer.plotting.style import (
    GRID,
    INK,
    METRIC_LABELS,
    ROLES,
    better_label,
    c2a_figure,
    metric_style,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from context_answer.provenance import git_short_sha

STEM = "c3_ctxans_geometry"
SOURCE_REL = Path("issue_1901/ctxans_geometry/summary.json")


def mid(rows: list[dict]) -> np.ndarray:
    """Bin midpoints from each row's ``lo``/``hi`` edges."""
    return np.array([(r["lo"] + r["hi"]) / 2 for r in rows])


def main(records_dir: Path, out_dir: Path) -> list[Path]:
    """Render the figure from ``records_dir`` into ``out_dir``; return written paths."""
    source = records_dir / SOURCE_REL
    d = json.loads(source.read_text())
    A, B = d["panel_a"], d["panel_b"]
    teal = ROLES["linear"]
    m_top1 = metric_style("top1")

    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.40)
    axA = fig.add_axes([0.10, 0.23, 0.335, 0.55])
    axB = fig.add_axes([0.615, 0.23, 0.335, 0.55])

    xa = mid(A)
    axA.axhline(d["global"]["answer_cos_all_pairs_mean"], color=GRID, lw=1.0, zorder=1)
    axA.fill_between(
        xa, [r["p10"] for r in A], [r["p90"] for r in A], color=INK, alpha=0.13, lw=0, zorder=2
    )
    axA.plot(xa, [r["mean"] for r in A], color=INK, lw=1.6, marker="o", ms=3.2, zorder=3)
    style_axis(axA)
    axA.set_xlabel("Cosine between two context vectors")
    axA.set_ylabel("Answer-vector cosine")
    axA.set_xlim(-0.55, 1.05)
    axA.set_xticks([-0.5, 0.0, 0.5, 1.0])
    # Context and pair counts removed from the canvas: the caption and the surrounding
    # text carry them. The eyebrow keeps only the panel letter (panel_header renders a
    # dangling "A  ·  " for an empty kicker, so the letter goes in the kicker slot),
    # which leaves the row occupied and opens no whitespace band.
    panel_header(axA, "", "A", "Answer vs context similarity")

    xb = mid(B)
    top1 = np.array([r["top1"] for r in B])
    # Wilson centers are shifted from p, so a 100%-correct bin yields ci_hi < p;
    # clip at zero rather than let matplotlib reject a negative error bar.
    err = np.clip(
        np.vstack(
            [top1 - np.array([r["ci_lo"] for r in B]), np.array([r["ci_hi"] for r in B]) - top1]
        ),
        0.0,
        None,
    )
    axB.errorbar(
        xb,
        top1,
        yerr=err,
        color=teal.color,
        lw=1.6,
        marker=teal.marker,
        ms=3.6,
        linestyle=m_top1["linestyle"],
        markerfacecolor="none",
        markeredgewidth=1.1,
        elinewidth=0.9,
        capsize=1.8,
        zorder=3,
    )
    style_axis(axB)
    axB.set_xlabel("Cosine to the nearest other context")
    axB.set_ylabel(better_label(METRIC_LABELS["top1"]))
    axB.set_xlim(-0.55, 1.05)
    axB.set_xticks([-0.5, 0.0, 0.5, 1.0])
    axB.set_ylim(0, 1.02)
    # Held-out-query and candidate counts removed from the canvas: the caption states
    # "the 942 held-out queries" and the surrounding text the 10,000-candidate pool.
    panel_header(axB, "", "B", "Retrieval vs nearest-context similarity")

    out_dir.mkdir(parents=True, exist_ok=True)
    res = save_c2a_figure(
        fig,
        out_dir / STEM,
        title="Context-space vs answer-space geometry",
        subject="Affine Anticipation, appendix A.4",
        creator=f"figures/{Path(__file__).name}",
        include_width=frac,
    )
    meta = {
        "source": str(SOURCE_REL),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_commit": git_short_sha(),
        "plotted": {
            "panel_a": A,
            "panel_b": B,
            "global": d["global"],
            "convention": d["convention"],
            "answer_draws": d["answer_draws"],
        },
        "render": res["record"],
    }
    meta_path = out_dir / f"{STEM}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
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
