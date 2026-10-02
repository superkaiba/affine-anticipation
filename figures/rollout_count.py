#!/usr/bin/env python3
"""Paper figure ``c1_rollout_count`` -- evaluation-target rollout count.

Appendix section A.12. All 31 nonempty subsets of five on-policy answer vectors
were scored on the frozen Figure 2 bank, with maps, whitening, context rows, and
duplicate policy held fixed. K-wise estimates average the subset-specific
metrics; bootstrap intervals resample contexts, conditional on that five-draw
bank and the fixed retrieval candidate pool. This is not a training-K ablation.
The figure joins the persisted K=1-5 and K=10 results without rescoring
anything.

Ported from ``scripts/issue1901_k_rollout_ablation.py`` (commit ``210b5f4d2f3``)
of the ``explore-persona-space`` research repository, where this result was
produced under issue 1901. Only the figure path is ported, which is what the
manuscript render used (``--plot-through-k10 --paper-style``): the scoring path
imported three sibling scripts and read GPU-side activation banks, so the two
summaries it produced are committed under ``records/`` instead. The four names
the figure path borrows from those siblings are in
``figures/_rollout_count_support.py``.

Note: the submitted PDF carries a post-hoc heading band rewrite, applied by a
separate tool that replaced the two panel headings without touching any plotted
value ("A Variance explained" / "B Answer retrieval" in place of this script's
kicker-plus-title pairs). The ``heading_migration`` block of
``figures/reference/c1_rollout_count.meta.json`` records that substitution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from figures import _rollout_count_support as support

ARMS = ("ridge", "mlp", "identity_bias")
METRICS = ("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean")
SEED = support.BOOT_SEED
N_BOOT = support.BOOT_N

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = REPO_ROOT / "records"
DEFAULT_OUT = REPO_ROOT / "out"

RECORD_PATHS: dict[str, str] = {
    "k5_summary": "issue_1901/k_rollout_ablation/summary.json",
    "k10_summary": "issue_1901/k10_rollout_ablation/summary.json",
}
"""Result records this figure reads, relative to the ``--records`` directory.

The relative paths are the producing repository's ``eval_results/`` paths with
that prefix removed.
"""

OUT_STEM = "c1_rollout_count"


def make_figure(
    summary: dict,
    stem: Path,
    *,
    include_baseline: bool = False,
    curve: dict | None = None,
    source_metadata: dict | None = None,
    paper: bool = False,
) -> list[Path]:
    """Render both outcome curves using the manuscript's shared visual system."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from context_answer.plotting import style

    style.set_c2a_style()
    fig, frac = style.c2a_figure("full", aspect=0.50 if curve is not None and not paper else 0.44)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.28 if curve is not None and not paper else 0.20,
        top=0.78,
        wspace=0.28,
    )
    roles = {"ridge": "linear", "mlp": "nonlinear", "identity_bias": "control"}
    labels = {"ridge": "Linear map", "mlp": "Nonlinear map", "identity_bias": "Identity + bias"}
    if paper:
        labels.update(ridge="Linear metamodel", mlp="Nonlinear metamodel")
    cells = summary["all_subsets"]["per_k"] if curve is None else curve
    ks = sorted(map(int, cells))
    for arm in ARMS if include_baseline else ARMS[:2]:
        series = style.ROLES[roles[arm]]
        for j, metric in enumerate(("r2", "top1")):
            vals = [cells[str(k)]["arms"][arm] for k in ks]
            vals = [v["r2"] if j == 0 else v["retrieval"]["whiten_csls"] for v in vals]
            y = np.array([v["mean"] for v in vals])
            lo, hi = np.array([v["ci95"] for v in vals]).T
            axes[j].plot(
                ks,
                y,
                color=series.color,
                marker=series.marker,
                linestyle="-" if j == 0 else "--",
                markerfacecolor=series.color if j == 0 else "white",
                label=labels[arm],
                linewidth=2,
                markersize=7,
            )
            axes[j].fill_between(ks, lo, hi, color=series.color, alpha=0.12, linewidth=0)
    for ax, label in zip(axes, ("Held-out $R^2$", "Top-1 retrieval"), strict=True):
        style.style_axis(ax)
        ax.set(
            xlabel="Rollouts averaged, $K$",
            ylabel=style.better_label(label),
            xticks=ks,
            xlim=(ks[0] - 0.3, ks[-1] + 0.3),
        )
    if include_baseline:
        axes[1].set_ylim(0, 1.03)
        axes[1].axhline(summary["chance_top1"], color=style.MUTED, linewidth=1, linestyle=":")
    else:
        axes[0].set_ylim(0.73, 0.88)
        axes[1].set_ylim(0.945, 0.995)
    axes[0].set_title("Variance explained", loc="left")
    axes[1].set_title("Answer identification", loc="left")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False
    )
    if paper:
        style.panel_header(axes[0], "A", "Reconstruction", kicker_y=1.12)
        style.panel_header(axes[1], "B", "Retrieval", kicker_y=1.12)
    else:
        fig.text(
            0.5,
            0.035,
            "Frozen maps · 1,000 contexts · 942 candidates · Whitened cosine + CSLS · 95% intervals",
            ha="center",
            color=style.MUTED,
            fontsize=14,
        )
    subject = "All subsets of five fixed on-policy draws; evaluation K only"
    if curve is not None:
        subject = "K=1–5: subset-averaged metrics from five draws; K=10: all ten draws"
        if not paper:
            fig.text(0.5, 0.085, subject, ha="center", color=style.MUTED, fontsize=14)
    exported = style.save_c2a_figure(
        fig,
        stem,
        title="Effect of rollout averaging on mapping quality",
        subject=subject,
        creator=Path(__file__).name,
        include_width=frac,
    )
    meta_path = stem.with_suffix(".meta.json")
    support._write_json(
        meta_path,
        {
            **exported["record"],
            "plotted_arms": list(ARMS if include_baseline else ARMS[:2]),
            "data": summary["all_subsets"] if curve is None else {"per_k": curve},
            "plotted_k": ks,
            "sources": source_metadata,
            "paper_style": paper,
            "outputs_sha256": {
                k: support._sha256(exported[k]) for k in ("pdf", "png", "grayscale")
            },
        },
    )
    plt.close(fig)
    return [exported["pdf"], exported["png"], exported["grayscale"], meta_path]


def make_combined_figure(
    k5_path: Path, k10_path: Path, stem: Path, *, paper: bool = False
) -> list[Path]:
    """Join verified, persisted results without inference or metric recomputation."""
    old, new = [json.loads(path.read_text()) for path in (k5_path, k10_path)]
    for key in ("model", "layer", "n_train", "n_test", "whitening", "provenance", "chance_top1"):
        if old[key] != new[key]:
            raise ValueError(f"K=5/K=10 source mismatch: {key}")
    if old["duplicate_audit"]["realized_n_pool"] != new["n_candidates"]:
        raise ValueError("Retrieval pool sizes differ")
    for key in ("n", "seed"):
        if old["bootstrap"][key] != new["paired_bootstrap"][key]:
            raise ValueError(f"Bootstrap {key} differs")
    cells = old["all_subsets"]["per_k"]
    if sorted(map(int, cells)) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected complete original K=1–5 results")
    for arm in ARMS:
        a, b = cells["5"]["arms"][arm], new["targets"]["existing_K5"][arm]
        for metric in ("r2", *METRICS):
            x, y = (
                (a["r2"], b["r2"])
                if metric == "r2"
                else (a["retrieval"][metric], b["retrieval"][metric])
            )
            np.testing.assert_allclose(
                [x["mean"], *x["ci95"]], [y["mean"], *y["ci95"]], atol=1e-10, rtol=0
            )
    curve = {**cells, "10": {"n_subsets": 1, "arms": new["targets"]["K10"]}}
    sources = {
        "input_files": [
            {"path": _display_path(path), "sha256": support._sha256(path)}
            for path in (k5_path, k10_path)
        ],
        "producer_sha256": support._sha256(Path(__file__)),
        "shared_endpoint_parity": "PASS: all predictors/metrics, point estimates and intervals",
        "scope": "Frozen maps; K=1–5 uses subsets of five draws; K=10 uses all ten draws",
        "intervals": "Pointwise 95% context bootstrap, conditional on observed draws and fixed pool",
        "unmeasured_k": [6, 7, 8, 9],
        "chance_top1": new["chance_top1"],
    }
    return make_figure(old, stem, curve=curve, source_metadata=sources, paper=paper)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def main(records_dir: Path = DEFAULT_RECORDS, out_dir: Path = DEFAULT_OUT) -> list[Path]:
    """Render the rollout-count figure from the two persisted summaries."""

    sources = {name: records_dir / rel for name, rel in RECORD_PATHS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing result records: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = make_combined_figure(
        sources["k5_summary"], sources["k10_summary"], out_dir / OUT_STEM, paper=True
    )
    for path in written:
        print(f"{path.name}: {path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.records, args.out)
