#!/usr/bin/env python3
"""Paper figure ``c5_preimage_qualitative_top100`` (method details, preimage examples).

Renders verified excerpts from each persona preimage's top 100 contexts as a
three-row table: the behavior direction, the prompt categories found among its
top 100, and literal excerpts from individual ranked prompts.

Ported from EPS ``scripts/issue1739_qualitative_preimage_figure.py`` at commit
709cef84784c3fe5d1cbb19f5a0ac29b9e0e9fb2 on branch
``codex/1739-covariance-ablation-20260916`` (EPS issue 1739), the commit that
introduced both the script and the figure.  The script is on no other branch and
in no working tree, so it is extracted with ``git show``.

Inputs.  Upstream the script read one spec,
``eval_results/issue_1739/qualitative_top100_20260918/figure_inputs.json``, and
then re-derived every excerpt from the behavior-explorer dashboard the spec pins
(``dashboard/public/tasks/1739/behavior-explorer.html``).  That dashboard is 14.9
MB and embeds the prompt and response text of 952,067 real user conversations, so
it is deliberately NOT committed to this release.  The spec ships as
``records/issue_1739/qualitative_top100_20260918/figure_inputs.json`` and the
verification step's output -- the rendered excerpts, ranks, per-prompt hashes and
top-100 id lists -- ships beside it as ``verified_groups.json``, lifted from the
generating commit's own figure sidecar.  ``verified_examples`` and ``excerpt`` are
kept unchanged and still run: point ``--dashboard`` at a copy of the dashboard
HTML and the script re-derives the groups, asserts the pinned source sha256, and
checks the result equals the committed ``verified_groups.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt

from context_answer.plotting.style import (
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
)

STEM = "c5_preimage_qualitative_top100"
SPEC_REL = Path("issue_1739/qualitative_top100_20260918/figure_inputs.json")
GROUPS_REL = Path("issue_1739/qualitative_top100_20260918/verified_groups.json")


def digest(path):
    """Hex sha256 of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(text, parts):
    """Accept only literal, ordered source spans and mark every omission."""
    cursor = 0
    for part in parts:
        start = text.find(part, cursor)
        if start < 0:
            raise ValueError(f"Excerpt not present in source: {part!r}")
        cursor = start + len(part)
    value = " … ".join(parts)
    if text.find(parts[0]) > 0:
        value = "… " + value
    if cursor < len(text):
        value += " …"
    return "“" + value + "”"


def verified_examples(spec, dashboard: Path):
    """Re-derive every excerpt from the pinned dashboard snapshot."""
    if digest(dashboard) != spec["source_sha256"]:
        raise ValueError("The source snapshot changed")
    match = re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>', dashboard.read_text(), re.S)
    if match is None:
        raise ValueError("Dashboard data block missing")
    data = json.loads(match.group(1))
    records = {r["ci"]: r for r in data["records"]}
    if data["pool"]["n_unique_candidate_prompts"] != spec["selection"]["pool_size"]:
        raise ValueError("Candidate population mismatch")
    groups = []
    for group in spec["groups"]:
        behavior = group["behavior"]
        members = sorted(
            (r for r in records.values() if r["ranks"][behavior]["preimage_cosine"]["rank"] <= 100),
            key=lambda r: r["ranks"][behavior]["preimage_cosine"]["rank"],
        )
        if len(members) != 100:
            raise ValueError("Incomplete top-100 coverage")
        entries = []
        for example in group["examples"]:
            record = records[example["ci"]]
            rank = record["ranks"][behavior]["preimage_cosine"]["rank"]
            if not 1 <= rank <= 100:
                raise ValueError("Example outside requested top 100")
            rendered = dict(
                **example,
                rank=rank,
                prompt_excerpt=excerpt(record["prompt"], example["prompt_parts"]),
                prompt_sha256=hashlib.sha256(record["prompt"].encode()).hexdigest(),
                response_sha256=record["response_sha256"],
            )
            if "response_parts" in example:
                rendered["response_excerpt"] = excerpt(
                    record["response"], example["response_parts"]
                )
            entries.append(rendered)
        groups.append(
            dict(**group, rendered_examples=entries, top100_ids=[r["ci"] for r in members])
        )
    return groups


def main(records_dir: Path, out_dir: Path, dashboard: Path | None = None) -> list[Path]:
    """Render the figure from ``records_dir`` into ``out_dir``; return written paths."""
    spec_path = records_dir / SPEC_REL
    groups_path = records_dir / GROUPS_REL
    spec = json.loads(spec_path.read_text())
    banked = json.loads(groups_path.read_text())
    groups = banked["groups"]
    if dashboard is not None:
        rederived = verified_examples(spec, dashboard)
        if rederived != groups:
            raise ValueError("Re-derived groups differ from the committed verified groups")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.72)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    for x, label in [
        (0, "DIRECTION"),
        (0.20, "PROMPT CATEGORIES"),
        (0.45, "EXAMPLE EXCERPTS"),
    ]:
        ax.text(x, 0.985, label, va="top", fontweight=700, color=MUTED)
    row_tops = [0.925, 0.700, 0.335]
    separators = [0.95, 0.727, 0.362, 0.010]
    artists = []
    for group, y in zip(groups, row_tops, strict=True):
        ax.text(0, y, textwrap.fill(group["label"], 15), va="top", fontweight=700, color=INK)
        categories = "\n\n".join(textwrap.fill(s, 22) for s in group["categories"])
        ax.text(0.20, y, categories, va="top", color=ROLES["linear"].color, linespacing=1.15)
        blocks = []
        for item in group["rendered_examples"]:
            value = f"Rank {item['rank']}: {item['prompt_excerpt']}"
            block = textwrap.fill(value, 56)
            if "response_excerpt" in item:
                block += "\n" + textwrap.fill("Saved answer: " + item["response_excerpt"], 56)
            if "reference_note" in item:
                block += "\n" + textwrap.fill(item["reference_note"], 56)
            blocks.append(block)
        artists.append(ax.text(0.45, y, "\n\n".join(blocks), va="top", color=INK, linespacing=1.12))
    for y in separators:
        ax.plot([0, 1], [y, y], color=MUTED, linewidth=0.8, alpha=0.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, lower in zip(artists, separators[1:], strict=True):
        bounds = artist.get_window_extent(renderer).transformed(ax.transAxes.inverted())
        if bounds.y0 < lower + 0.005 or bounds.x1 > 1.005:
            raise ValueError(f"Example text overflows its row: {bounds}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / STEM
    saved = save_c2a_figure(
        fig,
        target,
        title="Prompt categories among the top 100 preimage contexts",
        subject=spec["interpretation"],
        creator=f"figures/{Path(__file__).name}",
        include_width=fraction,
    )
    meta = dict(
        input_sha256=digest(spec_path),
        verified_groups_sha256=digest(groups_path),
        script_sha256=digest(Path(__file__)),
        source_sha256=spec["source_sha256"],
        selection=spec["selection"],
        interpretation=spec["interpretation"],
        groups=groups,
        render=saved["record"],
        output_sha256={key: digest(saved[key]) for key in ["pdf", "png", "grayscale"]},
    )
    meta_path = target.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    plt.close(fig)
    written = [saved["pdf"], saved["png"], saved["grayscale"], meta_path]
    for path in written:
        print(f"  {path}")
    return written


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("records"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=None,
        help=(
            "Optional path to the behavior-explorer dashboard HTML the spec pins. When given, "
            "the excerpts are re-derived from it and checked against verified_groups.json."
        ),
    )
    args = parser.parse_args()
    main(args.records, args.out, args.dashboard)


if __name__ == "__main__":
    _cli()
