"""Regenerate every registered paper figure and check the numbers it draws.

Four checks run per paper figure stem:

* ``test_figure_renders`` -- the generator writes a non-empty PDF for the stem,
  and a submitted reference PDF exists in ``figures/reference/``.
* ``test_data_matches_reference`` -- every block of the submitted sidecar that
  is not provenance is reproduced leaf for leaf by the fresh render (numbers to
  1e-9).
* ``test_data_matches_baseline`` -- the same comparison against
  ``figures/baseline/``.
* ``test_numeric_coverage`` -- the two comparisons above together compare at
  least one number for every stem outside :data:`RENDER_ONLY_STEMS`. This is the
  guard that keeps a stem from silently dropping to zero numeric coverage while
  its two data tests still report a pass.

``figures/reference/`` and ``figures/baseline/`` hold different kinds of
artifact and are not interchangeable:

* A **reference** is the sidecar of the figure PDF as submitted. It was
  rendered on another machine from the research repository, and it is the
  strongest check available, because it ties this repository's output to the
  paper itself.
* A **baseline** is this repository's own first render, committed as a
  regression guard. It is used where the submitted sidecar cannot serve: nine
  stems shipped without a sidecar, and two shipped one that records only
  settings and prose, no plotted values. A baseline says nothing about the
  paper. It fails when a change to a generator, to a record, or to the shared
  fitting and plotting code moves a value that the figure draws.

Both comparisons select blocks by exclusion: everything a sidecar carries is
compared except keys that are provenance (checksums, git state, timestamps,
output file names, the style and render record, PDF title metadata) and leaves
that hold file paths, which differ because the reference was rendered from a
different checkout. Selecting by exclusion rather than by an allowlist of known
data blocks is deliberate: an allowlist silently drops every block a new
generator invents, which is how ``layer``, ``scaling``, ``extension`` and
``plotted`` went unchecked.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from figures.registry import GENERATORS, Generator

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDS = REPO_ROOT / "records"
REFERENCE = REPO_ROOT / "figures" / "reference"
BASELINE = REPO_ROOT / "figures" / "baseline"

# The sidecar files a generator may write for one stem, in comparison order.
SIDECAR_SUFFIXES = (".meta.json", "_data.json")

# Top-level sidecar keys that record how and where a figure was produced rather
# than what it draws. Matched exactly, plus any key ending in ``_sha256``. Every
# other top-level key is compared.
#
# These are deliberately NOT matched at depth. A nested key can reuse one of
# these names for something else entirely: ``c4_shared_speakers`` keeps its
# per-speaker fits under ``data/<group>/sources``, which has nothing to do with
# input files and holds several thousand numbers.
PROVENANCE_KEYS = frozenset(
    {
        # checksums and input manifests
        "output_sha256",
        "outputs_sha256",
        "sources",
        "source",
        "inputs",
        "input",
        "provenance",
        "outputs",
        # repository and environment state
        "git",
        "git_commit",
        "git_head_at_render",
        "source_status_at_render",
        "status",
        "font",
        "resolved_font",
        # the render record: save_c2a_figure's own return value, either nested
        # under one of these keys or flattened into the top level of a sidecar
        "render",
        "record",
        "save_record",
        "rendered_asset",
        "legend_asset",
        "style_version",
        "style_module",
        "plotting_script",
        "script",
        "paper_style",
        "authoring_size_inches",
        "exported_size_inches",
        "include_width_frac",
        "latex_include_line",
        "realized_font_pt",
        "png_dpi",
        "background",
        "scale",
        "text",
        # how to re-run, and migration notes carried over from the research repo
        "reproduction_command",
        "heading_migration",
        "inherited_scientific_metadata_status",
        # PDF document metadata
        "title",
        "subject",
    }
)

# Names denied wherever they appear on the path to a leaf inside a compared
# block. Kept as small as possible: these are the values that necessarily differ
# between two renders of the same data, so comparing them would fail for reasons
# that have nothing to do with the figure. ``sources``, ``input`` and the other
# ambiguous names from the list above are deliberately absent -- at depth they
# mean something else.
PROVENANCE_LEAF_NAMES = frozenset(
    {
        "sha256",
        "generated_utc",
        "timestamp_utc",
        "git_commit",
        "git_head_at_render",
        "source_status_at_render",
        "resolved_font",
        "font",
    }
)

# Key names that hold a filesystem location. Matched exactly, never as a
# substring: "dir" as a substring also matches "direction", which is how 21
# numeric leaves of c3_features_and_shifts went uncompared. A value-shaped check
# catches the paths these names miss.
PATH_LEAF_NAMES = frozenset(
    {"path", "file", "dir", "root", "source_path", "input_files", "paths", "files"}
)
PATH_VALUE_SUFFIXES = (".json", ".jsonl", ".npz", ".py", ".pdf", ".png", ".csv")

# The one stem that draws no data: the standalone legend for Figure 2, which is
# swatches and labels built from the style module. Its generator writes no
# sidecar, so there is nothing to compare and the render check is the whole
# check. Every other stem must compare at least one number
# (``test_numeric_coverage``).
RENDER_ONLY_STEMS = frozenset({"c1_controls_and_scaling_legend"})

# (stem, block) pairs left out of the reference comparison, with the reason.
# Narrow by design: each entry is a place where the ported generator and the
# submitted figure's own producer disagree about a block's SHAPE, which no
# change to this repository can reconcile. Numbers are never exempted.
REFERENCE_BLOCK_EXEMPTIONS: dict[tuple[str, str], str] = {
    ("c4_shared_speakers", "visual_encodings"): (
        "the ported generator writes a shorter encoding description than the submitted "
        "producer did (fewer keys, reworded prose, panel_b_x_scale in place of "
        "panel_b_axis). The values the figure draws are in 'data' and 'turn_matrices', "
        "both compared in full"
    ),
}


def _segments(key: str) -> list[str]:
    """Path segments of a flattened leaf key, with list indices stripped."""
    return [part.split("[", 1)[0] for part in key.split("/") if part]


def _is_provenance(key: str) -> bool:
    """True when a top-level sidecar key names provenance rather than drawn values."""
    return key in PROVENANCE_KEYS or key.endswith("_sha256")


def _is_ignored_name(name: str) -> bool:
    """True when a key on the path to a leaf makes that leaf uncomparable."""
    return (
        name.lower() in PATH_LEAF_NAMES
        or name in PROVENANCE_LEAF_NAMES
        or name.endswith("_sha256")
    )


def _is_ignored(key: str, value) -> bool:
    """True when a leaf of a compared block holds a path, a hash, or a timestamp.

    The test applies this to a leaf's full flattened key and
    :func:`tests.refresh_baselines.baseline_body` applies it while descending, so
    testing every segment rather than only the last keeps the two in step: a
    whole subtree dropped by one is dropped by the other. ``underlying_sources_sha256``
    is the case that needs it -- it maps file paths to hashes, so the hashes sit
    at leaves whose own last segment is a file name.
    """
    if any(_is_ignored_name(part) for part in _segments(key)):
        return True
    return isinstance(value, str) and "/" in value and value.endswith(PATH_VALUE_SUFFIXES)


def _data_blocks(sidecar: dict) -> list[str]:
    """Top-level sidecar keys worth comparing, in the order the file lists them."""
    return [key for key in sidecar if not _is_provenance(key)]


def _leaves(obj, prefix=""):
    """Flatten nested dict/list structures into (path, value) pairs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _comparable(obj) -> dict[str, object]:
    """Leaves of one sidecar block that carry drawn values."""
    return {key: value for key, value in _leaves(obj) if not _is_ignored(key, value)}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _diff(expected, fresh) -> list[str]:
    """Every leaf-level disagreement between an expected and a fresh block."""
    ref = _comparable(expected)
    new = _comparable(fresh)
    problems = [f"missing in fresh render: {k}" for k in sorted(ref.keys() - new.keys())]
    problems += [f"extra in fresh render: {k}" for k in sorted(new.keys() - ref.keys())]
    for k in sorted(ref.keys() & new.keys()):
        a, b = ref[k], new[k]
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                problems.append(f"{k}: {a!r} != {b!r}")
        elif _is_number(a) and _is_number(b):
            both_nan = (
                isinstance(a, float)
                and isinstance(b, float)
                and math.isnan(a)
                and math.isnan(b)
            )
            if not both_nan and not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9):
                problems.append(f"{k}: {a!r} != {b!r}")
        elif a != b:
            problems.append(f"{k}: {a!r} != {b!r}")
    return problems


def _plan(expected_dir: Path, stem: str, out_dir: Path) -> list[tuple[Path, dict, list[str], Path]]:
    """Sidecar comparisons available for ``stem`` under ``expected_dir``.

    One entry per committed sidecar that carries at least one comparable block.
    A committed sidecar made entirely of provenance contributes nothing and is
    skipped rather than asserted against -- ``figures/reference/
    c1_k5_length_control.meta.json`` is such a file, left over from a render
    that wrote a sidecar the current generator does not.
    """
    plan = []
    for suffix in SIDECAR_SUFFIXES:
        expected_path = expected_dir / f"{stem}{suffix}"
        if not expected_path.exists():
            continue
        expected = json.loads(expected_path.read_text())
        blocks = [
            block
            for block in _data_blocks(expected)
            if expected_dir != REFERENCE or (stem, block) not in REFERENCE_BLOCK_EXEMPTIONS
        ]
        if blocks:
            plan.append((expected_path, expected, blocks, out_dir / f"{stem}{suffix}"))
    return plan


def numeric_leaf_count(plan: list[tuple[Path, dict, list[str], Path]]) -> int:
    """How many numbers a comparison plan actually compares."""
    total = 0
    for _, expected, blocks, _ in plan:
        for block in blocks:
            total += sum(1 for value in _comparable(expected[block]).values() if _is_number(value))
    return total


def _run(plan: list[tuple[Path, dict, list[str], Path]]) -> list[str]:
    """Execute a comparison plan and return every problem found."""
    problems: list[str] = []
    for expected_path, expected, blocks, fresh_path in plan:
        if not fresh_path.exists():
            problems.append(f"fresh render wrote no {fresh_path.name} to compare")
            continue
        fresh = json.loads(fresh_path.read_text())
        for block in blocks:
            if block not in fresh:
                problems.append(f"{fresh_path.name}: fresh sidecar lacks block {block!r}")
                continue
            problems += [
                f"{fresh_path.name}/{block}{p}" for p in _diff(expected[block], fresh[block])
            ]
    return problems


def _why_no_plan(stem: str, absent: str) -> str:
    """Skip reason when ``absent`` pins nothing for ``stem``, naming what does instead."""
    if stem in RENDER_ONLY_STEMS:
        return f"{absent} pins no data, and the figure draws none, so the render check is all of it"
    if absent.startswith("the submitted"):
        return f"{absent} pins no data for this stem, so a regression baseline covers it"
    return f"{absent} is not committed for this stem, which the submitted sidecar covers"


def _report(stem: str, kind: str, problems: list[str]) -> str:
    head = f"{stem}: data differs from the committed {kind} ({len(problems)} problem(s)):"
    shown = problems[:50]
    body = "\n  ".join(shown)
    if len(problems) > len(shown):
        body += f"\n  ... and {len(problems) - len(shown)} more"
    return f"{head}\n  {body}"


@pytest.fixture(scope="session")
def rendered(tmp_path_factory) -> dict[str, Path | Exception]:
    """Run every generator once, mapping each module to its output directory or to the
    exception it raised."""
    outputs: dict[str, Path | Exception] = {}
    for gen in GENERATORS:
        out_dir = tmp_path_factory.mktemp(gen.module.rsplit(".", 1)[-1])
        try:
            importlib.import_module(gen.module).main(RECORDS, out_dir)
        except Exception as exc:  # one broken generator must not hide the others' results
            outputs[gen.module] = exc
        else:
            outputs[gen.module] = out_dir
    return outputs


def _cases() -> list[tuple[Generator, str]]:
    return [(gen, stem) for gen in GENERATORS for stem in gen.stems]


CASES = _cases()
CASE_IDS = [stem for _, stem in CASES]


def _out_dir(rendered, gen: Generator) -> Path:
    out_dir = rendered[gen.module]
    if isinstance(out_dir, Exception):
        pytest.fail(f"{gen.module} raised {type(out_dir).__name__}: {out_dir}")
    return out_dir


@pytest.mark.parametrize("gen,stem", CASES, ids=CASE_IDS)
def test_figure_renders(rendered, gen: Generator, stem: str) -> None:
    out_dir = _out_dir(rendered, gen)
    pdf = out_dir / f"{stem}.pdf"
    assert pdf.exists(), f"{gen.module} did not write {stem}.pdf"
    assert pdf.stat().st_size > 1024, f"{stem}.pdf is suspiciously small"
    assert (REFERENCE / f"{stem}.pdf").exists(), f"no submitted reference PDF for {stem}"


@pytest.mark.parametrize("gen,stem", CASES, ids=CASE_IDS)
def test_data_matches_reference(rendered, gen: Generator, stem: str) -> None:
    out_dir = _out_dir(rendered, gen)
    plan = _plan(REFERENCE, stem, out_dir)
    if not plan:
        pytest.skip(f"{stem}: {_why_no_plan(stem, 'the submitted sidecar')}")
    problems = _run(plan)
    assert not problems, _report(stem, "reference sidecar", problems)


@pytest.mark.parametrize("gen,stem", CASES, ids=CASE_IDS)
def test_data_matches_baseline(rendered, gen: Generator, stem: str) -> None:
    out_dir = _out_dir(rendered, gen)
    plan = _plan(BASELINE, stem, out_dir)
    if not plan:
        pytest.skip(f"{stem}: {_why_no_plan(stem, 'a regression baseline')}")
    problems = _run(plan)
    assert not problems, _report(stem, "regression baseline", problems)


def test_reference_exemptions_are_live() -> None:
    """Every entry of :data:`REFERENCE_BLOCK_EXEMPTIONS` still suppresses a real block.

    A typo in a stem name, or a block the submitted sidecar no longer carries,
    would sit in the table suppressing nothing while reading as if it suppressed
    something.
    """
    for (stem, block), reason in REFERENCE_BLOCK_EXEMPTIONS.items():
        assert reason.strip(), f"{stem}/{block}: an exemption needs a stated reason"
        carriers = [
            suffix
            for suffix in SIDECAR_SUFFIXES
            if (REFERENCE / f"{stem}{suffix}").exists()
            and block in json.loads((REFERENCE / f"{stem}{suffix}").read_text())
        ]
        assert carriers, f"no committed reference sidecar for {stem} carries block {block!r}"


@pytest.mark.parametrize("gen,stem", CASES, ids=CASE_IDS)
def test_numeric_coverage(rendered, gen: Generator, stem: str) -> None:
    """Every stem that draws numbers has at least one of them under comparison.

    Without this the two data tests above can report a pass while comparing
    nothing: a sidecar whose every block is filtered out leaves an empty
    comparison, which trivially agrees with itself.
    """
    out_dir = _out_dir(rendered, gen)
    counts = {
        "reference": numeric_leaf_count(_plan(REFERENCE, stem, out_dir)),
        "baseline": numeric_leaf_count(_plan(BASELINE, stem, out_dir)),
    }
    total = sum(counts.values())
    if stem in RENDER_ONLY_STEMS:
        assert total == 0, (
            f"{stem} is listed as render-only but now compares {total} number(s) "
            f"({counts}). Drop it from RENDER_ONLY_STEMS"
        )
        pytest.skip(f"{stem}: render-only, its sidecar carries no numbers")
    assert total > 0, (
        f"{stem}: no numeric leaf is compared against either the reference sidecar or a "
        f"regression baseline, so a wrong plotted value would not fail any test"
    )
