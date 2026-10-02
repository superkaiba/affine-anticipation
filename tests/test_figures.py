"""Regenerate every registered paper figure and check it against the submitted reference.

Two checks per paper figure stem:

* ``test_figure_renders``: the generator writes a non-empty PDF for the stem and
  a submitted reference PDF exists in ``figures/reference/``.
* ``test_data_matches_reference``: every data block the submitted sidecar
  (``figures/reference/<stem>.meta.json``) carries is reproduced leaf for leaf by
  the fresh render's sidecar (numbers to 1e-9). Leaves that hold file paths are
  ignored, since the reference was rendered from a different checkout. Stems
  whose reference sidecar carries none of the known data blocks are skipped
  here and covered by the render check only.
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

# Sidecar keys that hold plotted values rather than provenance. Each generator
# writes its own sidecar shape; the shared style module writes ``displayed_data``.
DATA_BLOCKS = (
    "displayed_data",
    "plotted",
    "stats",
    "data",
    "cells",
    "encodings",
    "turn_matrices",
    "groups",
    "original_panels",
    "comparison_panel",
    "displayed_metrics",
    "displayed_settings",
    "transfer_errorbars",
    "rows",
    "per_k",
)
PATH_LEAF_MARKERS = ("path", "file", "dir", "root", "source_path", "input_files")


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


def _is_path_leaf(key: str, value) -> bool:
    last = key.rsplit("/", 1)[-1].split("[", 1)[0].lower()
    if any(m in last for m in PATH_LEAF_MARKERS):
        return True
    return isinstance(value, str) and ("/" in value) and (value.endswith((".json", ".jsonl", ".npz", ".py", ".pdf", ".png")))


def _diff(reference, fresh) -> list[str]:
    ref = {k: v for k, v in _leaves(reference) if not _is_path_leaf(k, v)}
    new = {k: v for k, v in _leaves(fresh) if not _is_path_leaf(k, v)}
    problems = [f"missing in fresh render: {k}" for k in sorted(ref.keys() - new.keys())]
    problems += [f"extra in fresh render: {k}" for k in sorted(new.keys() - ref.keys())]
    for k in sorted(ref.keys() & new.keys()):
        a, b = ref[k], new[k]
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                problems.append(f"{k}: {a!r} != {b!r}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
            if not both_nan and not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9):
                problems.append(f"{k}: {a!r} != {b!r}")
        elif a != b:
            problems.append(f"{k}: {a!r} != {b!r}")
    return problems


@pytest.fixture(scope="session")
def rendered(tmp_path_factory) -> dict[str, Path | BaseException]:
    """Run every generator once; map module name -> its output directory, or the exception it raised."""
    outputs: dict[str, Path | BaseException] = {}
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


def _out_dir(rendered, gen: Generator) -> Path:
    out_dir = rendered[gen.module]
    if isinstance(out_dir, BaseException):
        pytest.fail(f"{gen.module} raised {type(out_dir).__name__}: {out_dir}")
    return out_dir


@pytest.mark.parametrize("gen,stem", _cases(), ids=[stem for _, stem in _cases()])
def test_figure_renders(rendered, gen: Generator, stem: str) -> None:
    out_dir = _out_dir(rendered, gen)
    pdf = out_dir / f"{stem}.pdf"
    assert pdf.exists(), f"{gen.module} did not write {stem}.pdf"
    assert pdf.stat().st_size > 1024, f"{stem}.pdf is suspiciously small"
    assert (REFERENCE / f"{stem}.pdf").exists(), f"no submitted reference PDF for {stem}"


@pytest.mark.parametrize("gen,stem", _cases(), ids=[stem for _, stem in _cases()])
def test_data_matches_reference(rendered, gen: Generator, stem: str) -> None:
    out_dir = _out_dir(rendered, gen)
    ref_meta = REFERENCE / f"{stem}.meta.json"
    if not ref_meta.exists():
        pytest.skip(f"{stem}: the submitted figure has no sidecar")
    reference = json.loads(ref_meta.read_text())
    blocks = [k for k in DATA_BLOCKS if k in reference]
    if not blocks:
        pytest.skip(f"{stem}: reference sidecar has no data block ({', '.join(list(reference)[:6])})")
    fresh_meta = out_dir / f"{stem}.meta.json"
    assert fresh_meta.exists(), f"{stem}: fresh render wrote no sidecar to compare"
    fresh = json.loads(fresh_meta.read_text())
    problems: list[str] = []
    for block in blocks:
        if block not in fresh:
            problems.append(f"fresh sidecar lacks block {block!r}")
            continue
        problems += [f"{block}{p}" for p in _diff(reference[block], fresh[block])]
    assert not problems, f"{stem}: data differs from the submitted sidecar:\n  " + "\n  ".join(problems[:25])
