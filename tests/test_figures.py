"""Regenerate every registered paper figure and check it against the submitted reference.

For each generator: run ``main`` into a temporary directory, require a non-empty
PDF per paper stem, and when both the submitted sidecar
(``figures/reference/<stem>.meta.json``) and the fresh sidecar carry a
``displayed_data`` block, require every leaf to match (numbers to 1e-9).
Stems whose reference sidecar has no ``displayed_data`` get the render check only.
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


def _diff_displayed_data(reference: dict, fresh: dict) -> list[str]:
    ref = dict(_leaves(reference))
    new = dict(_leaves(fresh))
    problems = [f"missing in fresh render: {k}" for k in ref.keys() - new.keys()]
    problems += [f"extra in fresh render: {k}" for k in new.keys() - ref.keys()]
    for k in ref.keys() & new.keys():
        a, b = ref[k], new[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) and not (
                isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
            ):
                problems.append(f"{k}: {a!r} != {b!r}")
        elif a != b:
            problems.append(f"{k}: {a!r} != {b!r}")
    return problems


@pytest.fixture(scope="session")
def rendered(tmp_path_factory) -> dict[str, Path]:
    """Run every generator once; map module name -> its output directory."""
    outputs: dict[str, Path] = {}
    for gen in GENERATORS:
        out_dir = tmp_path_factory.mktemp(gen.module.rsplit(".", 1)[-1])
        importlib.import_module(gen.module).main(RECORDS, out_dir)
        outputs[gen.module] = out_dir
    return outputs


def _cases() -> list[tuple[Generator, str]]:
    return [(gen, stem) for gen in GENERATORS for stem in gen.stems]


@pytest.mark.parametrize("gen,stem", _cases(), ids=[stem for _, stem in _cases()])
def test_figure_renders_and_matches_reference(rendered, gen: Generator, stem: str) -> None:
    out_dir = rendered[gen.module]
    pdf = out_dir / f"{stem}.pdf"
    assert pdf.exists(), f"{gen.module} did not write {stem}.pdf"
    assert pdf.stat().st_size > 1024, f"{stem}.pdf is suspiciously small"
    assert (REFERENCE / f"{stem}.pdf").exists(), f"no submitted reference PDF for {stem}"

    ref_meta = REFERENCE / f"{stem}.meta.json"
    fresh_meta = out_dir / f"{stem}.meta.json"
    if not ref_meta.exists():
        pytest.skip(f"{stem}: no reference sidecar; render check only")
    reference = json.loads(ref_meta.read_text())
    if "displayed_data" not in reference:
        pytest.skip(f"{stem}: reference sidecar has no displayed_data; render check only")
    assert fresh_meta.exists(), f"{stem}: fresh render wrote no sidecar to compare"
    fresh = json.loads(fresh_meta.read_text())
    problems = _diff_displayed_data(reference["displayed_data"], fresh.get("displayed_data", {}))
    assert not problems, f"{stem}: displayed_data differs:\n  " + "\n  ".join(problems[:20])
