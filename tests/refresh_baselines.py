"""Write the regression baselines in ``figures/baseline/``.

A baseline is this repository's own first render of a stem's sidecar, committed
so that ``tests/test_figures.py`` can catch a change that moves a drawn value.
It is needed where the submitted sidecar in ``figures/reference/`` cannot serve:
either the submitted figure shipped without a sidecar, or the sidecar it shipped
records settings and prose but none of the plotted numbers.

Run it after a deliberate change to a figure's data, and commit the diff::

    python tests/refresh_baselines.py                 # only the stems that need one
    python tests/refresh_baselines.py --check         # fail if a baseline is stale
    python tests/refresh_baselines.py --stem c1_layer_sweep

Provenance is stripped before writing: checksums, git state, timestamps, output
file names, the style record, and every leaf holding a filesystem path. Those are
the leaves ``tests/test_figures.py`` ignores anyway, and a committed file has no
business carrying the absolute paths of the machine that produced it.

This module is not a test. It imports the test module's own filters so the two
cannot drift apart.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):  # invoked as a script rather than imported
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.registry import GENERATORS, Generator  # noqa: E402
from tests.test_figures import (  # noqa: E402
    BASELINE,
    REFERENCE,
    SIDECAR_SUFFIXES,
    _data_blocks,
    _is_ignored,
    _plan,
    numeric_leaf_count,
)

RECORDS = Path(__file__).resolve().parent.parent / "records"


def _prune(obj, prefix: str = ""):
    """Drop provenance and path leaves, keeping list positions intact."""
    if isinstance(obj, dict):
        kept = {}
        for key, value in obj.items():
            path = f"{prefix}/{key}"
            if _is_ignored(path, value):
                continue
            kept[key] = _prune(value, path)
        return kept
    if isinstance(obj, list):
        return [_prune(value, f"{prefix}[{i}]") for i, value in enumerate(obj)]
    return obj


def baseline_body(sidecar: dict) -> dict:
    """The part of a fresh sidecar worth committing as a baseline."""
    return {key: _prune(sidecar[key], f"/{key}") for key in _data_blocks(sidecar)}


def needs_baseline(stem: str, out_dir: Path) -> list[str]:
    """Why ``stem`` needs a baseline, or an empty list when it does not.

    Two reasons, both read off the committed reference and the fresh render:
    the reference pins no numbers for this stem, or the fresh sidecar carries a
    data block the reference has no counterpart for.
    """
    reasons = []
    if numeric_leaf_count(_plan(REFERENCE, stem, out_dir)) == 0:
        reasons.append("the submitted sidecar pins no numbers")
    for suffix in SIDECAR_SUFFIXES:
        fresh_path = out_dir / f"{stem}{suffix}"
        reference_path = REFERENCE / f"{stem}{suffix}"
        if not fresh_path.exists():
            continue
        fresh_blocks = set(_data_blocks(json.loads(fresh_path.read_text())))
        referenced = (
            set(_data_blocks(json.loads(reference_path.read_text())))
            if reference_path.exists()
            else set()
        )
        for block in sorted(fresh_blocks - referenced):
            reasons.append(f"{fresh_path.name} block {block!r} has no reference counterpart")
    return reasons


def render(gen: Generator, out_dir: Path) -> None:
    importlib.import_module(gen.module).main(RECORDS, out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stem", action="append", help="limit to these stems, repeatable")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare instead of writing, exiting 1 when a baseline is missing or stale",
    )
    args = parser.parse_args(argv)
    wanted = set(args.stem) if args.stem else None

    BASELINE.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    written: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for gen in GENERATORS:
            stems = [s for s in gen.stems if wanted is None or s in wanted]
            if not stems:
                continue
            out_dir = Path(tmp) / gen.module.rsplit(".", 1)[-1]
            render(gen, out_dir)
            for stem in stems:
                reasons = needs_baseline(stem, out_dir)
                if not reasons:
                    print(f"-- {stem}: covered by the submitted sidecar, no baseline")
                    continue
                sidecars = [s for s in SIDECAR_SUFFIXES if (out_dir / f"{stem}{s}").exists()]
                if not sidecars:
                    print(f"-- {stem}: draws no data and writes no sidecar, so it is render-only")
                    continue
                print(f"== {stem}: {', '.join(reasons)}")
                for suffix in sidecars:
                    body = baseline_body(json.loads((out_dir / f"{stem}{suffix}").read_text()))
                    if not body:
                        continue
                    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
                    target = BASELINE / f"{stem}{suffix}"
                    if args.check:
                        if not target.exists():
                            stale.append(f"{target.name} is missing")
                        elif target.read_text() != text:
                            stale.append(f"{target.name} is stale")
                        continue
                    target.write_text(text)
                    written.append(f"{target.name} ({len(text):,} bytes)")
                    print(f"   wrote figures/baseline/{target.name}")

    if args.check:
        for problem in stale:
            print(f"STALE {problem}", file=sys.stderr)
        print(f"{len(stale)} baseline problem(s)")
        return 1 if stale else 0
    print(f"{len(written)} baseline file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
