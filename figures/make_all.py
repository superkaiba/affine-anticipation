"""Regenerate every paper figure from the committed records.

Usage::

    python -m figures.make_all                # all figures -> out/
    python -m figures.make_all --only c3_sae_tier_gradient c4_shared_speakers
    python -m figures.make_all --records records --out out

Each generator in :mod:`figures.registry` is imported and its ``main`` called
once; the table printed at the end says which stems were written.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from figures.registry import GENERATORS, Generator, generator_for

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_generator(gen: Generator, records_dir: Path, out_dir: Path) -> list[Path]:
    """Import ``gen.module`` and run its ``main``; return the files it reports."""
    module = importlib.import_module(gen.module)
    return [Path(p) for p in module.main(records_dir, out_dir)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, default=REPO_ROOT / "records")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--only", nargs="+", metavar="STEM", help="paper figure stems to render")
    args = parser.parse_args(argv)

    if args.only:
        wanted = {generator_for(stem).module for stem in args.only}
        gens = [g for g in GENERATORS if g.module in wanted]
    else:
        gens = list(GENERATORS)

    args.out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for gen in gens:
        print(f"== {gen.module}  ({', '.join(gen.stems)})", flush=True)
        run_generator(gen, args.records, args.out)
        for stem in gen.stems:
            if args.only and stem not in args.only:
                continue
            pdf = args.out / f"{stem}.pdf"
            if pdf.exists() and pdf.stat().st_size > 0:
                print(f"   wrote {pdf.relative_to(REPO_ROOT) if pdf.is_relative_to(REPO_ROOT) else pdf}"
                      f"  ({pdf.stat().st_size:,} bytes)")
            else:
                missing.append(stem)
                print(f"   MISSING {stem}.pdf")
    if missing:
        print(f"{len(missing)} figure(s) not written: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"{sum(len(g.stems) for g in gens)} paper figure(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
