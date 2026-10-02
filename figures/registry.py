"""Registry of the paper's figures and the module that regenerates each one.

Every generator module exposes ``main(records_dir, out_dir) -> list[Path]``
and writes ``<stem>.pdf`` (plus ``<stem>.png`` and a ``<stem>.meta.json``
provenance sidecar where the shared style module supports it) into
``out_dir``. ``python -m figures.make_all`` and ``tests/test_figures.py``
both drive this table; add one entry per ported generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Generator:
    """One ported generator script and the paper figure stems it produces."""

    module: str  # importable module name under figures/
    stems: tuple[str, ...]  # paper figure stems (match figures/reference/<stem>.pdf)
    sections: str  # where the paper uses them
    extra_stems: tuple[str, ...] = field(default_factory=tuple)  # also written, not in the paper
    source: str = ""  # EPS script this was ported from


GENERATORS: tuple[Generator, ...] = (
    Generator(
        module="figures.controls_and_scaling",
        stems=("c1_controls_and_scaling", "c1_controls_and_scaling_legend", "c1_layer_sweep"),
        sections="Introduction (Figure 2), Appendix A.8",
        source="scripts/make_paper_figure2.py",
    ),
    Generator(
        module="figures.crosslayer_grid",
        stems=("c1_crosslayer_grid",),
        sections="Appendix A.8",
        source="scripts/paper_fig_crosslayer_grid.py",
    ),
    Generator(
        module="figures.rollout_count",
        stems=("c1_rollout_count",),
        sections="Appendix A.12",
        source="scripts/issue1901_k_rollout_ablation.py",
    ),
    Generator(
        module="figures.posttraining_dynamics",
        stems=("c1_posttraining_dynamics",),
        sections="Section 4.3",
        source="scripts/section43_posttraining_onpolicy_figure.py",
    ),
    Generator(
        module="figures.section42",
        stems=(
            "c3_direction_r2_spectrum",
            "c3_features_and_shifts",
            "c3_refusal_swaps_by_class",
            "c3_sae_tier_gradient",
        ),
        sections="Section 4.2, Appendix A.3, method details",
        extra_stems=(
            "c3_pair_shifts",
            "c3_directions_and_pairs",
            "c3_directions_and_features",
            "c3_failures_and_shifts",
            "c3_element_shifts",
            "c3_element_shifts_by_slot",
        ),
        source="scripts/make_paper_section42_figures.py",
    ),
    Generator(
        module="figures.qualitative_discrimination",
        stems=("c3_qualitative_discrimination",),
        sections="Appendix A.4",
        source="scripts/issue1901_qualitative_retrieval_failures.py",
    ),
    Generator(
        module="figures.shared_speakers",
        stems=("c4_shared_speakers",),
        sections="Section 4.4 (two-panel predecessor of the submitted figure)",
        source="scripts/paper_fig_shared_speakers.py",
    ),
)


def generator_for(stem: str) -> Generator:
    """Return the generator that produces ``stem``; raise KeyError when none does."""
    for gen in GENERATORS:
        if stem in gen.stems or stem in gen.extra_stems:
            return gen
    raise KeyError(stem)


def paper_stems() -> list[str]:
    """Every paper figure stem the registry can regenerate, in registry order."""
    return [stem for gen in GENERATORS for stem in gen.stems]
