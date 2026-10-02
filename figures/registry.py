"""Registry of the paper's figures and the module that regenerates each one.

Every generator module exposes ``main(records_dir, out_dir) -> list[Path]``
and writes ``<stem>.pdf`` (plus ``<stem>.png`` and a ``<stem>.meta.json``
provenance sidecar) into ``out_dir``. ``python -m figures.make_all`` and
``tests/test_figures.py`` both drive this table; ``make_all --table`` renders
it as the README's figure table.
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
    source: str = ""  # script this was ported from (research repository path)


GENERATORS: tuple[Generator, ...] = (
    Generator(
        module="figures.controls_and_scaling",
        stems=("c1_controls_and_scaling", "c1_controls_and_scaling_legend", "c1_layer_sweep"),
        sections="Figure 2 (introduction); Appendix A.8",
        source="scripts/make_paper_figure2.py",
    ),
    Generator(
        module="figures.crosslayer_grid",
        stems=("c1_crosslayer_grid",),
        sections="Appendix A.8",
        source="scripts/paper_fig_crosslayer_grid.py",
    ),
    Generator(
        module="figures.distractor_scaling",
        stems=("c1_distractor_scaling",),
        sections="Appendix A.4",
        source="scripts/issue1901_distractor_curve.py",
    ),
    Generator(
        module="figures.rollout_count",
        stems=("c1_rollout_count",),
        sections="Appendix A.12",
        source="scripts/issue1901_k_rollout_ablation.py",
    ),
    Generator(
        module="figures.k5_length_control",
        stems=("c1_k5_length_control",),
        sections="Method details",
        source="scripts/issue1902_k5_length_control.py",
    ),
    Generator(
        module="figures.offpolicy_origin",
        stems=("c1_offpolicy_origin",),
        sections="Appendix A.5",
        source="scripts/section4_offpolicy_figure.py",
    ),
    Generator(
        module="figures.posttraining_dynamics",
        stems=("c1_posttraining_dynamics",),
        sections="Section 4.3",
        source="scripts/section43_posttraining_onpolicy_figure.py",
    ),
    Generator(
        module="figures.cot_capability",
        stems=("c1_cot_capability",),
        sections="Section 4.5",
        source="scripts/paper_fig_cot_capability.py",
    ),
    Generator(
        module="figures.section42",
        stems=(
            "c3_direction_r2_spectrum",
            "c3_features_and_shifts",
            "c3_refusal_swaps_by_class",
            "c3_sae_tier_gradient",
        ),
        sections="Section 4.2; Appendix A.3; method details",
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
        module="figures.ctxans_geometry",
        stems=("c3_ctxans_geometry",),
        sections="Appendix A.4",
        source="scripts/issue1901_ctxans_geometry_figure.py",
    ),
    Generator(
        module="figures.ctxans_projection",
        stems=("c3_ctxans_projection",),
        sections="Appendix A.4",
        source="scripts/issue1901_ctxans_projection_figure.py",
    ),
    Generator(
        module="figures.shared_speakers",
        stems=("c4_shared_speakers",),
        sections="Section 4.4",
        source="figures/paper/source/c4_shared_speakers.py (Overleaf archive)",
    ),
    Generator(
        module="figures.speaker_transfer_full",
        stems=("c4_speaker_transfer_full",),
        sections="Method details",
        source="scripts/issue2054_story_transfer_appendix.py",
    ),
    Generator(
        module="figures.turn_transfer_full",
        stems=("c4_turn_transfer_full",),
        sections="Method details",
        source="scripts/plot_issue825_turn_matched.py",
    ),
    Generator(
        module="figures.behavior_transfer_original_combined",
        stems=("c5_behavior_transfer_original_combined",),
        sections="Section 4.6",
        source="scripts/issue1739_old_combined_figure.py",
    ),
    Generator(
        module="figures.preimage_qualitative_top100",
        stems=("c5_preimage_qualitative_top100",),
        sections="Method details",
        source="scripts/issue1739_qualitative_preimage_figure.py",
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


def markdown_table() -> str:
    """Render the registry as the README's figure table."""
    lines = [
        "| Figure | Paper location | Regenerate with | Ported from |",
        "|---|---|---|---|",
    ]
    for gen in GENERATORS:
        cmd = f"`python -m {gen.module}`"
        for stem in gen.stems:
            lines.append(f"| `{stem}` | {gen.sections} | {cmd} | `{gen.source}` |")
    lines.append("")
    lines.append(
        "`fig1_schematic` is a TikZ drawing; compile `figures/fig1_schematic/fig1_schematic.tex` "
        "with `pdflatex` (see its README)."
    )
    return "\n".join(lines)
