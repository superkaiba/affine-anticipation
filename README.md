# affine-anticipation

Code and result records for *Affine Anticipation: LLM Answer Representations
Are Largely Linearly Predictable from Context*.

The repository has two tiers:

- **Figures from records (tested).** All 22 paper figures regenerate from the
  result records committed under `records/` with one command. The test suite
  regenerates every one of them and checks that it writes a non-empty PDF.
  Beyond that the checks differ by figure. Where the submitted PDF shipped a
  sidecar that carries its displayed data, the regenerated sidecar is compared
  against it leaf for leaf. The remaining figures are compared against a
  regression baseline committed here, which records this repository's own first
  render and so says nothing about the paper. 21 of the 22 have at least one
  drawn number under comparison. The exception is the standalone legend for
  Figure 2, which draws no data.
- **Experiment pipeline (archived).** The code that produced those records:
  answer generation and activation capture, metamodel fitting, retrieval
  evaluation, and the per-section analyses. It was ported from the research
  repository the work was done in, *explore-persona-space* (EPS,
  https://github.com/superkaiba/explore-persona-space), whose name and issue
  numbers the module docstrings and record paths still carry. It is documented
  per record in [`pipeline/README.md`](pipeline/README.md) and points at the
  public Hugging Face repositories that hold the activation banks,
  generated-answer banks and fitted maps. It is not packaged as an end-to-end
  reproduction. Its GPU and corpus paths need `transformers`, `vllm`,
  `datasets`, `wandb`, `psutil` and `requests` on top of the `pipeline` extra.
  Those are left undeclared because that tier is not a reproduction path.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest
pip install -e ".[pipeline]"     # add torch, huggingface_hub, API clients (tier 2 only)
```

Python 3.11 or later, and the Inter font:

```bash
sudo apt-get install fonts-inter          # Debian, Ubuntu
```

Three generators require Inter and raise without it, because the canvas-fit
guard in `context_answer/plotting/style.py` is calibrated to its text metrics
and refuses content wider than the canvas. They are
`figures/controls_and_scaling.py` (`c1_controls_and_scaling`,
`c1_controls_and_scaling_legend`, `c1_layer_sweep`),
`figures/ctxans_geometry.py` (`c3_ctxans_geometry`) and
`figures/preimage_qualitative_top100.py` (`c5_preimage_qualitative_top100`),
five paper figures between them. The other 17 fall back to Noto Sans or
DejaVu Sans and render with slightly different text metrics.

## Regenerate the figures

```bash
python -m figures.make_all                 # every paper figure -> out/
python -m figures.make_all --only c3_sae_tier_gradient
pytest                                     # regenerate, then compare against
                                           # figures/reference/ and figures/baseline/
```

Each generator is a module under `figures/` with a
`main(records_dir, out_dir)` entry point and a command-line interface
(`--records`, `--out`). The table below is generated from
`figures/registry.py`.

<!-- FIGURE TABLE START -->
| Figure | Paper location | Regenerate with | Ported from |
|---|---|---|---|
| `c1_controls_and_scaling` | Figure 2 (introduction); Appendix A.8 | `python -m figures.controls_and_scaling` | `scripts/make_paper_figure2.py` |
| `c1_controls_and_scaling_legend` | Figure 2 (introduction); Appendix A.8 | `python -m figures.controls_and_scaling` | `scripts/make_paper_figure2.py` |
| `c1_layer_sweep` | Figure 2 (introduction); Appendix A.8 | `python -m figures.controls_and_scaling` | `scripts/make_paper_figure2.py` |
| `c1_crosslayer_grid` | Appendix A.8 | `python -m figures.crosslayer_grid` | `scripts/paper_fig_crosslayer_grid.py` |
| `c1_distractor_scaling` | Appendix A.4 | `python -m figures.distractor_scaling` | `scripts/issue1901_distractor_curve.py` |
| `c1_rollout_count` | Appendix A.12 | `python -m figures.rollout_count` | `scripts/issue1901_k_rollout_ablation.py` |
| `c1_k5_length_control` | Method details | `python -m figures.k5_length_control` | `scripts/issue1902_k5_length_control.py` |
| `c1_offpolicy_origin` | Appendix A.5 | `python -m figures.offpolicy_origin` | `scripts/section4_offpolicy_figure.py` |
| `c1_posttraining_dynamics` | Section 4.3 | `python -m figures.posttraining_dynamics` | `scripts/section43_posttraining_onpolicy_figure.py` |
| `c1_cot_capability` | Section 4.5 | `python -m figures.cot_capability` | `scripts/paper_fig_cot_capability.py` |
| `c3_direction_r2_spectrum` | Section 4.2; Appendix A.3; method details | `python -m figures.section42` | `scripts/make_paper_section42_figures.py` |
| `c3_features_and_shifts` | Section 4.2; Appendix A.3; method details | `python -m figures.section42` | `scripts/make_paper_section42_figures.py` |
| `c3_refusal_swaps_by_class` | Section 4.2; Appendix A.3; method details | `python -m figures.section42` | `scripts/make_paper_section42_figures.py` |
| `c3_sae_tier_gradient` | Section 4.2; Appendix A.3; method details | `python -m figures.section42` | `scripts/make_paper_section42_figures.py` |
| `c3_qualitative_discrimination` | Appendix A.4 | `python -m figures.qualitative_discrimination` | `scripts/issue1901_qualitative_retrieval_failures.py` |
| `c3_ctxans_geometry` | Appendix A.4 | `python -m figures.ctxans_geometry` | `scripts/issue1901_ctxans_geometry_figure.py` |
| `c3_ctxans_projection` | Appendix A.4 | `python -m figures.ctxans_projection` | `scripts/issue1901_ctxans_projection_figure.py` |
| `c4_shared_speakers` | Section 4.4 | `python -m figures.shared_speakers` | `figures/paper/source/c4_shared_speakers.py (Overleaf archive)` |
| `c4_speaker_transfer_full` | Method details | `python -m figures.speaker_transfer_full` | `scripts/issue2054_story_transfer_appendix.py` |
| `c4_turn_transfer_full` | Method details | `python -m figures.turn_transfer_full` | `scripts/plot_issue825_turn_matched.py` |
| `c5_behavior_transfer_original_combined` | Section 4.6 | `python -m figures.behavior_transfer_original_combined` | `scripts/issue1739_old_combined_figure.py` |
| `c5_preimage_qualitative_top100` | Method details | `python -m figures.preimage_qualitative_top100` | `scripts/issue1739_qualitative_preimage_figure.py` |

`fig1_schematic` is a TikZ drawing; compile `figures/fig1_schematic/fig1_schematic.tex` with `pdflatex` (see its README).
<!-- FIGURE TABLE END -->

`figures/reference/` holds the PDF of each figure as submitted and, where
the generator wrote one, its provenance sidecar (`<stem>.meta.json`: inputs
with checksums, the displayed data, the rendering environment).
`figures/reference/fig1_schematic.pdf` is the submitted outlined TikZ build. It
has no generator here, and `pytest` does not cover it.

`figures/baseline/` holds the regression baselines: the data blocks of this
repository's own first render, for the stems whose submitted sidecar cannot
serve as a check, because it either shipped no sidecar or shipped one recording
settings and prose but no plotted values. A baseline carries evidence about this
repository alone. It fails when a change to a generator, a record, or the shared
fitting and plotting code moves a drawn value. Rewrite the baselines
deliberately with `python tests/refresh_baselines.py`, and commit the diff.

## Records

`records/<experiment>/...` keeps each input record under the path the
experiment pipeline wrote it to. The pipeline README maps every record to
the script that produced it and to the Hugging Face prefix holding the raw
inputs it was computed from.

Records and the submitted sidecars in `figures/reference/` are committed
byte-identical to the research repository, so they carry the provenance
metadata they were computed with: EPS issue numbers, Hugging Face revisions,
checksums, timestamps, and file paths from the machine that produced them.
Those paths describe where a value came from and are not paths into this
repository. The figure tests ignore them for that reason.

## Data and models

- Result records, generated-answer banks and activation summaries:
  https://huggingface.co/datasets/superkaiba1/explore-persona-space-data
- Fine-tuned adapters and merged checkpoints:
  https://huggingface.co/superkaiba1/explore-persona-space

## Third-party material

Evaluation questions and rubric text adapted from *Emergent Misalignment*
(Betley et al., 2025, MIT) and *Persona Vectors* (Chen et al., 2025,
Apache-2.0); see `THIRD_PARTY_NOTICES.md` and `licenses/`.

## Citation

```bibtex
@inproceedings{affine-anticipation-2027,
  title     = {Affine Anticipation: LLM Answer Representations Are Largely Linearly Predictable from Context},
  booktitle = {Submitted to the International Conference on Learning Representations (ICLR)},
  year      = {2027},
}
```

## License

MIT, see `LICENSE`.
