# affine-anticipation

Code and result records for *Affine Anticipation: LLM Answer Representations
Are Largely Linearly Predictable from Context*.

The repository has two tiers:

- **Figures from records (tested).** Every figure in the paper regenerates
  from the result records committed under `records/` with one command. A
  test compares each regenerated figure's displayed data against the
  sidecar of the submitted PDF in `figures/reference/`.
- **Experiment pipeline (archived).** The code that produced those records:
  answer generation and activation capture, metamodel fitting, retrieval
  evaluation, and the per-section analyses. It is documented per record in
  [`pipeline/README.md`](pipeline/README.md) and points at the public
  Hugging Face repositories that hold the activation banks, generated-answer
  banks and fitted maps. It is not packaged as an end-to-end reproduction.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest
pip install -e ".[pipeline]"     # add torch, huggingface_hub, API clients (tier 2 only)
```

Python 3.11 or later. Figures render with the Inter font when it is
installed and fall back to Noto Sans or DejaVu Sans otherwise.

## Regenerate the figures

```bash
python -m figures.make_all                 # every paper figure -> out/
python -m figures.make_all --only c3_sae_tier_gradient
pytest                                     # regenerate and compare against figures/reference/
```

Each generator is a module under `figures/` with a
`main(records_dir, out_dir)` entry point and a command-line interface
(`--records`, `--out`). The table below is generated from
`figures/registry.py`.

<!-- FIGURE TABLE START -->
<!-- FIGURE TABLE END -->

`figures/reference/` holds the PDF of each figure as submitted and, where
the generator wrote one, its provenance sidecar (`<stem>.meta.json`: inputs
with checksums, the displayed data, the rendering environment).

## Records

`records/<experiment>/...` keeps each input record under the path the
experiment pipeline wrote it to. The pipeline README maps every record to
the script that produced it and to the Hugging Face prefix holding the raw
inputs it was computed from.

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
