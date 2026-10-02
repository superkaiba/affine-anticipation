# Experiment pipeline (archived)

This directory holds the experiment code behind the paper's result records:
answer generation and activation capture, metamodel fits, retrieval evaluation,
and the behavior and speaker transfer analyses. It is an archive for inspection,
not a reproduction harness. See "Scope and limits" below before trying to run
anything.

The figure tier is separate and does run end to end. It regenerates every paper
figure from the committed records under `records/`, with no code in this
directory involved.

## Layout

```
pipeline/scripts/        32 archived entry points and their local modules
pipeline/verify_records.py   offline consistency check over records/pipeline/
pipeline/SHA256SUMS      checksums of records/pipeline/
records/pipeline/        the 13 result records these scripts produced
context_answer/          the library modules the scripts import
```

`context_answer/` also carries the figure tier's plotting package. The archived
scripts import `context_answer.plotting.paper_plots` for figure style, and
`context_answer.{analysis,eval,llm,orchestrate}`, `context_answer.personas` and
`context_answer.atomic_io` for everything else.

## Offline verification

```sh
python pipeline/verify_records.py
```

Standard library only, Python 3.11 or later. It verifies `pipeline/SHA256SUMS`
against `records/pipeline/`, cross-checks the ten capability-panel rows against
the archived capability analysis and the selected-layer fit records, prints the
appendix table values, and checks the retrieval pool composition and chance
rate. It performs no downloads and no model inference.

## Entry points

### `scripts/capture_answers.py`

Answer generation and activation capture at the largest training size. Builds a
roughly 960,000-context prompt pool by streaming LMSYS-1M to exhaustion and
topping up from WildChat-1M, with a near-duplicate gate against the 1,400 pinned
validation and test prompts, writes the pool as a manifest so every capture
shard reads the same deterministic context list, then captures the prompt-last
context vector and the mean-response answer vector at layers 14, 19 and 26.
Sharded across pods and GPUs, with an upload, verify and purge cycle per chunk.

Inputs: the two Hugging Face conversation corpora (`lmsys/lmsys-chat-1m`,
`allenai/WildChat-1M`), Qwen2.5-7B-Instruct weights, and a GPU. Writes capture
shards, the pool manifest and rollout text to the data repo under
`issue779_monitoring/fitter-fair-comparison-n1m/`.

### `scripts/fit_mlp.py`

Nonlinear metamodel fitting, three jobs. Job A fits one MLP of width 8192 per
layer 0 to 27 at n_train 3,600 on the all-layer bundle, gated against the banked
fair-comparison parity cells. Job B adds layer-19 scaling points at n_train
5,000 and 10,000. Job C runs the dense layer-19 ladder over eight fresh rungs
from 5,000 to 500,000 plus the banked 963,444 point, with MLP, ridge and
blockwise identity-plus-bias fits on identical train subsets per rung, and
reports the whitened-CSLS retrieval battery alongside held-out R-squared.

Inputs: the staged pass-B activation bundle and the n1m capture shards from the
data repo, and a GPU. Writes per-fit records and the dense ladder under
`eval_results/issue_1901/paper_densify/`.

### `scripts/evaluate_retrieval.py`

The main 10,000-candidate retrieval evaluation. Holds every banked prediction
fixed and changes only the pool: 942 deduplicated query targets plus the first
9,058 distractors of the banked distractor order, each entry a five-rollout mean
so the pool stays homogeneous with the targets. Layer 19, whitened cosine plus
two-sided CSLS at k equal to 10, strict top-k with ties counted as failures.
Gated by reproducing the banked 942-pool top-1 and top-5 of every arm to within
1e-9 before any 10,000-candidate number is reported. Zero GPU, no refit.

Inputs: the banked predictions, the five-rollout resample shards and the
layer-19 whitening statistics from the data repo. Produces
`records/pipeline/retrieval.json`.

## Record and label table

`EPS issue` is the issue number in the authors' private research repository, from
which every script here was ported. The supplement renamed `issueNNNN_*` scripts
to `experiment_0NN_*`, and this release keeps those script names. Hugging Face
path prefixes are not renamed: the `HF prefix` column below is what the code
actually uses, so a prefix read out of a script resolves against the public
repos as written.

| Script | Private-repo original | Issue | Computes | Record produced | HF prefix consumed |
|---|---|---|---|---|---|
| `capture_answers.py` | `issue779_ffc_n1m_generate_capture.py` | 779 | n1m pool manifest plus trimmed context and answer capture at layers 14, 19, 26 | none (produces the n1m banks) | `issue779_monitoring/fitter-fair-comparison-n1m/` |
| `fit_mlp.py` | `issue1901_paper_densify_mlp.py` | 1901 | MLP width-8192 layer curve, layer-19 scaling points, dense layer-19 ladder with ridge and identity-plus-bias baselines | none (writes `eval_results/issue_1901/paper_densify/`) | `issue779_monitoring/analysis_tensors/pass_b`, `issue1901_mlpdense`, `issue1491_scale_ladder` |
| `evaluate_retrieval.py` | `issue1901_fig2_pool10k.py` | 1901 | Figure 2 panels B and C rescored at a 10,000-candidate pool, whitened cosine plus CSLS | `retrieval.json` | `issue1901_avgpool/analysis_tensors/kresample`, `issue1901_fig2baselines/analysis_tensors`, `issue1901_metrics` |
| `experiment_005_ladder_fits.py` | `issue1491_ladder_fits.py` | 1491 | per-scale ladder driver: primary cell, floors, noise ceiling, WildChat transfer | none (writes `eval_results/issue_1491/scale_ladder/`) | `issue1491_scale_ladder/<slug>` |
| `experiment_009_figure2_five_rollout_scaling.py` | `issue1901_figure2_five_rollout_scaling.py` | 1901 | Figure 2 training-size sweep rescored on five-rollout targets with duplicate collapse | none (writes `eval_results/issue_1901/figure2_five_rollout_scaling.json`) | `issue1901_avgpool`, `issue1901_mlpdense`, `issue779_monitoring` |
| `experiment_009_metric_battery.py` | `issue1901_metric_battery.py` | 1901 | mapping-quality metric battery: 1M-context maps against every baseline on every metric | none (writes `eval_results/issue_1901/metric_battery/`) | `issue1901_metrics`, `issue1901_wildchat`, `issue779_monitoring/n1m_readout/weights` |
| `experiment_009_paper_densify.py` | `issue1901_paper_densify.py` | 1901 | CPU-side paper-figure densification of the ridge ladder | none (writes `eval_results/issue_1901/paper_densify/`) | `issue1901_paper_densify`, `issue779_monitoring` |
| `experiment_009_singleturn_retrieval_final.py` | `issue1901_singleturn_retrieval_final.py` | 1901 | final single-turn retrieval read with training-bank whitening and duplicate removal | none (writes `eval_results/issue_1901/singleturn_retrieval_final/summary.json`) | `issue1901_avgpool`, `issue1901_metrics`, `issue1901_mlpdense` |
| `experiment_020_common.py` | `issue404_common.py` | 404 | shared pair definitions, dataset fetchers and context builders for the persona-distance predictors | none (module) | `issue404_pair_<cell>_seed<S>`, `issue458_prep_datasets` |
| `experiment_031_common.py` | `issue658_common.py` | 658 | shared constants, store loaders and probe-pool pins for the base-model foundation line | none (module) | `issue658_theory_assumptions`, `issue594_context_geometry` |
| `experiment_031_fit_predictors.py` | `issue658_fit_predictors.py` | 658 | predictor fits and statistics over the base-model activation store: leave-one-context-out MLP and projection reads of E0 | none (writes `eval_results/issue_658/`) | `issue658_theory_assumptions` |
| `experiment_038_arm_headline.py` | `issue779_arm_headline.py` | 779 | three-arm headline, target construction, noise ceiling and grouping | none (writes `eval_results/issue_779/`) | `issue779_monitoring` |
| `experiment_038_arm_headline_summaries.py` | `issue779_arm_headline_summaries.py` | 779 | three-arm headline rerun on alternative answer summaries | none (writes `eval_results/issue_779/arm_headline_summaries.json`) | `issue779_monitoring` |
| `experiment_038_arm_headline_summaries2.py` | `issue779_arm_headline_summaries2.py` | 779 | arm-headline extension: pass-2 and cross-layer summary variants | none (writes `eval_results/issue_779/arm_headline_summaries2.json`) | `issue779_monitoring` |
| `experiment_038_capture_answer_summaries.py` | `issue779_capture_answer_summaries.py` | 779 | teacher-forced answer-summary capture, GPU phase | none (uploads capture tensors) | `issue779_monitoring` |
| `experiment_038_capture_answer_summaries_pass2.py` | `issue779_capture_answer_summaries_pass2.py` | 779 | answer-summary capture pass 2 at next-turn template positions | none (uploads capture tensors) | `issue779_monitoring` |
| `experiment_038_collect.py` | `issue779_collect.py` | 779 | shared data-collection pass producing the R1, R2 and R3 inputs and the dependent variable | none (uploads rollouts and tables) | `issue779_monitoring` |
| `experiment_038_common.py` | `issue779_common.py` | 779 | shared constants and helpers for the context-to-answer monitoring line, including the adapted Persona Vectors trait artifacts | none (module) | `issue779_monitoring` |
| `experiment_038_extract_rb.py` | `issue779_extract_rb.py` | 779 | persona-vector `r_B` extraction by mean difference of contrastive activations | none (uploads `r_b.pt`) | `issue779_monitoring` |
| `experiment_038_ffc_baselines.py` | `issue779_ffc_baselines.py` | 779 | consecutive-token-similarity baselines, shuffled-pairing floor and predict-the-mean, merged into the fitter comparison | none (merges into `fair_comparison.json`) | `issue779_monitoring/analysis_tensors/pass_b` |
| `experiment_038_ffc_n10k_generate_capture.py` | `issue779_ffc_n10k_generate_capture.py` | 779 | corpus extension and capture at n_train 10,000 | none (produces the n10k banks) | `issue779_monitoring/fitter-fair-comparison-n10k` |
| `experiment_038_ffc_n1m_fits.py` | `issue779_ffc_n1m_fits.py` | 779 | fits at n_train up to about 1,000,000 over the LMSYS plus WildChat pool, provenance-aware | none (writes `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json`) | `issue779_monitoring/fitter-fair-comparison-n1m` |
| `experiment_038_ffc_n50k_fits.py` | `issue779_ffc_n50k_fits.py` | 779 | fits at n_train 50,000 | none (writes `eval_results/issue_779/fitter-fair-comparison-n50k/n50k_fits.json`) | `issue779_monitoring/fitter-fair-comparison-n50k/final_token_capture` |
| `experiment_038_ffc_n50k_generate_capture.py` | `issue779_ffc_n50k_generate_capture.py` | 779 | corpus extension and 28-layer capture at n_train 50,000, multi-GPU sharded | none (produces the n50k banks) | `issue779_monitoring/fitter-fair-comparison-n50k` |
| `experiment_038_fitter_fair_comparison.py` | `issue779_fitter_fair_comparison.py` | 779 | validation-selected fair comparison of ridge, MLP, residual-skip and kernel fitters under one fixed split | none (writes `eval_results/issue_779/fitter-fair-comparison/fair_comparison.json`) | `issue779_monitoring/analysis_tensors/pass_b` |
| `experiment_038_percontext_recon.py` | `issue779_percontext_recon.py` | 779 | two per-context reconstruction reads for the answer-side map `h` | none (writes `eval_results/issue_779/`) | `issue779_monitoring` |
| `experiment_038_stage1.py` | `issue779_stage1.py` | 779 | stage-1 headline: ridge and MLP for `h` and `g`, R1 and R2 read-outs, gate 1 | none (writes `eval_results/issue_779/stage1_headline.json`) | `issue779_monitoring` |
| `experiment_039_adhoc_lofo_heatmaps.py` | `issue810_adhoc_lofo_heatmaps.py` | 810 | leave-one-feature-out heatmaps of reconstruction skill by answer summary | none (writes `eval_results/issue_810/reconstruction_skill_by_summary.json`) | `issue658_theory_assumptions`, `issue594_context_geometry` |
| `experiment_039_batched_null.py` | `issue810_batched_null.py` | 810 | batched shuffle-null refits for the reconstruction and read-out fits | none (feeds the 810 fit records) | `issue658_theory_assumptions` |
| `experiment_039_common.py` | `issue810_common.py` | 810 | shared helpers for the answer-side summary and position sweep, including the probe-pool pins | none (module) | `issue658_theory_assumptions`, `issue594_context_geometry` |
| `experiment_039_fit_readout.py` | `issue810_fit_readout.py` | 810 | behavior read-out: held-out Spearman of predicted against graded E0, fixed `r_B` projection and trained leave-one-context-out ridge | none (writes `eval_results/issue_810/`) | `issue658_theory_assumptions` |
| `experiment_039_fit_reconstruction.py` | `issue810_fit_reconstruction.py` | 810 | reconstruction: held-out skill-over-mean R-squared per layer and summary, with a selection-symmetric label-shuffle null | none (writes `eval_results/issue_810/`) | `issue658_theory_assumptions`, `issue594_context_geometry` |

The capability records come from a later experiment that reuses the fitting and
retrieval recipes above rather than from a script archived here:

| Record | Source | Issue | Contents |
|---|---|---|---|
| `capability.json` | `eval_results/issue_2588/mapping_rank_vs_capability.json` | 2588 | the per-model mapping scores, rank definitions and rank analyses across the eleven models |
| `capability_panel.json` | selection over issue 2588's `rank_relationships.json` and `mapping_rank_vs_capability.json` | 2588 | the ten rows plotted in the capability panel, with the permutation tests, the familywise correction and the excluded row |
| `fits/*.json` (10) | per-cell prompt-state fit records, held on the data repo | 2588 | layer sweep, selected layer and penalty, sample counts, retrieval metrics and baseline floors, one file per model cell |

## Inputs the scripts expect

None of the large inputs are in this repository. They live on two public Hugging
Face repos:

- data: `superkaiba1/explore-persona-space-data`
- models and adapters: `superkaiba1/explore-persona-space`

Activation banks, generated-answer banks, rollout text, fitted weight tensors
and sufficient statistics are all under the data repo at the prefixes in the
table above. Base model weights come from the usual public model repos
(Qwen2.5-7B-Instruct for the main line, the Qwen3 and OLMo families for the
capability panel).

There is a third repo, `superkaiba1/explore-persona-space-overflow`, used as a
fallback when a prefix hits the per-repo file-count limit. It is private and not
publicly fetchable. The constant is kept so the three-repo upload logic reads
correctly.

Prefixes appear in the code exactly as they exist on the repos, for example
`issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/` in
`capture_answers.py` and `issue1901_mlpdense/analysis_tensors/` in `fit_mlp.py`.

The scripts resolve `context_answer` from the installed package, so either
install the repository (`pip install -e .`) or put the repository root on
`PYTHONPATH`. Sibling-script imports such as `import capture_answers` resolve
from `pipeline/scripts/` without extra setup. One library module,
`context_answer/analysis/vectorized_mlp_skill.py`, imports a sibling experiment
script at module level, so importing it also needs `pipeline/scripts/` on
`PYTHONPATH`.

## Experiment label map

Every `experiment_0NN` label still in this directory, in the records, and in the
code comments maps to one private-repo issue number. The labels survive as
script and module names, logger names, cross-references in docstrings, and
`eval_results/` paths, so the map is what you need to follow a reference such as
"the #038 pass-B bundle" or `experiment_031_fit_predictors` back to its issue.
Hugging Face prefixes are not in that set: those carry their real
`issue<N>_<slug>` spelling. The map is complete, and the labels were assigned in
lexicographic order of the issue number, which is why `experiment_016` sits
between `experiment_015` and `experiment_017`.

| Label | Issue | Label | Issue | Label | Issue | Label | Issue |
|---|---|---|---|---|---|---|---|
| `experiment_001` | 1090 | `experiment_013` | 2225 | `experiment_025` | 503 | `experiment_037` | 778 |
| `experiment_002` | 1434 | `experiment_014` | 2329 | `experiment_026` | 504 | `experiment_038` | 779 |
| `experiment_003` | 1481 | `experiment_015` | 2332 | `experiment_027` | 594 | `experiment_039` | 810 |
| `experiment_004` | 1489 | `experiment_016` | 2588 | `experiment_028` | 604 | `experiment_040` | 811 |
| `experiment_005` | 1491 | `experiment_017` | 281 | `experiment_029` | 650 | `experiment_041` | 825 |
| `experiment_006` | 1586 | `experiment_018` | 354 | `experiment_030` | 651 | `experiment_042` | 841 |
| `experiment_007` | 1739 | `experiment_019` | 389 | `experiment_031` | 658 | `experiment_043` | 906 |
| `experiment_008` | 1774 | `experiment_020` | 404 | `experiment_032` | 664 | `experiment_044` | 922 |
| `experiment_009` | 1901 | `experiment_021` | 444 | `experiment_033` | 667 | `experiment_045` | 928 |
| `experiment_010` | 2202 | `experiment_022` | 458 | `experiment_034` | 671 | `experiment_046` | 931 |
| `experiment_011` | 2203 | `experiment_023` | 493 | `experiment_035` | 722 | | |
| `experiment_012` | 2224 | `experiment_024` | 502 | `experiment_036` | 763 | | |

## Scope and limits

The code here is archived, not maintained, and not a working reproduction path.

- The banks, weights and fitted tensors it reads are not in this repository.
  Several phases need multiple GPUs and days of wall time.
- There is no lockfile for this tier. The library versions each fit actually ran
  under are recorded in the `meta` block of every file in `records/pipeline/fits/`.
- The code retains historical research infrastructure: cloud transfer, pod
  bootstrap, preflight checks, multi-organization API dispatch. Those paths
  depend on accounts and hosts that are not part of this release.
- The renamed data-disk mount `/mnt/context_answer-data/` is a research-VM
  leftover and appears at about twenty sites, mostly throughout the
  orchestration modules as a shared-VM default: `orchestrate/env.py`
  (`_SHARED_VM_DATA_DISK`), `orchestrate/preflight.py` (`CA_DATA_DISK_MOUNT`
  and a swapfile path), and command-line help and defaults in
  `pipeline/scripts/experiment_009_metric_battery.py`. Two of those paths
  additionally carry an `anonymous_author` placeholder (in
  `experiment_038_arm_headline.py` and
  `experiment_038_arm_headline_summaries.py`). None of them resolve in this
  release; all are overridable.
- Several pod-side scripts default to `/workspace/...` for the Hugging Face
  cache, log sink and capture directories. Those are configurable defaults, set
  by `os.environ.setdefault` or overridable on the command line.
- Six revision identifiers could not be recovered and keep their
  `anonymous_revision_*` placeholder: `anonymous_revision_003` and
  `anonymous_revision_014` in `capability_panel.json`, and
  `anonymous_revision_009`, `anonymous_revision_017`, `anonymous_revision_020`
  and `anonymous_revision_030` in the `fits/*.json` `meta.git_sha` field. The
  records they label were produced on compute that did not commit them to the
  research repository, so there is no original to read the value from. Every
  other revision and every artifact-repository placeholder has been restored to
  its real value.

Third-party evaluation material included here is attributed in
`THIRD_PARTY_NOTICES.md` at the repository root, with the upstream licenses
under `licenses/`.
