# Release plan: affine-anticipation

Public code release for *Affine Anticipation: LLM Answer Representations Are
Largely Linearly Predictable from Context* (ICLR 2027 submission mTuspheEuX).
Decisions (Thomas, 2026-10-02): tiered scope, public under `superkaiba`, MIT,
repo `affine-anticipation`, package `context_answer`.

Source repo: `/home/thomasjiralerspong/explore-persona-space` (EPS). Package
there is `explore_persona_space`. Anonymous ICLR supplement (already renamed to
`context_answer`): `/mnt/eps-data/thomasjiralerspong/exports/iclr2027-supplement/extracted/anonymous-supplement/`.
Submitted figure PDFs and sidecars: `/home/thomasjiralerspong/overleaf-6a59c927/figures/paper/`.

## Tiers

- **Tier 1, tested.** Every paper figure regenerates from committed result
  records with one command. `python -m figures.make_all` writes every figure to
  `out/`. `pytest` regenerates each figure and checks it against the committed
  reference (the submitted PDF's sidecar `displayed_data` where one exists,
  otherwise that the script runs and writes a non-empty PDF).
- **Tier 2, archived.** The experiment pipeline (answer capture, activation
  capture, metamodel fits, retrieval evaluation, SAE reads, speaker and
  behavior transfer) ported from the supplement into `pipeline/`, documented
  per result record, with Hugging Face pointers to the banks it consumed. Not
  promised to run end-to-end.

## Layout

```
context_answer/            # package (tier 1 + shared helpers)
  plotting/style.py        # from EPS analysis/c2a_plot_style.py
  plotting/row_labels.py   # from EPS analysis/c2a_row_labels.py
  plotting/paired_ci.py    # from EPS analysis/paired_ci.py
  plotting/paper_plots.py  # from EPS analysis/paper_plots.py
  provenance.py            # git short sha helper (replaces orchestrate.provenance)
figures/
  registry.py              # stem -> (module, function); one entry per paper figure
  make_all.py              # python -m figures.make_all [--only STEM]
  <group>.py               # ported generator scripts, one per original script
  reference/<stem>.pdf     # the submitted PDF (copied from Overleaf figures/paper)
  reference/<stem>.meta.json  # its sidecar when one exists
records/<issue_N>/...      # input JSON/JSONL, path = EPS eval_results path minus the prefix
pipeline/                  # tier 2 (from the supplement's source/)
tests/test_figures.py
README.md, LICENSE (MIT), THIRD_PARTY_NOTICES.md, licenses/, pyproject.toml
```

## Porting rules (every workstream)

1. Keep the original algorithm and numeric settings. Rename, re-path, and
   strip; never "improve" the science.
2. No `explore_persona_space` imports. Replace `orchestrate.env.load_dotenv`
   with nothing, `orchestrate.provenance._git_short_sha` with
   `context_answer.provenance.git_short_sha`, `task_workflow.repo_root()` with
   a `Path(__file__)`-relative repo root. No HF downloads, no `uv run`
   assumptions, no `.env`. Plain `python`.
3. Inputs: scripts read `records/...` (same relative path as the EPS
   `eval_results/...`). Copy each input file in; if one is over 20 MB, stop and
   report it instead of committing it.
4. Outputs: scripts write to an `--out` directory (default `out/`), never into
   `figures/reference/`.
5. No absolute VM paths, no tokens, no `.claude` or `tasks/` references, no
   issue numbers in user-facing names (an `issue_N` directory under `records/`
   is fine; a docstring may say "EPS issue 1901" for traceability).
6. Each ported script gets a module docstring naming the paper figure stem(s)
   it produces, the paper section, and the EPS script + commit it came from.
7. Commit in your worktree by explicit path, small commits, message
   `<stem(s)>: port <original script>`. Report every figure you could not port
   and why.

## Paper figures (23) and where their generators are

| Stem | Section | Generator (EPS) | Inputs |
|---|---|---|---|
| c1_controls_and_scaling (+_legend) | intro | scripts/make_paper_figure2.py | eval_results/issue_1901/{avgtarget_plots/plot1_avg.json, figure2_five_rollout_scaling.json, fig2_pool10k/fig2_pool10k.json} |
| c1_layer_sweep | a8 | scripts/make_paper_figure2.py | same |
| c1_crosslayer_grid | a8 | scripts/paper_fig_crosslayer_grid.py | eval_results/issue_1901/xlayer_grid/{grid,bootstrap}.json |
| c3_direction_r2_spectrum | method details | scripts/make_paper_section42_figures.py | eval_results/issue_779/plot3_redesign/plot3_redesign.json |
| c3_features_and_shifts | 4.2 | same | eval_results/issue_1482/plot4_redesign/plot4_decoder_direction.json, eval_results/issue_2564/section42_element_shifts.json, eval_results/issue_1482/tier_concordance.json |
| c3_refusal_swaps_by_class | a3 | same | eval_results/issue_2617/svmp_verbharm/{perpair.jsonl,summary.json} |
| c3_sae_tier_gradient | method details | same | eval_results/issue_1482/plot4_redesign/plot4_decoder_direction.json |
| c3_qualitative_discrimination | a4 | scripts/issue1901_qualitative_retrieval_failures.py | (read the script) |
| c1_posttraining_dynamics | 4.3 | scripts/section43_posttraining_onpolicy_figure.py (also Overleaf figures/paper/source/ + .render.py) | (read the script) |
| c4_shared_speakers | 4.4 | scripts/paper_fig_shared_speakers.py | eval_results/issue_2054/{section44_k5/k5_results.json, assistant_story_k5/results.json, manuscript_story_transfer/turns_source.json, shared_seven/results.json}; NOT on main, on branch codex/issue2054-k5-stage-transfer-20260917 (use `git show <branch>:<path>`) |
| c1_rollout_count | a12 | scripts/issue1901_k_rollout_ablation.py | inputs were in a Codex worktree `k-rollout-ablation-*`; search `~/.codex/worktrees/` and EPS branches |
| c1_cot_capability | 4.5 | scripts/paper_fig_cot_capability.py in /mnt/eps-data/thomasjiralerspong/wt-cot-capability-merged-20260911 | (read the script) |
| c1_k5_length_control | method details | scripts/issue1902_k5_length_control.py in /mnt/eps-data/thomasjiralerspong/wt-1902-k5 | (read the script) |
| c1_offpolicy_origin | a5 | scripts/section4_offpolicy_figure.py in ~/.codex/worktrees/825-matched-turn-transfer-20260915 | eval_results/issue_823/{ridge_r2_by_arm.json, identity_baseline.json, inconsistent_origin_ladder/ladder_analysis_summary.json, inconsistent_origin_ladder/ladder_baselines.json} (on main) |
| c4_speaker_transfer_full | method details | Overleaf figures/paper/source/c4_speaker_transfer_full.py + c4_speaker_transfer_full.data.json + .render.py | the .data.json |
| c3_ctxans_geometry | a4 | scripts/issue1901_ctxans_geometry_figure.py (on main) | eval_results/issue_1901/ctxans_geometry/summary.json |
| c3_ctxans_projection | a4 | scripts/issue1901_ctxans_projection_figure.py (on main) | eval_results/issue_1901/ctxans_projection/{summary.json,coords.npz} |
| c1_distractor_scaling | a4 | unknown: not on main, no commit mentions the stem. Search ~/.codex/worktrees/*/scripts, ~/wt-*/scripts, /mnt/eps-data/thomasjiralerspong/wt-*/scripts, EPS `.claude/worktrees/*/scripts`, and the Overleaf clone's git history for figures/paper/c1_distractor_scaling.pdf (`git -C ~/overleaf-6a59c927 log --format='%h %cs %s' -- figures/paper/c1_distractor_scaling.pdf`) to date it, then the EPS commits of that day | find |
| c4_turn_transfer_full | method details | unknown (sidecar empty); same search; likely a sibling of c4_speaker_transfer_full (Overleaf figures/paper/source/, scripts/issue2054_story_transfer_appendix.py in ~/.codex/worktrees/appendix-terminology-20260923) | find |
| c5_behavior_transfer_original_combined | 4.6 | unknown; same search; EPS issue 2054/1739 behavior-transfer scripts (`git grep -l behavior_transfer scripts/`) | find |
| c5_preimage_qualitative_top100 | method details | unknown; same search; `git grep -l preimage scripts/` | find |
| fig1_schematic | intro | docs/paper_context_answer_map/fig1_schematic.tex (TikZ) | none; copy the .tex |

## Workstreams

- WS1a (worktree `ws1a`): package port + make_paper_figure2, crosslayer_grid, rollout_count, posttraining.
- WS1b (worktree `ws1b`): section42 figures, qualitative_discrimination, shared_speakers.
- WS2 (worktree `ws2`): the off-main and unknown-generator figures.
- WS3 (worktree `ws3`): tier 2 pipeline from the supplement + notices.
- WS4 (orchestrator, after merge): registry, make_all, tests, README, CI, review round, publish.
