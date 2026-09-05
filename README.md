# rlens-probe-swap

J-lens and R-lens read a language model's intermediate thoughts out of its residual stream. J-lens can also edit them: swap the lens coordinates of one entity for another across a band of layers and, on some two-hop prompts, the answer follows. R-lens, which replaces the Jacobian with relevance propagation and reads earlier layers more faithfully, had only been tested by reading and ablation. This repository holds the code, data and per-item results of a study that ran the swap experiment on both lenses side by side, on Qwen3.5-9B, Qwen3.5-4B and Qwen3-4B, using Anthropic's published probe-swap prompts.

The study found that the published swap protocol cancels itself when applied across a band, because the released hook re-reads the coordinates from the already-edited activation at every layer; an idempotent clamp fixes this and roughly quadruples the flip rate. With the clamp, the internal representation is exchanged on almost every prompt but the answer changes on fewer than half of them, and about half of those changes are a direct push on the answer logit rather than a rewrite. Whether a rewrite is used depends on two things, whether the edited position holds the content the second hop needs and whether a transport head downstream is reading that position. R-lens does not beat J-lens in the band where the edit works. The protocol issue is reported upstream as TransformerLens [issue #1746](https://github.com/TransformerLensOrg/TransformerLens/issues/1746) with a fix in [PR #1747](https://github.com/TransformerLensOrg/TransformerLens/pull/1747).

## What is here

`code/rlens/` is the library: model and lens loading, the published coordinate swap and the idempotent clamp, donor pastes, subspace projectors, readout, and the cluster bootstrap used for every interval. Each experiment block has one GPU driver named `code/qNN_*.py`, whose docstring states the question it asks, the arms it runs and the prediction written down before running. The CPU analyses that print the tables from the raw results are `code/aNN_*.py`; `code/f01_figures.py` regenerates `figures/` and `code/figpaper/` regenerates the figures used in the write-up. Tests live in `code/tests/`. The scripts in `code/audit_recompute/` recompute the logged numbers from the raw per-item files; they were run by a separate Claude Code sub-agent that read only the result files, and the author's own checks are described in the write-up rather than here.

`results/` contains the raw per-item output of every block as JSONL, `figures/` and `figures_paper/` the figures, `logs/` the run logs, and `docs/provenance/` the pinned asset revisions, hashes and environment versions. `code/data/` holds the repaired single-hop templates and the 32 non-geographic held-out items that were drafted by Claude and screened by the author. The fitted-vector files under `block17_fitladder/` and `block34_refit_pile/` are git-ignored because of their size and can be regenerated with `q22_fit_ladder.py` and `q37_refit_pile.py`.

## Running things

The CPU analyses need only the results folder:

```bash
python3 code/a03_fact_cluster_ci.py results
python3 code/f01_figures.py
```

Set `RLENS_RESULTS` if `results/` lives elsewhere. The GPU drivers expect `MATS_ROOT` to point at a folder containing `assets/` with the models and lenses, `src/jacobian-lens/` at the pinned commit (it provides `probe-swap.json`), and `work/results/`. The batch scripts in `code/remote_chains/` are the ones that ran on the GPU machine; their paths default to that machine and follow `MATS_ROOT`.

## Sources and licence

The probe-swap prompts come from Anthropic's `jacobian-lens` repository, the matched J/R lenses from `camilablank/workspace-lenses`, and the swap hook from TransformerLens; all three are pinned by commit or revision in `NOTICE.md`, together with the licences and the papers this work responds to. The code here is released under Apache-2.0 (see `LICENSE`). Model weights and the published lenses are not redistributed.
