# rlens-probe-swap

Code and raw results for a study of Jacobian-lens (J-lens) and R-lens coordinate interventions on
two-hop factual prompts (the `probe-swap` items) in Qwen3.5-9B, Qwen3.5-4B and Qwen3-4B.

- `code/rlens/` — model/lens loading, the published coordinate swap and an idempotent clamp variant,
  donor pastes, subspace projectors, readout, cluster bootstrap.
- `code/q*.py` — one GPU driver per experiment block (docstrings state question, arms and predictions).
- `code/a*.py` — CPU analyses that print the tables from `results/`; `code/f01_figures.py` regenerates `figures/`.
- `code/tests/` — tests; `code/audit_recompute/` — independent recomputation scripts. These audits were run by a separate Claude Code sub-agent that read only the raw result files; the author's own checks are described in the write-up.
- `code/data/` — repaired single-hop templates and 32 LLM-written non-geographic items.
- `results/` — raw per-item outputs of every block; `figures/` — all figures; `logs/` — run logs;
  `docs/provenance/` — pinned asset revisions, hashes and environment versions.

Run CPU analyses with `python3 code/aNN_*.py results` and figures with `python3 code/f01_figures.py`
(set `RLENS_RESULTS` if `results/` is elsewhere). GPU drivers need `MATS_ROOT` pointing to a folder with
`assets/` (models, lenses), `src/jacobian-lens/` (pinned commit, provides `probe-swap.json`) and `work/results/`.
Third-party material and licences: `NOTICE.md`.

Notes for readers of `results/`: the seven 50 MB fitted-vector files (`vectors_*.pt` under `block17_fitladder/` and `block34_refit_pile/`) are git-ignored and can be regenerated with `q22_fit_ladder.py` / `q37_refit_pile.py`; every JSONL result is included. `code/remote_chains/` are the batch scripts that ran on the GPU box (paths default to that box; set `MATS_ROOT`). `logs/` are the raw run logs.
