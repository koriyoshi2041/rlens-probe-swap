# Third-party material and provenance

This repository builds on the following work. Nothing here should be read as a reproduction of the original papers' tables; we reuse their released artefacts and counterfactual data to ask a different question.

## Data
- **probe-swap items (90 two-hop prompts)** — `data/experiments/probe-swap.json` in Anthropic's `jacobian-lens` repository, pinned commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`, **Apache License 2.0**. Not copied into this repository; the code reads it from `$MATS_ROOT/src/jacobian-lens/`. The repository's `data/experiments/README.md` defines the swap protocol as clamping a lens coordinate at every band layer.
- **Repaired single-hop templates** (`code/data/single_hop_templates.json`) and the **32 non-geographic v2 items** (`code/data/new_items_nongeo*.json`) were written for this project; the v2 items were drafted by Claude and screened by the author.
- Lens fitting corpus for the third model: `NeelNanda/pile-10k` (Hugging Face dataset), 25 documents.

## Models and lenses
- **Qwen3.5-9B**, **Qwen3.5-4B** (Alibaba/Qwen; Hugging Face revisions and SHA-256 in `docs/provenance/asset_manifest.json`; Qwen licence terms apply to the weights, which are not redistributed here).
- **Qwen3-4B** (dense attention), downloaded via `code/dl_qwen3_4b.py`; the J-lens fitted on it with the reference estimator is ours (release asset, 459 MB).
- **Published matched J/R lenses** — `camilablank/workspace-lenses`, revision `d740106d1e0f95456dc8718fba2895e9c8ffd6ef` (Qwen3.5-9B and 4B, 25 fitting prompts, source layers 0–30). Licence: as stated on the Hugging Face model card (check before redistribution; not redistributed here).

## Code
- **TransformerLens** (MIT), pinned commit `c03d51037e32ae5e34b3e23d63f63d0e6e16bd1b`, including `transformer_lens.tools.analysis.jacobian_lens` (`JacobianLens.fit/load/swap_hooks`, `_make_intervention_hook`, hook names). Our `rlens.interventions.coordinate_map_hooks` re-implements the published swap semantics for the involution arm; `clamp_hooks` is our idempotent variant.
- **jacobian-lens** (Apache-2.0) — estimator conventions and artefact format.

## Papers and posts this work responds to
- Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits Thread, 2026 (J-lens; coordinate swap `h ← h + V(σ(c) − c)`; probe-swap results 61/60/28/6%).
- Blank, Bhatia, Nanda, *R-lens: Making J-lens More Faithful on Early Layers*, Alignment Forum, 2026 (LRP rules LN/Identity/Half; no intervention experiments).
- Neel Nanda, *A review of Anthropic's Global Workspace paper*, LessWrong, 2026.

## LLM assistance
Experiment code, batch execution, first-pass analysis, figures, the research log and the audit scripts were produced with Claude (Claude Code); the author set the question, approved the pre-registered design, directed each round of checking and digging, read raw samples, and wrote the application text. The LLM-written parts of the data are identified above.
