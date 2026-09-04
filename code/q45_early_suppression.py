#!/usr/bin/env python
"""Block 40: is R's early-band advantage CAUSED by stronger suppression of the source entity?

Block 11 found that after an early-band (L3-8) clamp, R pushes the original entity far
further down than J (cross-read rank 1186 vs 278) while installing the target slightly
worse, and block 13 found R's coordinate gap is larger at L3-5. Both are correlational.
Here the J clamp is given the extra suppression directly: after clamping, the unit source
direction v_s (J's) is projected out at the same layers and positions.

  J_clamp                reference
  J_clamp+ablate_src     J clamp, then project out v_s (J-lens) at every band layer
  J_clamp+ablate_rand    control: project out a random unit direction (seed per item)
  J_clamp+ablate_tgt     control: project out v_t (should HURT)
  R_clamp                reference
  R_clamp+ablate_src     does extra suppression still help R?

If J_clamp+ablate_src reaches R_clamp's flip rate and the random control does not,
suppression explains R's early advantage causally.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, ablation_hooks_with_stats, clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(3, 9))


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block40_early_suppression")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "early_suppression.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        gen = torch.Generator(device="cpu").manual_seed(index)
        for kind in ("J", "R"):
            lens = lenses[kind]
            bases, clamp = {}, []
            for layer in BAND:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords = cache[names[layer]].float() @ pinv.T
                clamp += clamp_hooks(model, bases[layer], [layer], {layer: coords})
            arms = {"clamp": []}
            arms["clamp+ablate_src"] = sum((ablation_hooks_with_stats(model, bases[l][0:1], [l]) for l in BAND), [])
            arms["clamp+ablate_tgt"] = sum((ablation_hooks_with_stats(model, bases[l][1:2], [l]) for l in BAND), [])
            rand = {l: torch.randn(1, model.cfg.d_model, generator=gen).to(bases[l].device) for l in BAND}
            arms["clamp+ablate_rand"] = sum((ablation_hooks_with_stats(model, rand[l], [l]) for l in BAND), [])
            arms["ablate_src_only"] = arms["clamp+ablate_src"]
            for arm, extra in arms.items():
                hooks = (clamp if arm != "ablate_src_only" else []) + extra
                stats = EnergyStats()
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "delta_logp_answer": float(logp[a] - clean[a]), "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q45] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block40] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
