#!/usr/bin/env python
"""Block 10: the full baseline ladder, and the two controls the story now needs.

Where this comes from. The activation-patching baseline showed the lens direction
carries only ~10% of the variance of the true counterfactual activation difference,
yet that 10%, rescaled to the full norm, flips the answer as often as the whole
difference does. So the published swap is not aiming in the wrong direction; it is
moving too little, and (across a band) alternating. Two questions follow.

1. Does the Jacobian transport matter at all, or would any two-dimensional
   coordinate system built from unembedding columns do? The logit lens is the
   identity transport, so a clamp in logit-lens coordinates is the naive control
   the clamp result has never faced.

2. If the swap heuristic (exchange the two coordinates within one run) is what
   under-delivers, then clamping the lens coordinates to the values they take in
   the genuine counterfactual run should close the gap. That is activation
   patching restricted to the two-dimensional lens subspace, and it is the direct
   test of "is the lens subspace sufficient".

Also sweeps clamp strength, since the clamp is idempotent and therefore stable
above alpha = 1 where the involution is not.
"""
from __future__ import annotations

import json
import sys
import time

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks, logit_lens_vectors
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

CLAMP_SCALES = [0.5, 1.0, 1.5, 2.0, 3.0]


def scaled_clamp_hooks(model, basis, layers, clean_coords, scale, stats=None):
    """Clamp to the clean coordinates pushed `scale` of the way past the exchange."""
    targets = {}
    for layer in layers:
        c = clean_coords[layer]
        exchanged = c[..., [1, 0]]
        targets[layer] = c + scale * (exchanged - c)
    hooks = []
    for layer in layers:
        hooks += clamp_hooks(model, basis[layer], [layer], {layer: targets[layer]}, exchange=False, stats=stats)
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block10_ladder")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    band = cfg.BANDS[cfg.PRIMARY_BAND]
    names = {layer: _resid_post_hook_name(layer) for layer in band}
    wanted = set(names.values())

    # ---- 1 & 3: logit-lens clamp and clamp strength, over all eligible items ----
    handle = (out_dir / "ladder.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind in ("J", "R", "logit"):
            bases, coords = {}, {}
            for layer in band:
                bases[layer] = (
                    logit_lens_vectors(model, ids) if kind == "logit" else lenses[kind].lens_vectors(model, ids, layer)
                )
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            for scale in CLAMP_SCALES:
                stats = EnergyStats()
                hooks = scaled_clamp_hooks(model, bases, band, coords, scale, stats)
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "category": r.item.category,
                    "lens": kind, "arm": "clamp", "scale": scale,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
            stats = EnergyStats()
            hooks = []
            for layer in band:
                hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats)
            logp = final_logprobs(model, tokens, hooks)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "category": r.item.category,
                "lens": kind, "arm": "involution", "scale": 1.0,
                "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                "delta_logp_answer": float(logp[a] - clean[a]),
                "delta_logp_swap_answer": float(logp[s] - clean[s]),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                "top1_str": model.tokenizer.decode([top1]),
                "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
            }) + "\n")
        if n % 10 == 0:
            print(f"[ladder] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()

    # ---- 2: clamp the lens coordinates to the genuine counterfactual run's values ----
    pairs = json.loads((RESULTS_DIR / "block01" / "counterfactual_pairs.json").read_text(encoding="utf-8"))
    handle = (out_dir / "subspace_patch.jsonl").open("w", encoding="utf-8")
    for n, (i, j) in enumerate(pairs):
        ri, rj = resolved[i], resolved[j]
        ti = model.to_tokens(ri.prompt, prepend_bos=False)
        tj = model.to_tokens(rj.prompt, prepend_bos=False)
        if ti.shape != tj.shape:
            continue
        clean = final_logprobs(model, ti)
        a, s = ri.answer.first_id, ri.swap_answer.first_id
        ids = [ri.src.first_id, ri.tgt.first_id]
        _, ci = model.run_with_cache(ti, names_filter=lambda x: x in wanted)
        _, cj = model.run_with_cache(tj, names_filter=lambda x: x in wanted)
        for kind in ("J", "R", "logit"):
            bases, self_coords, cf_coords = {}, {}, {}
            for layer in band:
                bases[layer] = (
                    logit_lens_vectors(model, ids) if kind == "logit" else lenses[kind].lens_vectors(model, ids, layer)
                )
                pinv = torch.linalg.pinv(bases[layer].T.float())
                self_coords[layer] = ci[names[layer]].float() @ pinv.T
                cf_coords[layer] = cj[names[layer]].float() @ pinv.T
            arms = {
                "clamp_exchange": {layer: self_coords[layer] for layer in band},
                "clamp_counterfactual": {layer: cf_coords[layer] for layer in band},
            }
            for arm, target in arms.items():
                stats = EnergyStats()
                hooks = []
                for layer in band:
                    hooks += clamp_hooks(model, bases[layer], [layer], {layer: target[layer]},
                                         exchange=(arm == "clamp_exchange"), stats=stats)
                logp = final_logprobs(model, ti, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": i, "partner": j, "name": ri.item.name, "lens": kind, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
        print(f"[subspace] pair {n + 1}/{len(pairs)}", flush=True)
    handle.close()
    print("[block10] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
