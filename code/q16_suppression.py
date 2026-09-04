#!/usr/bin/env python
"""Block 12: is suppressing the original entity what makes the rewrite land?

The readout after a clamp shows the two groups differ mostly on one thing. Where
the answer flips, the bridge entity has been pushed to rank ~880; where it does
not, the entity is still at rank ~105 while the replacement sits at rank 5. Both
entities are present. A regression cannot settle this because suppression and the
entity-answer geometry are collinear (rho = 0.55) across items.

But the interventions already separate them within an item, holding geometry
exactly fixed. Writing (install) and erasing (remove) are the two idempotent
halves of the clamp:

    install : (c_s, c_t) -> (c_s, c_s)   raise the replacement, leave the original
    remove  : (c_s, c_t) -> (c_t, c_t)   lower the original, leave the replacement
    clamp   : (c_s, c_t) -> (c_t, c_s)   both

This run completes that 2x2 at the band, and then pushes suppression past what the
clamp does: clamp plus a full projection of the original entity's direction out of
the residual stream. If suppression is the binding constraint, the extra erasure
should raise the flip rate, and it should do so on the items where the entity
survived. A norm-matched random projection is the control for "any extra
perturbation helps".
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
from rlens.interventions import (
    EnergyStats,
    ablation_hooks_with_stats,
    clamp_hooks,
    coordinate_map_hooks,
    gram_matched_random_basis,
)
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block12_suppression")
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
    handle = (out_dir / "suppression.jsonl").open("w", encoding="utf-8")
    read_handle = (out_dir / "suppression_readout.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, coords = {}, {}
            for layer in band:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            arms = {}
            for name in ("install", "remove"):
                arms[name] = lambda layer, name=name: coordinate_map_hooks(
                    model, bases[layer], [layer], mode=name, alpha=1.0, stats=arms_stats[name]
                )
            arms_stats = {}
            built = {}
            for name in ("install", "remove", "clamp", "clamp_plus_ablate", "clamp_plus_random"):
                stats = EnergyStats()
                arms_stats[name] = stats
                hooks = []
                for layer in band:
                    if name in ("install", "remove"):
                        hooks += coordinate_map_hooks(model, bases[layer], [layer], mode=name, alpha=1.0, stats=stats)
                    else:
                        hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
                        if name == "clamp_plus_ablate":
                            hooks += ablation_hooks_with_stats(model, bases[layer][0:1], [layer], stats=stats)
                        elif name == "clamp_plus_random":
                            rand = gram_matched_random_basis(bases[layer], seed=1000 + layer)
                            hooks += ablation_hooks_with_stats(model, rand[0:1], [layer], stats=stats)
                built[name] = hooks
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "category": r.item.category,
                    "lens": kind, "arm": name,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "top1_str": model.tokenizer.decode([top1]),
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
            if kind == "J":
                reader = "R"
                for name in ("clamp", "clamp_plus_ablate", "clamp_plus_random", "remove", "install"):
                    with model.hooks(fwd_hooks=built[name]):
                        ranks, _ = rank_readout(model, lenses[reader], tokens, r.tracked_ids, band)
                    best = ranks[:-1].min(dim=1).values.numpy()
                    read_handle.write(json.dumps({
                        "index": index, "name": r.item.name, "arm": name, "reader": reader,
                        "best_intermediate": best[:, 0].tolist(), "best_swap_to": best[:, 1].tolist(),
                    }) + "\n")
        if n % 10 == 0:
            print(f"[suppression] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    read_handle.close()
    print("[block12] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
