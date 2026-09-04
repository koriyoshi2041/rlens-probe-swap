#!/usr/bin/env python
"""Block 09: put the clamp through every control the involution swap already faced.

The clamp arm flipped 41% of items where the published involution flipped 10%,
with a fifth of the perturbation. Before that number can be believed it has to
survive the same three questions:

  ortho_rescaled -- with the update's first-order push on the two answer logits
                    projected out and the norm restored. Half of the involution's
                    effect was that direct push.
  shuffled       -- clamping toward another item's entity pair, so the geometry
                    is the same and only the semantics are wrong.
  answer_pair    -- clamping the answer pair instead of the entity pair, which is
                    Neel's own diagnostic and a positive control for "pure direct
                    push": under orthogonalisation it should go to zero.
"""
from __future__ import annotations

import json
import random
import sys
import time

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block09_clamp_controls")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    rng = random.Random(0)
    order = list(eligible)
    while True:
        perm = order[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(order, perm)):
            break
    shuffle = dict(zip(order, perm))
    handle = (out_dir / "clamp_controls.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
        other = resolved[shuffle[index]]
        for band_name, band in cfg.BANDS.items():
            names = {layer: _resid_post_hook_name(layer) for layer in band}
            wanted = set(names.values())
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            for kind, lens in lenses.items():
                pair_ids = {
                    "entity": [r.src.first_id, r.tgt.first_id],
                    "answer_pair": [a, s],
                    "shuffled": [other.src.first_id, other.tgt.first_id],
                }
                for pair_name, ids in pair_ids.items():
                    if ids[0] == ids[1]:
                        continue
                    bases, coords, units = {}, {}, {}
                    for layer in band:
                        bases[layer] = lens.lens_vectors(model, ids, layer)
                        pinv = torch.linalg.pinv(bases[layer].T.float())
                        coords[layer] = cache[names[layer]].float() @ pinv.T
                        matrix = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                        units[layer] = matrix.T @ contrast
                    for arm, ortho in (("full", False), ("ortho_rescaled", True)):
                        stats = EnergyStats()
                        hooks = []
                        for layer in band:
                            hooks += clamp_hooks(
                                model, bases[layer], [layer], {layer: coords[layer]}, stats=stats,
                                orthogonalize_to=units[layer] if ortho else None,
                                rescale_after_orthogonalize=ortho,
                            )
                        logp = final_logprobs(model, tokens, hooks)
                        top1 = int(logp.argmax().item())
                        handle.write(json.dumps({
                            "index": index, "name": r.item.name, "category": r.item.category,
                            "band": band_name, "lens": kind, "pair": pair_name, "arm": arm,
                            "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                            "delta_logp_answer": float(logp[a] - clean[a]),
                            "delta_logp_swap_answer": float(logp[s] - clean[s]),
                            "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                            "top1_str": model.tokenizer.decode([top1]),
                            "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                        }) + "\n")
        if n % 10 == 0:
            print(f"[clamp_controls] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    print("[clamp_controls] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
