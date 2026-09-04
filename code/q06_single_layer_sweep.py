#!/usr/bin/env python
"""A proper energy-matched J-vs-R comparison.

Why this run exists: across a band the swap compounds (the antisymmetric eigenvalue
is 1-2*alpha), so ||dh|| is NOT proportional to alpha and a slope fitted from two
band-level alphas mixes two regimes. At a SINGLE layer the map is applied once, so
delta = alpha * (c_t - c_s)(v_s - v_t) is exactly linear in alpha and a dose-response
curve is well defined. We can then compare J and R at matched ||dh|| by interpolation
instead of at matched alpha.
"""
from __future__ import annotations

import json
import sys

import torch

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

LAYERS = [5, 6, 9, 12, 16]
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block03_sweep")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "sweep.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        for layer in LAYERS:
            bases = {k: lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer) for k, lens in lenses.items()}
            for kind in lenses:
                for alpha in ALPHAS:
                    stats = EnergyStats()
                    hooks = coordinate_map_hooks(model, bases[kind], [layer], mode="swap", alpha=alpha, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "layer": layer, "lens": kind, "alpha": alpha,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "kl": kl_divergence(clean, logp),
                        "dh_norm": float(stats.total_dh2 ** 0.5), "total_dh2": stats.total_dh2,
                    }) + "\n")
        if n % 10 == 0:
            print(f"[sweep] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[sweep] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
