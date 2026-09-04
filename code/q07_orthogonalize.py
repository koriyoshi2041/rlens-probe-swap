#!/usr/bin/env python
"""The decisive control: strip the update's direct push on the answer logits.

Motivation. Part A of the exploration found that the intervention direction
(v_t - v_s), read through the lens, already promotes `swap_answer` in the output
vocabulary for a third of items, and that those items carry almost all of the
behavioural effect (Spearman rho ~= -0.5 to -0.66). That is Neel's "direct answer
direction" alternative explanation to the rewrite story.

Test. At each layer, the direction that most directly moves the answer contrast is
u_l = J_l^T (W_U[:, swap_answer] - W_U[:, answer]) -- the lens vector of the answer
contrast. Projecting the swap update orthogonal to u_l removes its first-order direct
effect on those two logits while leaving everything else. Two arms:

  ortho        -- project out u_l (update becomes slightly smaller)
  ortho_rescaled -- project out u_l, then restore the original norm (energy-matched)

If the effect survives, the rewrite is mediated by the model's computation. If it
collapses, the "targeted rewrite" was largely a direct logit push.
"""
from __future__ import annotations

import json
import sys

import torch

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

SINGLE_LAYERS = [6, 12]


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block03_ortho")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "ortho.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        contrast = (model.W_U[:, s].float() - model.W_U[:, a].float())
        groups = {name: layers for name, layers in cfg.BANDS.items()}
        groups.update({f"L{layer}": [layer] for layer in SINGLE_LAYERS})
        for group_name, layers in groups.items():
            for kind, lens in lenses.items():
                bases, units, cosines = {}, {}, {}
                for layer in layers:
                    bases[layer] = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    matrix = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                    units[layer] = matrix.T @ contrast
                    direction = bases[layer][1] - bases[layer][0]
                    cosines[layer] = float(torch.nn.functional.cosine_similarity(direction[None], units[layer][None]).item())
                for arm, ortho, rescale in (("full", False, False), ("ortho", True, False), ("ortho_rescaled", True, True)):
                    stats = EnergyStats()
                    hooks = []
                    for layer in layers:
                        hooks += coordinate_map_hooks(
                            model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats,
                            orthogonalize_to=units[layer] if ortho else None,
                            rescale_after_orthogonalize=rescale,
                        )
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "group": group_name, "lens": kind, "arm": arm,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "kl": kl_divergence(clean, logp),
                        "total_dh2": stats.total_dh2,
                        "median_cos_direction_contrast": float(sorted(cosines.values())[len(cosines) // 2]),
                    }) + "\n")
        if n % 10 == 0:
            print(f"[ortho] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[ortho] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
