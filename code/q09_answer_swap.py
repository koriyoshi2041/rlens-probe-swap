#!/usr/bin/env python
"""Block 05: adjudicate the paper vs the public replication.

Source of the question. Neel Nanda's review of the Global Workspace paper
(lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S) argues the "direct answer direction"
explanation is unlikely *for the paper's setting*, citing its figure 15: in
workspace layers, swapping the intermediate is significantly more effective than
swapping the final answer. He then reports the opposite in his own Qwen
replication -- "swapping the answer turned out to strictly dominate" -- and
attributes it to the dataset being too easy, with linearly related pairs like
France and Paris.

We are on Qwen3.5-9B with exactly that kind of item set, so we can run his own
diagnostic layer by layer, for both lenses, alongside the orthogonalisation
control that measures the same confound directly.

Arms per layer: swap intermediate->swap_to, swap answer->swap_answer, and the
answer swap with the direct component projected out (the same control we applied
to the intermediate swap).

Also records per-item "linear relatedness": cosines among the unembedding columns
and the layer's lens vectors, so the dataset-easiness claim becomes measurable.
"""
from __future__ import annotations

import json
import sys
import time

import torch

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

LAYERS = [3, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a[None].float(), b[None].float()).item())


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block05_answer_swap")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    geom_handle = (out_dir / "geometry.jsonl").open("w", encoding="utf-8")
    handle = (out_dir / "answer_swap.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        src, tgt = r.src.first_id, r.tgt.first_id
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()

        # dataset-easiness geometry, in the unembedding basis (lens independent)
        wu = model.W_U.float()
        geom_handle.write(json.dumps({
            "index": index, "name": r.item.name, "category": r.item.category,
            "cos_intermediate_answer": cos(wu[:, src], wu[:, a]),
            "cos_swapto_swapanswer": cos(wu[:, tgt], wu[:, s]),
            "cos_intermediate_swapto": cos(wu[:, src], wu[:, tgt]),
            "cos_answer_swapanswer": cos(wu[:, a], wu[:, s]),
            "cos_entitydiff_answerdiff": cos(wu[:, tgt] - wu[:, src], wu[:, s] - wu[:, a]),
        }) + "\n")

        for layer in LAYERS:
            for kind, lens in lenses.items():
                matrix = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                unit = matrix.T @ contrast
                pairs = {
                    "intermediate": lens.lens_vectors(model, [src, tgt], layer),
                    "answer": lens.lens_vectors(model, [a, s], layer),
                }
                for pair_name, basis in pairs.items():
                    direction = basis[1] - basis[0]
                    for arm, ortho in (("full", False), ("ortho_rescaled", True)):
                        stats = EnergyStats()
                        hooks = coordinate_map_hooks(
                            model, basis, [layer], mode="swap", alpha=1.0, stats=stats,
                            orthogonalize_to=unit if ortho else None,
                            rescale_after_orthogonalize=ortho,
                        )
                        logp = final_logprobs(model, tokens, hooks)
                        top1 = int(logp.argmax().item())
                        handle.write(json.dumps({
                            "index": index, "name": r.item.name, "category": r.item.category,
                            "layer": layer, "lens": kind, "pair": pair_name, "arm": arm,
                            "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                            "delta_logp_answer": float(logp[a] - clean[a]),
                            "delta_logp_swap_answer": float(logp[s] - clean[s]),
                            "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                            "kl": kl_divergence(clean, logp),
                            "dh_norm": float(stats.total_dh2 ** 0.5),
                            "cos_direction_contrast": cos(direction, unit),
                        }) + "\n")
        if n % 10 == 0:
            print(f"[answer_swap] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    geom_handle.close()
    print("[answer_swap] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
