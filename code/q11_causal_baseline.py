#!/usr/bin/env python
"""Block 07: calibrate the lens direction against what the model actually uses.

The gap this fills. Everything so far compared two lenses to each other. Neither
was ever compared to the ground truth: the activation difference the model itself
produces when the bridge entity really is different. Without that baseline,
"the swap changes the answer 10% of the time" has no denominator, and the four
competing explanations for the weak behavioural effect cannot be separated:
(a) the lens direction is not the variable the model uses; (b) the edit is too
weak; (c) the second hop does not re-run where we edit; (d) the answer travels
by another route.

Two baselines, both energy-matched to the lens edit so size is never the
explanation.

PATCH (n=14, gold standard). The item set contains natural minimal pairs: item 23
is "the capital of the country where Lyon is located" (France -> Paris) and its
declared counterfactual, Italy -> Rome, is exactly item 24's prompt with Naples.
For such a pair we take the model's own residual difference at one layer and
apply it to the original run. Arms:
  patch_full  -- add the whole difference
  patch_par   -- add only its component along the lens swap direction
  patch_perp  -- add only the rest
  patch_par_matched / patch_perp_matched -- the same two rescaled to the full
                  difference's norm, so each is judged at equal strength
If patch_par alone reproduces patch_full, the lens direction carries the causal
variable. If only patch_perp does, the lens direction is nearly irrelevant.

DIFFMEAN (n=59). For every item we estimate the entity direction empirically:
run neutral carrier prompts containing the bridge entity and its replacement,
take the mean residual at the entity's last token, and difference them. This is
the standard difference-of-means probe direction. We then push along it, rescaled
to the exact norm of the lens edit, and compare.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

LAYERS = [6, 8, 10, 12, 14, 16, 20]
CARRIERS = ["Fact: {}", "The topic is {}", "Here is {}", "Regarding {}", "A: {}", "It says {}"]


def add_vector_hooks(model, vector: torch.Tensor, layers, stats=None):
    """Add one fixed residual-stream vector at every position of the given layers."""
    hooks = []
    for layer in layers:
        def transform(selected, vector=vector, layer=layer):
            v = vector.to(selected.device).float()
            h = selected.float()
            delta = v.expand_as(h)
            if stats is not None:
                stats.record(layer, h, delta)
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def add_tensor_hooks(model, delta_by_layer: Dict[int, torch.Tensor], stats=None):
    """Add a per-position residual difference tensor [1, pos, d] at each layer."""
    hooks = []
    for layer, delta in delta_by_layer.items():
        def transform(selected, delta=delta, layer=layer):
            d = delta.to(selected.device).float()
            h = selected.float()
            if stats is not None:
                stats.record(layer, h, d)
            return h + d

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def entity_directions(model, lenses, entities: List[str], layers) -> Dict[str, Dict[int, torch.Tensor]]:
    """Mean residual at the entity's last token, averaged over neutral carriers."""
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    names = {layer: _resid_post_hook_name(layer) for layer in layers}
    wanted = set(names.values())
    for entity in entities:
        acc = {layer: [] for layer in layers}
        for carrier in CARRIERS:
            text = carrier.format(entity.strip())
            tokens = model.to_tokens(text, prepend_bos=False)
            prefix = model.to_tokens(carrier.split("{}")[0], prepend_bos=False).shape[1]
            entity_len = len(model.tokenizer.encode(" " + entity.strip() if carrier.split("{}")[0].endswith(" ") else entity.strip(), add_special_tokens=False))
            pos = min(prefix + entity_len - 1, tokens.shape[1] - 1)
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in wanted)
            for layer in layers:
                acc[layer].append(cache[names[layer]][0, pos, :].float())
        out[entity] = {layer: torch.stack(acc[layer]).mean(0) for layer in layers}
    return out


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block07_causal")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    pairs = json.loads((RESULTS_DIR / "block01" / "counterfactual_pairs.json").read_text(encoding="utf-8"))
    names = {layer: _resid_post_hook_name(layer) for layer in LAYERS}
    wanted = set(names.values())

    # ---------------- PATCH ----------------
    handle = (out_dir / "patch.jsonl").open("w", encoding="utf-8")
    for n, (i, j) in enumerate(pairs):
        ri, rj = resolved[i], resolved[j]
        ti = model.to_tokens(ri.prompt, prepend_bos=False)
        tj = model.to_tokens(rj.prompt, prepend_bos=False)
        if ti.shape != tj.shape:
            continue
        clean = final_logprobs(model, ti)
        a, s = ri.answer.first_id, ri.swap_answer.first_id
        _, cache_i = model.run_with_cache(ti, names_filter=lambda x: x in wanted)
        _, cache_j = model.run_with_cache(tj, names_filter=lambda x: x in wanted)
        for layer in LAYERS:
            diff = (cache_j[names[layer]].float() - cache_i[names[layer]].float())  # [1, pos, d]
            for kind, lens in lenses.items():
                basis = lens.lens_vectors(model, [ri.src.first_id, ri.tgt.first_id], layer)
                unit = (basis[1] - basis[0])
                unit = unit / unit.norm()
                par = (diff @ unit).unsqueeze(-1) * unit
                perp = diff - par
                full_norm = diff.norm(dim=-1, keepdim=True)
                arms = {
                    "patch_full": diff,
                    "patch_par": par,
                    "patch_perp": perp,
                    "patch_par_matched": par * (full_norm / par.norm(dim=-1, keepdim=True).clamp_min(1e-9)),
                    "patch_perp_matched": perp * (full_norm / perp.norm(dim=-1, keepdim=True).clamp_min(1e-9)),
                }
                for arm, delta in arms.items():
                    stats = EnergyStats()
                    logp = final_logprobs(model, ti, add_tensor_hooks(model, {layer: delta}, stats))
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": i, "partner": j, "name": ri.item.name, "category": ri.item.category,
                        "layer": layer, "lens": kind, "arm": arm,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                        "kl": kl_divergence(clean, logp), "dh_norm": float(stats.total_dh2 ** 0.5),
                        "frac_var_explained_by_lens_dir": float((par.norm() ** 2 / diff.norm() ** 2).item()),
                    }) + "\n")
                # the lens edit itself, for a same-run comparison
                stats = EnergyStats()
                logp = final_logprobs(model, ti, coordinate_map_hooks(model, basis, [layer], mode="swap", alpha=1.0, stats=stats))
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": i, "partner": j, "name": ri.item.name, "category": ri.item.category,
                    "layer": layer, "lens": kind, "arm": "lens_swap",
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "dh_norm": float(stats.total_dh2 ** 0.5),
                    "frac_var_explained_by_lens_dir": None,
                }) + "\n")
        print(f"[patch] pair {n + 1}/{len(pairs)} ({ri.item.name} <- {rj.item.name})", flush=True)
    handle.close()

    # ---------------- DIFFMEAN ----------------
    entities = sorted({r.item.intermediate for r in resolved.values()} | {r.item.swap_to for r in resolved.values()})
    print(f"[diffmean] estimating directions for {len(entities)} entities over {len(CARRIERS)} carriers", flush=True)
    t0 = time.time()
    dirs = entity_directions(model, lenses, entities, LAYERS)
    print(f"[diffmean] directions ready in {time.time() - t0:.0f}s", flush=True)
    handle = (out_dir / "diffmean.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        for layer in LAYERS:
            emp = dirs[r.item.swap_to][layer] - dirs[r.item.intermediate][layer]
            for kind, lens in lenses.items():
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                lens_dir = basis[1] - basis[0]
                stats = EnergyStats()
                lp_lens = final_logprobs(model, tokens, coordinate_map_hooks(model, basis, [layer], mode="swap", alpha=1.0, stats=stats))
                lens_norm_per_pos = (stats.total_dh2 / max(tokens.shape[1], 1)) ** 0.5
                scaled = emp / emp.norm() * lens_norm_per_pos
                stats2 = EnergyStats()
                lp_emp = final_logprobs(model, tokens, add_vector_hooks(model, scaled, [layer], stats2))
                for arm, lp, st in (("lens_swap", lp_lens, stats), ("diffmean_matched", lp_emp, stats2)):
                    top1 = int(lp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "layer": layer, "lens": kind, "arm": arm,
                        "delta_margin": float((lp[s] - lp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(lp[a] - clean[a]),
                        "delta_logp_swap_answer": float(lp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                        "kl": kl_divergence(clean, lp), "dh_norm": float(st.total_dh2 ** 0.5),
                        "cos_lens_diffmean": float(torch.nn.functional.cosine_similarity(lens_dir[None], emp[None]).item()),
                    }) + "\n")
        if n % 15 == 0:
            print(f"[diffmean] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block07] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
