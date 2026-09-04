#!/usr/bin/env python
"""Block 19: does corpus size change what the intervention does?

Runs the clamp and the published involution with our own fitted vectors at each
rung of the corpus ladder, and compares every rung with the published n=25
artifact both geometrically (cosine per layer) and behaviourally (flip rate).

Only the J-lens can be refitted here: R-lens uses relevance propagation, which the
reference package does not implement, so the R artifact stays as published and the
ladder speaks to J alone.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.fit_vectors import FittedVectorLens
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block19_ladder_eval")
    fit_dir = RESULTS_DIR / "block17_fitladder"
    model = load_model()
    published = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    rungs = {}
    for path in sorted(fit_dir.glob("vectors_n*.pt"), key=lambda p: int(p.stem.split("n")[-1])):
        n = int(path.stem.split("n")[-1])
        rungs[f"ours_n{n}"] = FittedVectorLens.load(str(path))
    print(f"[eval] rungs: {list(rungs)}", flush=True)

    # geometry: how far is each rung from the published artifact, and from the top rung?
    geo = []
    top = rungs[max(rungs, key=lambda k: rungs[k].n_prompts)] if rungs else None
    for name, lens in rungs.items():
        for layer in BAND:
            rows = []
            for index in eligible:
                r = resolved[index]
                ours = lens.lens_vectors(model, r.tracked_ids[:2], layer)
                pub = published["J"].lens_vectors(model, r.tracked_ids[:2], layer)
                best = top.lens_vectors(model, r.tracked_ids[:2], layer)
                rows.append({
                    "cos_to_published": float(torch.nn.functional.cosine_similarity(ours, pub, dim=-1).mean().item()),
                    "cos_to_top_rung": float(torch.nn.functional.cosine_similarity(ours, best, dim=-1).mean().item()),
                    "norm_ratio_to_published": float((ours.norm(dim=-1) / pub.norm(dim=-1)).mean().item()),
                })
            geo.append({"rung": name, "n_prompts": lens.n_prompts, "layer": layer,
                        **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}})
    (out_dir / "geometry.json").write_text(json.dumps(geo, indent=1), encoding="utf-8")
    print("[eval] geometry written", flush=True)

    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "ladder_eval.jsonl").open("w", encoding="utf-8")
    all_lenses = dict(rungs)
    all_lenses["published_J"] = published["J"]
    all_lenses["published_R"] = published["R"]
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = r.tracked_ids[:2]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for name, lens in all_lenses.items():
            bases, coords = {}, {}
            for layer in BAND:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            for arm in ("clamp", "involution"):
                stats = EnergyStats()
                hooks = []
                for layer in BAND:
                    if arm == "clamp":
                        hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
                    else:
                        hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=1.0, stats=stats)
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "category": r.item.category,
                    "lens": name, "n_prompts": getattr(lens, "n_prompts", None), "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
        if n % 15 == 0:
            print(f"[eval] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block19] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
