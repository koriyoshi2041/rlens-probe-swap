#!/usr/bin/env python
"""Block 15: run the clamp intervention with our own lenses across the corpus-size ladder."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks
from rlens.model import load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lenses", required=True, help="comma-separated name=path pairs")
    parser.add_argument("--out", default="block15_fitted_eval")
    args = parser.parse_args()
    from transformer_lens.tools.analysis import JacobianLens

    out_dir = results_dir(args.out)
    model = load_model()
    lenses = {}
    for spec in args.lenses.split(","):
        name, path = spec.split("=", 1)
        lenses[name] = JacobianLens.load(path)
        print(f"[eval] {name}: n_prompts={lenses[name].n_prompts} layers={lenses[name].source_layers[0]}..{lenses[name].source_layers[-1]}")
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "fitted_eval.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for name, lens in lenses.items():
            bases, coords = {}, {}
            ok = True
            for layer in BAND:
                if layer not in lens.jacobians:
                    ok = False
                    break
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            if not ok:
                continue
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
                    "lens": name, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "delta_logp_answer": float(logp[a] - clean[a]),
                    "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
        if n % 15 == 0:
            print(f"[eval] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[eval] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
