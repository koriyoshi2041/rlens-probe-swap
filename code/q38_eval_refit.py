#!/usr/bin/env python
"""Block 34b: evaluate the pile-10k / skip-4 refits (q37) exactly like block 19 did for the wikitext ladder."""
from __future__ import annotations

import json
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
    out_dir = results_dir("block34_refit_pile")
    model = load_model()
    published = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    lenses = {"published_J": published["J"]}
    for path in sorted(out_dir.glob("vectors_pile_n*.pt")):
        lenses[f"pile_n{path.stem.split('n')[-1]}"] = FittedVectorLens.load(str(path))
    wiki = RESULTS_DIR / "block17_fitladder" / "vectors_n25.pt"
    if wiki.exists():
        lenses["wiki_n25_skip16"] = FittedVectorLens.load(str(wiki))
    print(f"[eval] lenses: {list(lenses)}", flush=True)
    geo = []
    for name, lens in lenses.items():
        if name == "published_J":
            continue
        for layer in BAND:
            cos, ratio = [], []
            for index in eligible:
                r = resolved[index]
                ours = lens.lens_vectors(model, r.tracked_ids[:2], layer)
                pub = published["J"].lens_vectors(model, r.tracked_ids[:2], layer)
                cos.append(float(torch.nn.functional.cosine_similarity(ours, pub, dim=-1).mean().item()))
                ratio.append(float((ours.norm(dim=-1) / pub.norm(dim=-1)).mean().item()))
            geo.append({"lens": name, "layer": layer, "cos_to_published": float(np.mean(cos)), "norm_ratio_to_published": float(np.mean(ratio))})
    (out_dir / "geometry.json").write_text(json.dumps(geo, indent=1), encoding="utf-8")
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "refit_eval.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = r.tracked_ids[:2]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for name, lens in lenses.items():
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
                    "index": index, "name": r.item.name, "lens": name, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
        if n % 15 == 0:
            print(f"[eval] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block34b] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
