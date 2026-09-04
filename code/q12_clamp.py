#!/usr/bin/env python
"""Block 08: clamp vs involution across the band.

The persistence readout showed the band swap alternating layer by layer -- the
bridge entity is pushed down at L8, back at rank 6 by L9, down again at L10.
That is the involution's eigenvalue -1 acting once per layer. The published
protocol calls the intervention "clamping a lens coordinate", and a clamp is
idempotent: it would hold the swapped state through the whole band. This run
compares the two, plus the idempotent install half, on all eligible items.
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
from rlens.interventions import EnergyStats, clamp_hooks, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block08_clamp")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "clamp.jsonl").open("w", encoding="utf-8")
    read_handle = (out_dir / "clamp_readout.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        for band_name, band in cfg.BANDS.items():
            names = {layer: _resid_post_hook_name(layer) for layer in band}
            wanted = set(names.values())
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            for kind, lens in lenses.items():
                bases, coords = {}, {}
                for layer in band:
                    bases[layer] = lens.lens_vectors(model, ids, layer)
                    pinv = torch.linalg.pinv(bases[layer].T.float())
                    coords[layer] = cache[names[layer]].float() @ pinv.T
                arms = {}
                for arm in ("swap", "install", "clamp"):
                    stats = EnergyStats()
                    hooks = []
                    for layer in band:
                        if arm == "clamp":
                            hooks += clamp_hooks(model, bases[layer], [layer], {layer: coords[layer]}, stats=stats)
                        else:
                            hooks += coordinate_map_hooks(model, bases[layer], [layer], mode=arm, alpha=1.0, stats=stats)
                    arms[arm] = (hooks, stats)
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "band": band_name, "lens": kind, "arm": arm,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                        "top1_str": model.tokenizer.decode([top1]),
                        "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                        "energy": stats.as_rows(),
                    }) + "\n")
                if band_name == cfg.PRIMARY_BAND:
                    reader = "R" if kind == "J" else "J"
                    for arm in ("swap", "clamp", "install"):
                        with model.hooks(fwd_hooks=arms[arm][0]):
                            ranks, _ = rank_readout(model, lenses[reader], tokens, r.tracked_ids, band)
                        best = ranks[:-1].min(dim=1).values.numpy()
                        read_handle.write(json.dumps({
                            "index": index, "name": r.item.name, "lens": kind, "reader": reader,
                            "arm": arm, "layers": band,
                            "best_intermediate": best[:, 0].tolist(), "best_swap_to": best[:, 1].tolist(),
                        }) + "\n")
        if n % 10 == 0:
            print(f"[clamp] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()
    read_handle.close()
    print("[clamp] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
