#!/usr/bin/env python
"""Block 18: is the bridge entity stored, or continuously re-derived?

Three observations point the same way. Clamping a single position does almost
nothing while clamping every position does nine times the sum of the parts.
Widening the band keeps helping, from 0.19 flips at one layer to 0.41 at thirteen.
And the measured effect is a quarter of what the gradient predicts. All three are
what you would see if the model keeps rebuilding the entity from the surrounding
context, so that any local edit is repaired downstream and only a sustained,
everywhere edit survives.

This run measures the repair directly, in coordinates rather than through a readout.

A. Depth recovery. Clamp layers 8..12 only, then track the source coordinate at
   layers 13..24 where nothing is clamped. A stored value would stay where it was
   put; a re-derived one climbs back. The band-wide clamp is the reference.

B. Position repair. Clamp every position except one, and compare with clamping
   every position. If the spared position is where the entity is rebuilt from,
   sparing it should cost far more than that position's own solo contribution.

C. Where does the repair come from? Repeat A with attention to the earlier
   positions cut off at the layers above the clamp, by clamping the same subspace
   at those positions too -- separating "rebuilt from other positions" from
   "rebuilt from this position's own MLP".
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import EnergyStats, clamp_hooks, swapped_fraction
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

READ_LAYERS = list(range(8, 25))
PARTIAL = list(range(8, 13))
FULL = list(range(8, 21))


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block18_rederivation")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in READ_LAYERS}
    wanted = set(names.values())
    depth_handle = (out_dir / "depth_recovery.jsonl").open("w", encoding="utf-8")
    pos_handle = (out_dir / "position_repair.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        seq = tokens.shape[1]
        clean_lp = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        base = float(clean_lp[s] - clean_lp[a])
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, pinvs, clean_coords = {}, {}, {}
            for layer in READ_LAYERS:
                bases[layer] = lens.lens_vectors(model, ids, layer)
                pinvs[layer] = torch.linalg.pinv(bases[layer].T.float())
                clean_coords[layer] = cache[names[layer]].float() @ pinvs[layer].T

            # ---- A: depth recovery ----
            for arm, clamp_layers in (("partial_8_12", PARTIAL), ("full_8_20", FULL)):
                probe = {}

                def watch(name):
                    def fn(act, hook):
                        probe[name] = act.detach().float()
                        return act
                    return fn

                hooks = []
                for layer in clamp_layers:
                    hooks += clamp_hooks(model, bases[layer], [layer], {layer: clean_coords[layer]})
                hooks += [(names[layer], watch(names[layer])) for layer in READ_LAYERS]
                logp = final_logprobs(model, tokens, hooks)
                rows = []
                for layer in READ_LAYERS:
                    got = probe[names[layer]] @ pinvs[layer].T
                    rows.append({"layer": layer,
                                 "swapped_fraction": swapped_fraction(got, clean_coords[layer])})
                depth_handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind, "arm": arm,
                    "clamped_layers": clamp_layers, "profile": rows,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "top1_is_swap": int(logp.argmax().item()) == s,
                }) + "\n")

            # ---- B: leave one position out ----
            if kind == "J":
                for spare in list(range(max(0, seq - 6), seq)) + [None]:
                    positions = [p for p in range(seq) if p != spare] if spare is not None else list(range(seq))
                    stats = EnergyStats()
                    hooks = []
                    for layer in FULL:
                        sel = clean_coords[layer][:, positions, :]
                        hooks += clamp_hooks(model, bases[layer], [layer], {layer: sel}, positions=positions, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    pos_handle.write(json.dumps({
                        "index": index, "name": r.item.name, "spared": spare,
                        "spared_offset": None if spare is None else spare - (seq - 1),
                        "seq_len": seq,
                        "delta_margin": float((logp[s] - logp[a]) - base),
                        "top1_is_swap": int(logp.argmax().item()) == s,
                        "total_dh2": stats.total_dh2,
                    }) + "\n")
        if n % 10 == 0:
            print(f"[rederivation] {n + 1}/{len(eligible)}", flush=True)
    depth_handle.close()
    pos_handle.close()
    print("[block18] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
