#!/usr/bin/env python
"""Block 41: is L23 head 8 NECESSARY for the rewrite to reach the answer?

Block 37 attributed most of the full-attention transfer of the new answer to a single
head, L23.h8 (75% of the layer's contribution, leads 20/24 flipped items). Attribution is
not necessity. Here the J clamp (L8-20, all positions) stays on and one head's output z is
replaced by its CLEAN-run value at the final position (so the head still runs, it just
cannot carry anything the clamp changed):

  L23.h8            the candidate carrier
  L23.h9            second-ranked head
  L23.h0            a head that attends to the bridge but carries little
  L23.random        a random other head (seed per item)
  L23.all           all 16 heads of L23 restored (upper bound for this layer)
  L31.h14           the final-layer read-out head
  L19.h11           the top head of L19 (small contribution)

Measured: flip rate and Δmargin under clamp + restoration, vs clamp only.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
TARGETS = {"L23.h8": (23, [8]), "L23.h9": (23, [9]), "L23.h0": (23, [0]), "L23.all": (23, list(range(16))),
           "L31.h14": (31, [14]), "L19.h11": (19, [11]), "L23.h8+h9": (23, [8, 9])}


def restore_heads_hook(clean_z, heads, position):
    def fn(act, hook):
        out = act.clone()
        out[:, position, heads, :] = clean_z[:, position, heads, :].to(act.dtype)
        return out
    return fn


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block41_head_necessity")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    band_names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    z_names = {layer: f"blocks.{layer}.attn.hook_z" for layer in (19, 23, 31)}
    wanted = set(band_names.values()) | set(z_names.values())
    handle = (out_dir / "head_necessity.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clamp = []
        for layer in BAND:
            basis = lens.lens_vectors(model, ids, layer)
            pinv = torch.linalg.pinv(basis.T.float())
            coords = cache[band_names[layer]].float() @ pinv.T
            clamp += clamp_hooks(model, basis, [layer], {layer: coords})
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        last = tokens.shape[1] - 1
        gen = torch.Generator(device="cpu").manual_seed(index)
        rand_head = int(torch.randint(0, 16, (1,), generator=gen).item())
        while rand_head in (8, 9):
            rand_head = int(torch.randint(0, 16, (1,), generator=gen).item())
        arms = {"clamp_only": []}
        for name, (layer, heads) in TARGETS.items():
            arms[name] = [(z_names[layer], restore_heads_hook(cache[z_names[layer]], heads, last))]
        arms[f"L23.random(h{rand_head})"] = [(z_names[23], restore_heads_hook(cache[z_names[23]], [rand_head], last))]
        arms["L23.h8@allpos"] = [(z_names[23], restore_heads_hook(cache[z_names[23]], [8], slice(None)))]
        for arm, extra in arms.items():
            logp = final_logprobs(model, tokens, clamp + extra)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm.split("(")[0],
                "delta_margin": float((logp[s] - logp[a]) - base),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q46] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block41] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
