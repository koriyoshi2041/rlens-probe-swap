#!/usr/bin/env python
"""Block 53: do the transfer layers read the rewritten entity FROM the bridge positions?

Blocks 39/45: restoring the attention outputs of L17-24 at the final position kills the
rewrite (0.41 -> 0.02); L20 / L24 (gated-delta linear attention) and L23 (full attention,
h8) are the largest single layers. Block 51 could not resolve linear-attention layers by
head. Here we resolve them by INPUT POSITION: the clamp stays on (J, L8-20, all positions)
and the sublayer's input (hook_attn_in / linear_attn.hook_in, i.e. what the layer reads at
each position) is restored to its clean value at chosen positions:

  bridge      the best-readout bridge position only (block 31)
  nonfinal    every position except the final one
  final       the final position only (the query side)
  none        clamp only (reference)

for L17, L20, L23, L24 (and L11 as an inside-band control). If restoring the bridge
position's input alone removes a large part of the flip while restoring the final
position's input does not, the layer is transferring the rewritten entity from the bridge.
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
LAYERS = [11, 17, 20, 23, 24]


def input_hook_name(model, layer):
    for cand in (f"blocks.{layer}.attn.hook_in", f"blocks.{layer}.linear_attn.hook_in"):
        if cand in model.hook_dict:
            return cand
    raise KeyError(layer)


def restore_at(clean_value, positions):
    def fn(act, hook):
        out = act.clone()
        out[:, positions, :] = clean_value[:, positions, :].to(act.dtype)
        return out
    return fn


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block53_transfer_input_positions")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    band_names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    in_names = {layer: input_hook_name(model, layer) for layer in LAYERS}
    print("[q59] input hooks:", in_names, flush=True)
    wanted = set(band_names.values()) | set(in_names.values())
    handle = (out_dir / "transfer_inputs.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        last = tokens.shape[1] - 1
        bp = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clamp = []
        for layer in BAND:
            basis = lens.lens_vectors(model, ids, layer)
            pinv = torch.linalg.pinv(basis.T.float())
            coords = cache[band_names[layer]].float() @ pinv.T
            clamp += clamp_hooks(model, basis, [layer], {layer: coords})
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        arms = {"clamp_only": []}
        for layer in LAYERS:
            cv = cache[in_names[layer]].float()
            arms[f"L{layer}@bridge"] = [(in_names[layer], restore_at(cv, [bp]))]
            arms[f"L{layer}@nonfinal"] = [(in_names[layer], restore_at(cv, list(range(last))))]
            arms[f"L{layer}@final"] = [(in_names[layer], restore_at(cv, [last]))]
        for arm, extra in arms.items():
            logp = final_logprobs(model, tokens, clamp + extra)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm, "bridge_pos": bp, "n_prompt": int(tokens.shape[1]),
                "delta_margin": float((logp[s] - logp[a]) - base),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q59] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block53] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
