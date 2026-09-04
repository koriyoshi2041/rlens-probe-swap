#!/usr/bin/env python
"""Block 43: which of R's two early-layer directions carries its advantage?

Block 40 showed that extra suppression does not let J catch up with R in the early band,
so R's advantage must sit in its DIRECTIONS. A clamp uses exactly two vectors per layer,
v_s (what is removed) and v_t (what is installed). Hybrid bases isolate them:

  J_s+J_t   plain J clamp        R_s+R_t   plain R clamp
  R_s+J_t   R's source vector, J's target vector
  J_s+R_t   J's source vector, R's target vector

Early band L3-8 (where R beats J) and, as a control, the primary band L8-20 (where they
tie). If R_s+J_t ~ R, the advantage lives in what R removes; if J_s+R_t ~ R, in what it
installs; if both hybrids fall between, the two directions must match each other.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BANDS = {"early_L3_8": list(range(3, 9)), "primary_L8_20": list(range(8, 21))}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block43b_hybrid_basis_unit" if os.environ.get("MATS_HYBRID_UNIT") else "block43_hybrid_basis")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "hybrid_basis.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        for band_name, band in BANDS.items():
            names = {layer: _resid_post_hook_name(layer) for layer in band}
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in set(names.values()))
            vec = {k: {layer: lenses[k].lens_vectors(model, ids, layer) for layer in band} for k in ("J", "R")}
            combos = {"J_s+J_t": ("J", "J"), "R_s+R_t": ("R", "R"), "R_s+J_t": ("R", "J"), "J_s+R_t": ("J", "R")}
            variants = [("raw", False), ("unit", True)] if os.environ.get("MATS_HYBRID_UNIT") else [("raw", False)]
            for arm0, (ks, kt) in combos.items():
              for vname, unit in variants:
                arm = arm0 if vname == "raw" else f"{arm0}|unit"
                hooks = []
                for layer in band:
                    basis = torch.stack([vec[ks][layer][0], vec[kt][layer][1]])
                    if unit:
                        basis = basis / basis.norm(dim=-1, keepdim=True)
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords = cache[names[layer]].float() @ pinv.T
                    hooks += clamp_hooks(model, basis, [layer], {layer: coords})
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "band": band_name, "arm": arm,
                    "delta_margin": float((logp[s] - logp[a]) - base),
                    "delta_logp_answer": float(logp[a] - clean[a]), "delta_logp_swap_answer": float(logp[s] - clean[s]),
                    "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q48] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block43] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
