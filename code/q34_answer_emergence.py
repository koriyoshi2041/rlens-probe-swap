#!/usr/bin/env python
"""Block 30: where, and by which sublayer, does the new answer get computed after the clamp?

Block 28 says the clamp's realised effect is 2.3x its first-order effect, 4x in the
flippable items. The first-order part is the clamp's direct injection; the excess is
computed by the network. This block locates that computation.

At the FINAL prompt position, for every layer l >= 8 we cache hook_resid_pre,
hook_resid_mid and hook_out for the clean run and the clamped run (J-lens clamp on
L8-20, all positions). The residual update of layer l splits exactly into

    out(l) - pre(l) = [mid(l) - pre(l)]  +  [out(l) - mid(l)]
                     = attention/linear-attention half + MLP half   (+ the clamp's own
                       delta at hook_out for l in the band, which we subtract off using
                       the captured delta)

and the difference clamped - clean of each half, projected onto the layer's
answer-contrast lens direction u_l = J_l^T(W_U[:,s]-W_U[:,a]) (unit), tells how much
answer signal each sublayer *computed* at that layer. We also record the logit-lens
margin (identity transport through ln_final+unembed) of the final-position residual
at every layer, clean vs clamped, so the "answer flips internally at layer L" curve is
available without any lens.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name, _unembed

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

import os
_b = os.environ.get("MATS_BAND", "8-20")
_lo, _hi = (int(x) for x in _b.split("-"))
BAND = list(range(_lo, _hi + 1))
READ = list(range(_lo, 32))
_pos = os.environ.get("MATS_POSITIONS", "all")  # "all" or "prompt" (= every position except the final one)
OUT_NAME = ("block30_answer_emergence" if _b == "8-20" else f"block30_answer_emergence_band{_lo}_{_hi}") + ("" if _pos == "all" else "_promptonly") + ("_logitdir" if os.environ.get("MATS_DIRECTION") == "logit" else "")


def capturing_clamp_hooks(model, bases, coords, layers, store, positions=None):
    hooks = []
    for layer in layers:
        matrix = bases[layer].T.float()
        pinv = torch.linalg.pinv(matrix)
        target = (coords[layer] if positions is None else coords[layer][:, positions, :])[..., [1, 0]].float()

        def transform(selected, matrix=matrix, pinv=pinv, target=target, layer=layer, n_total=len(coords[layer][0])):
            h = selected.float()
            c = h @ pinv.to(h.device).T
            delta = (target.to(h.device) - c) @ matrix.to(h.device).T
            full = torch.zeros(h.shape[:-2] + (n_total,) + h.shape[-1:], device=h.device, dtype=h.dtype) if positions is not None else None
            if full is not None:
                full[..., positions, :] = delta
                store[layer] = full.detach().clone()
            else:
                store[layer] = delta.detach().clone()
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, positions, model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir(OUT_NAME)
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    hook_names = {}
    for layer in READ:
        hook_names[(layer, "pre")] = f"blocks.{layer}.hook_resid_pre"
        hook_names[(layer, "mid")] = f"blocks.{layer}.hook_resid_mid"
        hook_names[(layer, "out")] = f"blocks.{layer}.hook_out"
    wanted = set(hook_names.values())
    handle = (out_dir / "answer_emergence.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
        _, clean_cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, coords, units = {}, {}, {}
            for layer in BAND:
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                bases[layer] = basis
                pinv = torch.linalg.pinv(basis.T.float())
                coords[layer] = clean_cache[hook_names[(layer, "out")]].float() @ pinv.T
            for layer in READ:
                if os.environ.get("MATS_DIRECTION") == "logit":
                    units[layer] = contrast  # fixed, UNNORMALISED final direction: contributions add up exactly to Δ<h_out, w>
                else:
                    jl = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32) if layer in lens.jacobians else None
                    u = jl.T @ contrast if jl is not None else contrast
                    units[layer] = u / u.norm()
            store = {}
            n_total = tokens.shape[1]
            pos_list = None if _pos == "all" else list(range(n_total - 1))
            hooks = capturing_clamp_hooks(model, bases, coords, BAND, store, positions=pos_list)
            with model.hooks(fwd_hooks=hooks):
                logits, hooked_cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            clean_lp = final_logprobs(model, tokens)
            rows = []
            for layer in READ:
                c_pre, c_mid, c_out = (clean_cache[hook_names[(layer, k)]][0, -1].float() for k in ("pre", "mid", "out"))
                h_pre, h_mid, h_out = (hooked_cache[hook_names[(layer, k)]][0, -1].float() for k in ("pre", "mid", "out"))
                d_attn = (h_mid - h_pre) - (c_mid - c_pre)
                d_mlp = (h_out - h_mid) - (c_out - c_mid)
                direct = store[layer][0, -1] if layer in store else torch.zeros_like(d_mlp)
                d_mlp = d_mlp - direct  # the clamp delta is added at hook_out; remove it from the MLP half
                u = units[layer]
                # logit-lens margin of the final-position residual (identity transport)
                ll_clean = _unembed(model, c_out.unsqueeze(0))[0]
                ll_hook = _unembed(model, h_out.unsqueeze(0))[0]
                rows.append({
                    "layer": layer,
                    "attn_answer": float(d_attn @ u), "mlp_answer": float(d_mlp @ u), "direct_answer": float(direct @ u),
                    "attn_norm": float(d_attn.norm()), "mlp_norm": float(d_mlp.norm()), "direct_norm": float(direct.norm()),
                    "resid_answer_delta": float((h_out - c_out) @ u),
                    "logit_lens_margin_clean": float(ll_clean[s] - ll_clean[a]),
                    "logit_lens_margin_hooked": float(ll_hook[s] - ll_hook[a]),
                })
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "lens": kind,
                "delta_margin": float((logp[s] - logp[a]) - (clean_lp[s] - clean_lp[a])),
                "top1_is_swap": int(logp.argmax().item()) == s,
                "layers": rows,
            }) + "\n")
        if n % 10 == 0:
            print(f"[q34] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block30] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
