#!/usr/bin/env python
"""Block 37: which attention heads carry the rewritten entity to the answer position?

Block 30 showed that after the clamp the new answer is first written into the final
position by the attention halves of L17-L24 (L23 is a full-attention layer; L17/20/24
are gated-delta linear-attention layers) and then amplified by the MLPs of L26-29.
This block decomposes the full-attention layers (L19, L23, L27, L31; 16 heads x 256)
head by head:

  * per-head output at the final position, clamped minus clean, projected onto the
    layer's answer-contrast lens direction u_l (unit) -> which heads write the new answer;
  * the attention pattern from the final query position onto the best bridge position
    (block31's best-readout position) and onto all clamped positions, clean vs clamped
    -> whether those heads look at the rewritten positions.

Per-head outputs are reconstructed as z_h @ W_O[h] from attn.hook_z (fallback to
attn.hook_result if the bridge provides it). Items split by whether the band clamp
flips them (block12).
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
FULL_ATTN = [19, 23, 27, 31]


def capturing_clamp_hooks(model, bases, coords, layers):
    hooks = []
    for layer in layers:
        matrix = bases[layer].T.float()
        pinv = torch.linalg.pinv(matrix)
        target = coords[layer][..., [1, 0]].float()

        def transform(selected, matrix=matrix, pinv=pinv, target=target):
            h = selected.float()
            c = h @ pinv.to(h.device).T
            return h + (target.to(h.device) - c) @ matrix.to(h.device).T

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def head_outputs(model, cache, layer):
    """[pos, n_heads, d_model] per-head output at every position."""
    name_r = f"blocks.{layer}.attn.hook_result"
    if name_r in cache and cache[name_r].abs().sum() > 0:
        return cache[name_r][0].float()
    z = cache[f"blocks.{layer}.attn.hook_z"][0].float()  # [pos, n_heads, d_head]
    # the bridge stacks W_O over FULL-attention blocks only (8 of 32 on Qwen3.5-9B)
    full_layers = [l for l in range(model.cfg.n_layers) if f"blocks.{l}.attn.hook_z" in model.hook_dict]
    w_o = model.W_O[full_layers.index(layer)].float()  # [n_heads, d_head, d_model]
    if w_o.shape[0] != z.shape[1]:
        w_o = w_o.reshape(z.shape[1], -1, w_o.shape[-1])
    return torch.einsum("phd,hdm->phm", z, w_o)


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block37_head_attribution")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    p = RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl"
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    sup = [json.loads(l) for l in (RESULTS_DIR / "block12_suppression" / "suppression.jsonl").read_text().splitlines() if l.strip()]
    band_flip = {r["index"]: r["top1_is_swap"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp"}
    print("W_O shape", tuple(model.W_O.shape), flush=True)
    wanted = set()
    for layer in FULL_ATTN:
        wanted |= {f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.attn.hook_result", f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.hook_attn_out"}
    wanted |= {_resid_post_hook_name(layer) for layer in BAND}
    handle = (out_dir / "head_attribution.jsonl").open("w", encoding="utf-8")
    checked = False
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
        _, clean_cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        bases, coords = {}, {}
        for layer in BAND:
            bases[layer] = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
            pinv = torch.linalg.pinv(bases[layer].T.float())
            coords[layer] = clean_cache[_resid_post_hook_name(layer)].float() @ pinv.T
        with model.hooks(fwd_hooks=capturing_clamp_hooks(model, bases, coords, BAND)):
            logits, hooked_cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        clean_lp = final_logprobs(model, tokens)
        bp = best_pos.get(index, tokens.shape[1] - 2)
        if not checked:
            for layer in FULL_ATTN:
                ho = head_outputs(model, clean_cache, layer).sum(dim=1)  # [pos, d_model]
                ref_name = f"blocks.{layer}.hook_attn_out"
                ref = clean_cache[ref_name][0].float() if ref_name in clean_cache else None
                if ref is not None:
                    rel = float((ho - ref).norm() / ref.norm())
                    print(f"[q42] sanity L{layer}: sum_h z_h W_O[h] vs hook_attn_out rel err {rel:.3g}", flush=True)
            checked = True
        layers_out = {}
        for layer in FULL_ATTN:
            jl = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32) if layer in lens.jacobians else None
            u = jl.T @ contrast if jl is not None else contrast
            u = u / u.norm()
            ho_c = head_outputs(model, clean_cache, layer)[-1]  # [n_heads, d_model] at final position
            ho_h = head_outputs(model, hooked_cache, layer)[-1]
            delta_proj = ((ho_h - ho_c) @ u).tolist()
            pat_c = clean_cache[f"blocks.{layer}.attn.hook_pattern"][0, :, -1, :].float()  # [n_heads, key_pos]
            pat_h = hooked_cache[f"blocks.{layer}.attn.hook_pattern"][0, :, -1, :].float()
            layers_out[str(layer)] = {
                "head_delta_answer": delta_proj,
                "head_delta_norm": (ho_h - ho_c).norm(dim=-1).tolist(),
                "attn_to_best_clean": pat_c[:, bp].tolist(), "attn_to_best_hooked": pat_h[:, bp].tolist(),
                "attn_to_prompt_clean": pat_c[:, :-1].sum(dim=-1).tolist(), "attn_to_prompt_hooked": pat_h[:, :-1].sum(dim=-1).tolist(),
            }
        handle.write(json.dumps({
            "index": index, "name": r.item.name, "band_flip": bool(band_flip.get(index, False)), "best_pos": bp, "n_prompt": int(tokens.shape[1]),
            "delta_margin": float((logp[s] - logp[a]) - (clean_lp[s] - clean_lp[a])),
            "top1_is_swap": int(logp.argmax().item()) == s, "layers": layers_out,
        }) + "\n")
        if n % 10 == 0:
            print(f"[q42] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block37] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
