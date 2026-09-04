#!/usr/bin/env python
"""Block 13: four experiments aimed at the three things we still cannot explain.

E3 -- linear leak or recomputation? A first-order account of the clamp predicts
     d(margin) = sum_l <grad_l, V_l dc_l>, where grad_l is the gradient of the
     answer margin with respect to the residual at layer l in the clean run. One
     backward pass per item gives that prediction; comparing it with the measured
     effect separates "we rode a linear path to the output" from "the model
     recomputed something". The dose curve says the two item groups differ in
     shape; this says whether the difference is first-order or not.

E2 -- is there a bridge variable at particular positions? Clamping every position
     beats clamping the last three. If the effect is a genuine bridge variable it
     should concentrate at the tokens describing the entity, not spread evenly and
     not sit at the final position.

E4 -- why does R suppress the original entity harder early? The clamp moves the
     source coordinate by (c_t - c_s), so the answer may simply be that R's clean
     coordinate gap is larger. Measured directly, per layer.

E5 -- how much of the effect is the Jacobian, gradedly? Interpolating the
     transport from the identity (logit lens) to J shows whether the transport's
     contribution is graded or a threshold, and lets the two be compared at
     matched perturbation size rather than matched alpha.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
E5_MIX = [0.0, 0.25, 0.5, 0.75, 1.0]


def band_clamp(model, bases, coords, layers, stats=None, scale=1.0):
    hooks = []
    for layer in layers:
        c = coords[layer]
        target = c + scale * (c[..., [1, 0]] - c)
        hooks += clamp_hooks(model, bases[layer], [layer], {layer: target}, exchange=False, stats=stats)
    return hooks


def e3_gradient(model, lenses, resolved, eligible, out_dir):
    """First-order prediction of the clamp's effect vs the measured effect."""
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    handle = (out_dir / "e3_gradient.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        captured = {}

        def grab(name):
            def fn(act, hook):
                act.retain_grad()
                captured[name] = act
                return act

            return fn

        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=[(names[layer], grab(names[layer])) for layer in BAND]):
            logits = model(tokens, return_type="logits")
        logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        margin = logprobs[s] - logprobs[a]
        margin.backward()
        grads = {layer: captured[names[layer]].grad[0].detach().float() for layer in BAND}
        acts = {layer: captured[names[layer]][0].detach().float() for layer in BAND}
        model.zero_grad(set_to_none=True)
        with torch.inference_mode():
            for kind, lens in lenses.items():
                bases, coords, predicted = {}, {}, 0.0
                for layer in BAND:
                    basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    bases[layer] = basis
                    pinv = torch.linalg.pinv(basis.T.float())
                    c = acts[layer] @ pinv.T
                    coords[layer] = c.unsqueeze(0)
                    delta = (c[..., [1, 0]] - c) @ basis.T.float().T
                    predicted += float((grads[layer] * delta).sum().item())
                stats = EnergyStats()
                clean = final_logprobs(model, tokens)
                logp = final_logprobs(model, tokens, band_clamp(model, bases, coords, BAND, stats))
                actual = float((logp[s] - logp[a]) - (clean[s] - clean[a]))
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind,
                    "predicted_linear": predicted, "actual": actual,
                    "ratio": actual / predicted if abs(predicted) > 1e-6 else None,
                    "grad_norm": float(sum(g.norm().item() for g in grads.values())),
                    "top1_is_swap": int(logp.argmax().item()) == s,
                }) + "\n")
        if n % 10 == 0:
            print(f"[E3] {n + 1}/{len(eligible)}", flush=True)
    handle.close()


@torch.inference_mode()
def e2_positions(model, lenses, resolved, eligible, out_dir):
    """Clamp one position at a time across the band."""
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "e2_positions.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        seq = tokens.shape[1]
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        str_tokens = model.to_str_tokens(r.prompt, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            bases, coords = {}, {}
            for layer in BAND:
                bases[layer] = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                pinv = torch.linalg.pinv(bases[layer].T.float())
                coords[layer] = cache[names[layer]].float() @ pinv.T
            for pos in range(seq):
                stats = EnergyStats()
                hooks = []
                for layer in BAND:
                    sel = coords[layer][:, pos : pos + 1, :]
                    hooks += clamp_hooks(model, bases[layer], [layer], {layer: sel}, positions=[pos], stats=stats)
                logp = final_logprobs(model, tokens, hooks)
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind,
                    "pos": pos, "offset_from_end": pos - (seq - 1), "token": str_tokens[pos], "seq_len": seq,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "top1_is_swap": int(logp.argmax().item()) == s,
                    "total_dh2": stats.total_dh2,
                }) + "\n")
        if n % 10 == 0:
            print(f"[E2] {n + 1}/{len(eligible)}", flush=True)
    handle.close()


@torch.inference_mode()
def e4_gaps(model, lenses, resolved, eligible, out_dir):
    """Clean coordinate gap |c_s - c_t| per lens and layer."""
    layers = list(range(3, 25))
    names = {layer: _resid_post_hook_name(layer) for layer in layers}
    wanted = set(names.values())
    rows = []
    for index in eligible:
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind, lens in lenses.items():
            for layer in layers:
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                pinv = torch.linalg.pinv(basis.T.float())
                c = cache[names[layer]].float() @ pinv.T
                gap = (c[..., 0] - c[..., 1]).abs().median()
                rows.append({
                    "index": index, "lens": kind, "layer": layer,
                    "median_abs_gap": float(gap.item()),
                    "median_abs_c_source": float(c[..., 0].abs().median().item()),
                    "basis_norm_ratio": float((basis[0].norm() / basis[1].norm()).item()),
                    "delta_norm": float((gap * (basis[0] - basis[1]).norm()).item()),
                })
    (out_dir / "e4_gaps.json").write_text(json.dumps(rows), encoding="utf-8")
    print("[E4] done")


@torch.inference_mode()
def e5_mix(model, lenses, resolved, eligible, out_dir):
    """Interpolate the transport from identity to J, clamp, and measure."""
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    handle = (out_dir / "e5_mix.jsonl").open("w", encoding="utf-8")
    wu = model.W_U.float()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for kind in ("J", "R"):
            lens = lenses[kind]
            for t in E5_MIX:
                bases, coords = {}, {}
                for layer in BAND:
                    matrix = lens.jacobians[layer].to(device=wu.device, dtype=torch.float32)
                    mixed = (1 - t) * torch.eye(matrix.shape[0], device=matrix.device) + t * matrix
                    basis = (mixed.T @ wu[:, ids]).T
                    bases[layer] = basis
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords[layer] = cache[names[layer]].float() @ pinv.T
                stats = EnergyStats()
                logp = final_logprobs(model, tokens, band_clamp(model, bases, coords, BAND, stats))
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind, "mix": t,
                    "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                    "top1_is_swap": int(logp.argmax().item()) == s,
                    "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                }) + "\n")
        if n % 15 == 0:
            print(f"[E5] {n + 1}/{len(eligible)}", flush=True)
    handle.close()


def main() -> int:
    out_dir = results_dir("block13_deep")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    t0 = time.time()
    e4_gaps(model, lenses, resolved, eligible, out_dir)
    e5_mix(model, lenses, resolved, eligible, out_dir)
    e3_gradient(model, lenses, resolved, eligible, out_dir)
    e2_positions(model, lenses, resolved, eligible, out_dir)
    print(f"[block13] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
