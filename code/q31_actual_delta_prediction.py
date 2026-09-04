#!/usr/bin/env python
"""Block 28: the clamp's first-order prediction, computed with the deltas it actually applies.

Why. Blocks 13/16/21/24 compared the measured clamp effect with a "first-order
prediction" built from the *clean-coordinate* delta at every band layer:
Σ_l <grad_l, V_l(σ(c_l^clean) − c_l^clean)>. But the clamp is idempotent: once L8
has moved the coordinates, L9…L20 find them already exchanged and apply almost
nothing (block08 energy profile: L8 carries 44% of the total). Summing the clean
delta at all 13 layers counts the L8 push up to 13 times, so the ~14 nat
"prediction" is inflated and the "saturation" story built on it is unsupported.

This block records, per item and lens:
  * grad_l of the final-position answer margin w.r.t. hook_out at every band layer
    (clean run, as before);
  * the delta the clamp actually applies at each layer (captured from the hook);
  * predicted_clean  = Σ_l <grad_l, δ_l^clean>   (the old, inflated number)
  * predicted_actual = Σ_l <grad_l, δ_l^actual>  (the consistent first-order number)
  * actual ΔM of the clamp;
  * ΔM when the captured actual deltas are re-injected additively at scale s ∈
    {0.25, 0.5, 1.0} (s=1.0 must reproduce the clamp exactly: a self-check), which
    gives a dose-response for the *same* direction field and lets us see whether the
    response is linear in the regime the clamp actually operates in.
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
SCALES = (0.25, 0.5, 1.0)


def capturing_clamp_hooks(model, bases, coords, layers, store):
    """Same algebra as rlens.interventions.clamp_hooks (exchange=True), but stores the applied delta per layer."""
    hooks = []
    for layer in layers:
        matrix = bases[layer].T.float()
        pinv = torch.linalg.pinv(matrix)
        target = coords[layer][..., [1, 0]].float()

        def transform(selected, matrix=matrix, pinv=pinv, target=target, layer=layer):
            h = selected.float()
            c = h @ pinv.to(h.device).T
            delta = (target.to(h.device) - c) @ matrix.to(h.device).T
            store[layer] = delta.detach().clone()
            return h + delta

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def additive_hooks(model, deltas, scale):
    hooks = []
    for layer, delta in deltas.items():
        def transform(selected, delta=delta, scale=scale):
            return selected.float() + scale * delta.to(selected.device).float()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def main() -> int:
    out_dir = results_dir("block28_actual_delta")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    handle = (out_dir / "actual_delta.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
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
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        (lp[s] - lp[a]).backward()
        grads = {layer: captured[names[layer]].grad[0].detach().float() for layer in BAND}
        acts = {layer: captured[names[layer]][0].detach().float() for layer in BAND}
        model.zero_grad(set_to_none=True)
        with torch.inference_mode():
            clean = final_logprobs(model, tokens)
            base = float(clean[s] - clean[a])
            for kind, lens in lenses.items():
                bases, coords, clean_delta = {}, {}, {}
                for layer in BAND:
                    basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    bases[layer] = basis
                    pinv = torch.linalg.pinv(basis.T.float())
                    c = acts[layer] @ pinv.T
                    coords[layer] = c.unsqueeze(0)
                    clean_delta[layer] = ((c[..., [1, 0]] - c) @ basis.float())
                store = {}
                logp = final_logprobs(model, tokens, capturing_clamp_hooks(model, bases, coords, BAND, store))
                actual = float((logp[s] - logp[a]) - base)
                actual_delta = {layer: store[layer][0] for layer in BAND}  # [pos, d]
                pred_clean = {layer: float((grads[layer] * clean_delta[layer]).sum().item()) for layer in BAND}
                pred_actual = {layer: float((grads[layer] * actual_delta[layer]).sum().item()) for layer in BAND}
                energy_actual = {layer: float((actual_delta[layer] ** 2).sum().item()) for layer in BAND}
                energy_clean = {layer: float((clean_delta[layer] ** 2).sum().item()) for layer in BAND}
                dose = {}
                for scale in SCALES:
                    lp_s = final_logprobs(model, tokens, additive_hooks(model, actual_delta, scale))
                    dose[str(scale)] = float((lp_s[s] - lp_s[a]) - base)
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind,
                    "actual_delta_margin": actual,
                    "top1_is_swap": int(logp.argmax().item()) == s,
                    "predicted_clean_sum": sum(pred_clean.values()),
                    "predicted_actual_sum": sum(pred_actual.values()),
                    "predicted_clean_per_layer": pred_clean,
                    "predicted_actual_per_layer": pred_actual,
                    "energy_actual_per_layer": energy_actual,
                    "energy_clean_per_layer": energy_clean,
                    "additive_reinjection_delta_margin": dose,
                }) + "\n")
        if n % 10 == 0:
            print(f"[q31] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block28] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
