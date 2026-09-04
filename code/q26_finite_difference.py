#!/usr/bin/env python
"""Block 21: is the gradient right, and is the shortfall second-order?

The damping sweep did not converge to the first-order prediction as the step
shrank -- it went to zero instead. The reason is measurement, not physics: the
intervention hooks cast back to bf16, whose relative precision is about 0.004,
and at the smallest steps the perturbation was 0.001 of the residual norm, i.e.
rounded away. Everything below roughly 3% relative perturbation is unusable in
this model's dtype.

Central differences settle it inside the usable range. For a direction d,
    f(+d) - f(-d) = 2 <grad, d> + O(||d||^3)
so the odd part of the response isolates the first-order term and cancels the
quadratic one, while
    f(+d) + f(-d) - 2 f(0) = <d, H d> + O(||d||^4)
gives the curvature that the one-sided measurement was mixing in.

Two directions are tested at each of several sizes: a random direction, which
checks that the gradient itself is correct, and the clamp's own direction, which
asks whether its shortfall survives once curvature is separated out.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
REL_SIZES = [0.03, 0.06, 0.12, 0.25]
N_ITEMS = 20


def delta_hooks(model, deltas, sign=1.0):
    hooks = []
    for layer, delta in deltas.items():
        def transform(selected, delta=delta, sign=sign):
            return selected.float() + sign * delta.to(selected.device).float()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def measure(model, tokens, deltas, a, s, base):
    out = {}
    for name, sign in (("plus", 1.0), ("minus", -1.0)):
        logp = final_logprobs(model, tokens, delta_hooks(model, deltas, sign))
        out[name] = float((logp[s] - logp[a]) - base)
    return out


def main() -> int:
    out_dir = results_dir("block21_finite_difference")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)[:N_ITEMS]
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    gen = torch.Generator(device="cpu").manual_seed(0)
    handle = (out_dir / "finite_difference.jsonl").open("w", encoding="utf-8")
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
            lens = lenses["J"]
            clamp_dir = {}
            for layer in BAND:
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                pinv = torch.linalg.pinv(basis.T.float())
                c = acts[layer] @ pinv.T
                clamp_dir[layer] = (c[..., [1, 0]] - c) @ basis.float()
            rand_dir = {layer: torch.randn(acts[layer].shape, generator=gen).to(acts[layer].device) for layer in BAND}
            for layer in BAND:  # match the clamp's per-position norm so the sizes are comparable
                target = clamp_dir[layer].norm(dim=-1, keepdim=True)
                rand_dir[layer] = rand_dir[layer] / rand_dir[layer].norm(dim=-1, keepdim=True) * target
            for label, direction in (("clamp", clamp_dir), ("random", rand_dir)):
                full_rel = float(np.median([
                    (direction[l].norm(dim=-1) / acts[l].norm(dim=-1)).median().item() for l in BAND
                ]))
                for rel in REL_SIZES:
                    scale = rel / full_rel if full_rel > 0 else 0.0
                    deltas = {l: direction[l] * scale for l in BAND}
                    predicted = sum(float((grads[l] * deltas[l]).sum().item()) for l in BAND)
                    got = measure(model, tokens, deltas, a, s, base)
                    central = (got["plus"] - got["minus"]) / 2
                    curvature = got["plus"] + got["minus"]
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "direction": label,
                        "target_rel": rel, "achieved_rel": rel,
                        "predicted_linear": predicted,
                        "one_sided": got["plus"], "central_difference": central, "curvature_term": curvature,
                        "ratio_one_sided": got["plus"] / predicted if abs(predicted) > 1e-4 else None,
                        "ratio_central": central / predicted if abs(predicted) > 1e-4 else None,
                    }) + "\n")
        if n % 5 == 0:
            print(f"[fd] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block21] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
