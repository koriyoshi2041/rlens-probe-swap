#!/usr/bin/env python
"""Block 24: is the clamp direction specifically resisted, or does any strong direction saturate?

Where this comes from. Central differences validated the gradient: a random
direction's measured response matches its first-order prediction (ratio 1.03 at a
3% perturbation). The clamp direction at the *same* perturbation size reaches only
0.28 of its prediction. That looks like the network resisting this particular edit
-- but the two are not comparable in the way that matters. The random direction's
predicted effect is 0.03 nat, so it cannot leave the linear regime; the clamp's is
15 nat, so sub-linearity could just be what any direction with a large first-order
effect does. (Softmax saturation is not the explanation: a log-prob margin is a
logit difference, which is unbounded.)

The control that settles it matches on *predicted effect* instead of on norm:

  gradient   -- steepest ascent on the margin. Large first-order effect by
                construction, but nothing to do with the entity.
  clamp      -- the intervention under study, rescaled to the same prediction.
  contrast   -- the lens vector of the answer contrast, the direct-push direction.

If the gradient direction realises its prediction while the clamp does not, the
clamp is specifically resisted. If all three fall short equally, sub-linearity is
generic and the E3 "damping" claim must be withdrawn.
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
TARGETS = [0.5, 1.0, 2.0, 5.0, 10.0]  # predicted first-order effect, in nat
N_ITEMS = 25


def delta_hooks(model, deltas, sign=1.0):
    hooks = []
    for layer, delta in deltas.items():
        def transform(selected, delta=delta, sign=sign):
            return selected.float() + sign * delta.to(selected.device).float()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def main() -> int:
    out_dir = results_dir("block24_direction_control")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)[:N_ITEMS]
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    handle = (out_dir / "direction_control.jsonl").open("w", encoding="utf-8")
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
            directions = {"gradient": {l: grads[l].clone() for l in BAND}}
            clamp_dir, contrast_dir = {}, {}
            contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
            for layer in BAND:
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                pinv = torch.linalg.pinv(basis.T.float())
                c = acts[layer] @ pinv.T
                clamp_dir[layer] = (c[..., [1, 0]] - c) @ basis.float()
                matrix = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32)
                v = (matrix.T @ contrast)
                contrast_dir[layer] = v.expand_as(acts[layer]).clone()
            directions["clamp"] = clamp_dir
            directions["contrast"] = contrast_dir
            for label, direction in directions.items():
                unit_pred = sum(float((grads[l] * direction[l]).sum().item()) for l in BAND)
                if abs(unit_pred) < 1e-6:
                    continue
                for target in TARGETS:
                    k = target / unit_pred
                    deltas = {l: direction[l] * k for l in BAND}
                    rel = float(np.median([(deltas[l].norm(dim=-1) / acts[l].norm(dim=-1)).median().item() for l in BAND]))
                    got = {}
                    for name, sign in (("plus", 1.0), ("minus", -1.0)):
                        logp = final_logprobs(model, tokens, delta_hooks(model, deltas, sign))
                        got[name] = float((logp[s] - logp[a]) - base)
                    central = (got["plus"] - got["minus"]) / 2
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "direction": label,
                        "predicted": target, "one_sided": got["plus"], "central": central,
                        "curvature": got["plus"] + got["minus"],
                        "ratio_one_sided": got["plus"] / target, "ratio_central": central / target,
                        "median_rel_perturbation": rel,
                    }) + "\n")
        if n % 5 == 0:
            print(f"[dircontrol] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block24] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
