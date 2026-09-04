#!/usr/bin/env python
"""Block 16: is the four-fold shortfall against the first-order prediction real?

The gradient of the answer margin with respect to the band residuals predicts the
clamp should move the margin by +14 nat; it moves it by +3.6. Before calling that
"the network damps the edit", the boring explanation has to go: the clamp is not a
small perturbation (||dh||/||h|| is 6-12% per layer), so a first-order extrapolation
over that distance may simply be invalid.

Test: sweep the clamp from a genuinely infinitesimal fraction of its full size
upward. If actual/predicted approaches 1 as the step shrinks, the gradient is right
and the shortfall is honest large-signal damping, whose size then becomes a real
quantity -- and one that differs between the items that flip and those that do not.
If the ratio stays near 0.26 even at 1% of the step, the linearisation itself is
wrong and the E3 result must be withdrawn.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import EnergyStats, clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BAND = list(range(8, 21))
SCALES = [0.01, 0.03, 0.1, 0.25, 0.5, 1.0]


def main() -> int:
    out_dir = results_dir("block16_damping")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    handle = (out_dir / "damping.jsonl").open("w", encoding="utf-8")
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
                bases, coords, full_delta = {}, {}, {}
                predicted_full = 0.0
                for layer in BAND:
                    basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    bases[layer] = basis
                    pinv = torch.linalg.pinv(basis.T.float())
                    c = acts[layer] @ pinv.T
                    coords[layer] = c.unsqueeze(0)
                    delta = (c[..., [1, 0]] - c) @ basis.float()
                    full_delta[layer] = delta
                    predicted_full += float((grads[layer] * delta).sum().item())
                for scale in SCALES:
                    stats = EnergyStats()
                    hooks = []
                    for layer in BAND:
                        c = coords[layer]
                        target = c + scale * (c[..., [1, 0]] - c)
                        hooks += clamp_hooks(model, bases[layer], [layer], {layer: target}, exchange=False, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    actual = float((logp[s] - logp[a]) - base)
                    predicted = predicted_full * scale
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "lens": kind, "scale": scale,
                        "predicted_linear": predicted, "actual": actual,
                        "ratio": actual / predicted if abs(predicted) > 1e-6 else None,
                        "total_dh2": stats.total_dh2,
                        "max_rel_perturbation": max(stats.max_rel.values()) if stats.max_rel else 0.0,
                        "top1_is_swap": int(logp.argmax().item()) == s,
                    }) + "\n")
        if n % 10 == 0:
            print(f"[damping] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block16] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
