#!/usr/bin/env python
"""Block 29: is the clamp's nonlinear gain specific to the entity-rewrite field?

Block 28 showed the clamp realises ~2.3x its first-order prediction (4x in the
flippable items). Before reading that as "the model completes the second hop on the
rewritten entity", the boring alternative must be excluded: any perturbation of that
size in that region might be amplified by generic curvature.

For each item and lens we compare, at the SAME per-layer, per-position norm profile
as the clamp's actual delta field (energy-matched) and, separately, rescaled so the
first-order prediction equals the clamp's (prediction-matched):

  clamp     : the clamp's actual delta field (reference; realised == clamp exactly)
  contrast  : the answer-contrast lens direction u_l = J_l^T(W_U[:,s] - W_U[:,a]) at
              every band layer and position (a pure "push the answer" field)
  ortho     : the clamp field with its component along u_l removed (the mediated part)
  random    : a Gaussian field with the same norm profile (curvature-only control;
              first-order prediction ~0, so only the realised effect is reported)

gain = realised / first-order prediction. If contrast shows the same gain as clamp,
the gain is generic; if clamp >> contrast, the model does something with the moved
entity that a direct push does not trigger.
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


def capturing_clamp_hooks(model, bases, coords, layers, store):
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


def additive_hooks(model, field, scale=1.0):
    hooks = []
    for layer, delta in field.items():
        def transform(selected, delta=delta, scale=scale):
            return selected.float() + scale * delta.to(selected.device).float()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def match_norms(field, reference):
    """Rescale each (layer, position) vector of `field` to the norm of the same vector in `reference`."""
    out = {}
    for layer, ref in reference.items():
        f = field[layer]
        out[layer] = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9) * ref.norm(dim=-1, keepdim=True)
    return out


def main() -> int:
    out_dir = results_dir("block29_gain_controls")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    gen = torch.Generator(device="cpu").manual_seed(0)
    handle = (out_dir / "gain_controls.jsonl").open("w", encoding="utf-8")
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
        contrast = model.W_U[:, s].float() - model.W_U[:, a].float()
        with torch.inference_mode():
            clean = final_logprobs(model, tokens)
            base = float(clean[s] - clean[a])
            for kind, lens in lenses.items():
                bases, coords, units = {}, {}, {}
                for layer in BAND:
                    basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    bases[layer] = basis
                    pinv = torch.linalg.pinv(basis.T.float())
                    coords[layer] = (acts[layer] @ pinv.T).unsqueeze(0)
                    u = lens.jacobians[layer].to(device=contrast.device, dtype=torch.float32).T @ contrast
                    units[layer] = u / u.norm()
                store = {}
                logp = final_logprobs(model, tokens, capturing_clamp_hooks(model, bases, coords, BAND, store))
                clamp_field = {layer: store[layer][0] for layer in BAND}  # [pos, d]
                fields = {"clamp": clamp_field}
                fields["contrast"] = match_norms({layer: units[layer].expand_as(clamp_field[layer]).clone() for layer in BAND}, clamp_field)
                fields["ortho"] = {layer: clamp_field[layer] - (clamp_field[layer] @ units[layer]).unsqueeze(-1) * units[layer] for layer in BAND}
                fields["random"] = match_norms({layer: torch.randn(clamp_field[layer].shape, generator=gen).to(clamp_field[layer].device) for layer in BAND}, clamp_field)
                preds = {k: sum(float((grads[l] * f[l]).sum().item()) for l in BAND) for k, f in fields.items()}
                energies = {k: sum(float((f[l] ** 2).sum().item()) for l in BAND) for k, f in fields.items()}
                realised = {}
                for k, f in fields.items():
                    lp_k = final_logprobs(model, tokens, additive_hooks(model, f))
                    realised[k] = float((lp_k[s] - lp_k[a]) - base)
                # prediction-matched arms: rescale contrast and ortho so their first-order prediction equals the clamp's
                matched = {}
                for k in ("contrast", "ortho"):
                    if abs(preds[k]) > 1e-6 and preds["clamp"] > 0:
                        scale = preds["clamp"] / preds[k]
                        lp_k = final_logprobs(model, tokens, additive_hooks(model, fields[k], scale))
                        matched[k] = {"scale": scale, "realised": float((lp_k[s] - lp_k[a]) - base),
                                      "energy": energies[k] * scale * scale}
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "lens": kind,
                    "clamp_delta_margin_from_hook": float((logp[s] - logp[a]) - base),
                    "predicted": preds, "realised": realised, "energy": energies,
                    "prediction_matched": matched,
                    "top1_is_swap": int(logp.argmax().item()) == s,
                }) + "\n")
        if n % 10 == 0:
            print(f"[q33] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block29] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
