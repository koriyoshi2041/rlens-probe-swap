#!/usr/bin/env python
"""Block 52: R vs J in the early band - is the advantage in the LAYER distribution or the POSITION distribution of displacement?

Block 43/47 found "J removes + R installs" beats both lenses in L8-20, but its KL is 1.6x
J's, and R's early-band clamp has 3.5x J's KL. A clamp is the additive injection of its
own delta field (block 28 self-check), so every arm's field can be captured and rescaled
to the SAME per-position norm profile as the J clamp before being re-injected. Arms per
band: J (reference), R, R_s+J_t, J_s+R_t, each in three versions: as-is, matched to J's
per-position norms, matched to J's total energy (single global scale). If the hybrid /
R advantages survive matching, they are about direction; if not, about magnitude.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

BANDS = {"early_L3_8": list(range(3, 9))}
COMBOS = {"J_s+J_t": ("J", "J"), "R_s+R_t": ("R", "R")}


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


def additive_hooks(model, field):
    hooks = []
    for layer, delta in field.items():
        def transform(selected, delta=delta):
            return selected.float() + delta.to(selected.device).float()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, None, model.cfg.d_model)))
    return hooks


def match_positions(field, ref):
    return {l: field[l] / field[l].norm(dim=-1, keepdim=True).clamp_min(1e-9) * ref[l].norm(dim=-1, keepdim=True) for l in ref}


def match_layers(field, ref):
    """Per-layer total energy matched to ref (keeps the within-layer position distribution)."""
    out = {}
    for l in ref:
        e_f = float((field[l] ** 2).sum()); e_r = float((ref[l] ** 2).sum())
        out[l] = field[l] * (e_r / max(e_f, 1e-12)) ** 0.5
    return out


def match_position_totals(field, ref):
    """Per-position energy summed over layers matched to ref (keeps the across-layer distribution at each position)."""
    e_f = torch.stack([(field[l] ** 2).sum(dim=-1) for l in ref]).sum(dim=0)  # [1, pos]
    e_r = torch.stack([(ref[l] ** 2).sum(dim=-1) for l in ref]).sum(dim=0)
    scale = (e_r / e_f.clamp_min(1e-12)).sqrt().unsqueeze(-1)  # [1, pos, 1]
    return {l: field[l] * scale for l in ref}


def match_total(field, ref):
    e_f = sum(float((field[l] ** 2).sum()) for l in ref)
    e_r = sum(float((ref[l] ** 2).sum()) for l in ref)
    s = (e_r / max(e_f, 1e-12)) ** 0.5
    return {l: field[l] * s for l in ref}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block52_layer_vs_position")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    handle = (out_dir / "energy_matched.jsonl").open("w", encoding="utf-8")
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
            vec = {k: {l: lenses[k].lens_vectors(model, ids, l).float() for l in band} for k in ("J", "R")}
            fields, results = {}, {}
            for arm, (ks, kt) in COMBOS.items():
                bases, coords = {}, {}
                for l in band:
                    bases[l] = torch.stack([vec[ks][l][0], vec[kt][l][1]])
                    pinv = torch.linalg.pinv(bases[l].T)
                    coords[l] = cache[names[l]].float() @ pinv.T
                store = {}
                logp = final_logprobs(model, tokens, capturing_clamp_hooks(model, bases, coords, band, store))
                fields[arm] = {l: store[l] for l in band}
                results[(arm, "as_is")] = logp
            ref = fields["J_s+J_t"]
            for arm in COMBOS:
                if arm == "J_s+J_t":
                    continue
                results[(arm, "match_positions")] = final_logprobs(model, tokens, additive_hooks(model, match_positions(fields[arm], ref)))
                results[(arm, "match_total")] = final_logprobs(model, tokens, additive_hooks(model, match_total(fields[arm], ref)))
                results[(arm, "match_layers")] = final_logprobs(model, tokens, additive_hooks(model, match_layers(fields[arm], ref)))
                results[(arm, "match_position_totals")] = final_logprobs(model, tokens, additive_hooks(model, match_position_totals(fields[arm], ref)))
            # J given R's per-layer energy profile / R's per-position profile (does J catch up when it distributes like R?)
            results[("J_s+J_t", "J_with_R_layer_profile")] = final_logprobs(model, tokens, additive_hooks(model, match_layers(ref, fields["R_s+R_t"])))
            results[("J_s+J_t", "J_with_R_position_profile")] = final_logprobs(model, tokens, additive_hooks(model, match_position_totals(ref, fields["R_s+R_t"])))
            # per-layer energy profiles for the record
            profile = {arm: [float((fields[arm][l] ** 2).sum()) for l in band] for arm in fields}
            for (arm, version), logp in results.items():
                top1 = int(logp.argmax().item())
                energy = sum(float((fields[arm][l] ** 2).sum()) for l in band) if version == "as_is" else None
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "band": band_name, "arm": arm, "version": version,
                    "delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                    "kl": kl_divergence(clean, logp), "energy_as_is": energy,
                    "layer_energy_profile": profile[arm] if version == "as_is" else None,
                }) + "\n")
        if n % 10 == 0:
            print(f"[q53] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block48] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
