#!/usr/bin/env python
"""Block 58: energy-matched comparison of the 2-D lens clamp and the donor paste (single bridge position).

Reasoning audit C-3: the plane-vs-complement comparison is confounded with perturbation
size (KL 0.02 vs 3.8), and the over-driven full-band clamp rescues 15/35 zero-flip items.
Here, at the single best bridge position, L8-20, for every item:
  clamp2d@s   the 2-D J clamp pushed s of the way past the exchange (s = 1, 1.5, 2, 3, 4)
  orth@s      h + s * (donor - h)_perp   (s = 0.1, 0.25, 0.5, 1)  -- complement paste scaled down
  full@s      h + s * (donor - h)        (s = 0.25, 0.5, 1)
  rand{k}     J plane + k RANDOM orthonormal directions (k = 16, 64, 256), dynamic paste (audit C18 control)
  Rtop256     J plane + top-256 right-singular directions of R_l (does R's spectrum differ?)
KL is recorded for every arm so flips can be compared at matched KL offline.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q35_donor_paste import find_last
from q51_subspace_ladder import orthonormal, projector_paste_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def scaled_paste_hooks(model, donors, layers, position, scale, bases=None, mode="full"):
    hooks = []
    for layer in layers:
        donor = donors[layer].float()
        if bases is not None:
            matrix = bases[layer].T.float(); pinv = torch.linalg.pinv(matrix)
        else:
            matrix = pinv = None

        def transform(selected, donor=donor, matrix=matrix, pinv=pinv):
            h = selected.float()
            d = donor.to(h.device).expand_as(h) - h
            if mode == "full":
                return h + scale * d
            in_plane = ((d @ pinv.to(h.device).T) @ matrix.to(h.device).T)
            return h + scale * (d - in_plane)

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block58_energy_matched_paste")
    model = load_model()
    lenses = load_lenses()
    lens = lenses["J"]
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block31_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    svd_R = {}
    for layer in BAND:
        j = lenses["R"].jacobians[layer].to(device=model.W_U.device, dtype=torch.float32)
        _, _, vh = torch.linalg.svd(j, full_matrices=False); svd_R[layer] = vh[:256]
    handle = (out_dir / "energy_matched_paste.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        position = best_pos[index]
        donor_text = templates[str(index)].format(r.item.swap_to)
        dt = model.to_tokens(donor_text, prepend_bos=False); dseq = dt[0].tolist()
        dpos = None
        for variant in (" " + r.item.swap_to, r.item.swap_to):
            dpos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
            if dpos is not None:
                break
        if dpos is None:
            continue
        _, dcache = model.run_with_cache(dt, names_filter=lambda x: x in wanted)
        donor = {layer: dcache[names[layer]][0, dpos] for layer in BAND}
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        bases = {l: lens.lens_vectors(model, ids, l) for l in BAND}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in BAND}
        gen = torch.Generator(device="cpu").manual_seed(index)
        arms = {}
        for sc in (1.0, 1.5, 2.0, 3.0, 4.0):
            hooks = []
            for l in BAND:
                c = coords[l][:, position:position + 1, :]
                target = c + sc * (c[..., [1, 0]] - c)
                hooks += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=[position])
            arms[f"clamp2d@{sc:g}"] = hooks
        for sc in (0.1, 0.25, 0.5, 1.0):
            arms[f"orth@{sc:g}"] = scaled_paste_hooks(model, donor, BAND, position, sc, bases, "orth")
        for sc in (0.25, 0.5, 1.0):
            arms[f"full@{sc:g}"] = scaled_paste_hooks(model, donor, BAND, position, sc, None, "full")
        for k in (16, 64, 256):
            sub = {l: orthonormal(torch.cat([bases[l].float(), torch.randn(k, model.cfg.d_model, generator=gen).to(bases[l].device)])) for l in BAND}
            arms[f"rand{k}"] = projector_paste_hooks(model, donor, BAND, position, sub)
        arms["Rtop256"] = projector_paste_hooks(model, donor, BAND, position, {l: orthonormal(torch.cat([bases[l].float(), svd_R[l].to(bases[l].device)])) for l in BAND})
        for arm, hooks in arms.items():
            logp = final_logprobs(model, tokens, hooks)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm, "position": position,
                "delta_margin": float((logp[s] - logp[a]) - base),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                "kl": kl_divergence(clean, logp),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q63] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block58] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
