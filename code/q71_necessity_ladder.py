#!/usr/bin/env python
"""Block 71: NECESSITY ladder — the full donor paste minus the top-k lens directions.

Blocks 46/62/64 are sufficiency ladders (J plane + top-k directions). The complementary
question: is the lens high-gain subspace NECESSARY for the donor paste to work, i.e. does the
full paste still flip when its component in the top-k right-singular directions of J_l is
removed?  At the best bridge position, L8-20, scale 0.25 (block 62: full@0.25 already
saturates at 0.66 / 0.51 on the never-flipped set):
  full − top-k(J)      k = 2 (the J plane itself), 16, 64, 256, 1024
  full − top-k(R)      k = 256
  full − random-k      k = 16, 64, 256, 1024   (same-dimension control, seeded per item)
  full                 reference
Records flips, Δmargin, KL and the injected energy per arm. If removing the top-256 lens
directions kills the paste while removing random-256 does not, the lens subspace is where the
transferable content lives (necessity), complementing block 64's propagation result.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q35_donor_paste import find_last
from q51_subspace_ladder import orthonormal
from q66_energy_of_arms import energy_wrapped
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
SCALE = 0.25
TOPK = (2, 16, 64, 256, 1024)
RANDK = (16, 64, 256, 1024)


def removal_paste_hooks(model, donors, layers, position, scale, remove_by_layer):
    """h <- h + scale * (I - QQ^T)(donor - h): the full paste with the subspace Q removed (Q=None: full)."""
    hooks = []
    for layer in layers:
        donor = donors[layer].float()
        q = None if remove_by_layer is None else remove_by_layer[layer].float()

        def transform(selected, donor=donor, q=q):
            h = selected.float()
            d = donor.to(h.device).expand_as(h) - h
            if q is not None:
                qq = q.to(h.device)
                d = d - (d @ qq.T) @ qq
            return h + scale * d

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block71_necessity_ladder")
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
    svd = {}
    for kind in ("J", "R"):
        svd[kind] = {}
        for layer in BAND:
            j = lenses[kind].jacobians[layer].to(device=model.W_U.device, dtype=torch.float32)
            _, _, vh = torch.linalg.svd(j, full_matrices=False)
            svd[kind][layer] = vh
    handle = (out_dir / "necessity_ladder.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        position = best_pos[index]
        donor_text = templates[str(index)].format(r.item.swap_to)
        dt = model.to_tokens(donor_text, prepend_bos=False)
        dseq = dt[0].tolist()
        dpos = None
        for variant in (" " + r.item.swap_to, r.item.swap_to):
            dpos = find_last(dseq, list(encode_variant(model.tokenizer, variant).ids))
            if dpos is not None:
                break
        if dpos is None:
            continue
        _, dcache = model.run_with_cache(dt, names_filter=lambda x: x in wanted)
        donor = {layer: dcache[names[layer]][0, dpos] for layer in BAND}
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        bases = {l: lens.lens_vectors(model, ids, l).float() for l in BAND}
        gen = torch.Generator(device="cpu").manual_seed(7000 + index)
        arms = {"full": None}
        for k in TOPK:
            arms[f"full-Jtop{k}"] = {l: orthonormal(torch.cat([bases[l], svd["J"][l][:k].to(bases[l].device)])) if k > 2 else orthonormal(bases[l]) for l in BAND}
        arms["full-Rtop256"] = {l: orthonormal(torch.cat([bases[l], svd["R"][l][:256].to(bases[l].device)])) for l in BAND}
        for k in RANDK:
            arms[f"full-rand{k}"] = {l: orthonormal(torch.cat([bases[l], torch.randn(k, model.cfg.d_model, generator=gen).to(bases[l].device)])) for l in BAND}
        for arm, remove in arms.items():
            store = {}
            logp = final_logprobs(model, tokens, energy_wrapped(removal_paste_hooks(model, donor, BAND, position, SCALE, remove), position, store))
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm, "position": position, "energy": store.get("energy", 0.0),
                "delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                "top1_str": model.tokenizer.decode([top1]), "kl": kl_divergence(clean, logp),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q71] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block71] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
