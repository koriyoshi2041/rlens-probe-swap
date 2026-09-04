#!/usr/bin/env python
"""Block 54: a k-dimensional lens intervention without a donor sentence.

Block 46/49: what the 2-D lens plane misses lies in the top ~64-256 right-singular
directions of J_l. A practical k-dim lens intervention needs a target direction inside that
subspace without a context-matched donor. Cheapest source: the ENTITY DIFFERENCE between the
target and source tokens in a context-free carrier ("Fact: Consider the following: X"),
d_l = h_l(target) - h_l(source), restricted to the chosen subspace and ADDED (not pasted) at
the bridge position, at every band layer:

  plane_J        P = J plane (2-D)                        -> the 2-D steer analogue of the clamp
  top{k}         P = J plane + top-k right-singular directions of J_l, k in 16/64/256
  random256      P = J plane + 256 random directions (control)
  full_diff      d added in full (no projection)
  clamp2d        the ordinary 2-D clamp at that position (reference)
  paste_full     full-vector donor replacement (block 31 reference, N2 donor)

Scale: d is used as is (scale 1) and at scale 2. Flip rates and KL are recorded.
"""
from __future__ import annotations

import json
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q35_donor_paste import find_last
from q51_subspace_ladder import orthonormal
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TOPK = [16, 64, 256]
CARRIER = "Fact: Consider the following: {}"


def entity_vec(model, entity, wanted, names):
    tokens = model.to_tokens(CARRIER.format(entity), prepend_bos=False)
    seq = tokens[0].tolist()
    pos = None
    for variant in (" " + entity, entity):
        pos = find_last(seq, list(encode_variant(model.tokenizer, variant).ids))
        if pos is not None:
            break
    if pos is None:
        return None
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {layer: cache[names[layer]][0, pos].float() for layer in BAND}


def add_hooks(model, deltas, position, scale):
    hooks = []
    for layer, delta in deltas.items():
        def transform(selected, delta=delta, scale=scale):
            return selected.float() + scale * delta.to(selected.device)
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


def paste_hooks(model, donor, position):
    hooks = []
    for layer, vec in donor.items():
        def transform(selected, vec=vec):
            return vec.to(selected.device).expand_as(selected.float()).clone()
        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block54_kdim_lens_steer")
    model = load_model()
    lens = load_lenses()["J"]
    items = load_items()
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
    svd_v = {}
    for layer in BAND:
        j = lens.jacobians[layer].to(device=model.W_U.device, dtype=torch.float32)
        _, _, vh = torch.linalg.svd(j, full_matrices=False)
        svd_v[layer] = vh
    handle = (out_dir / "kdim_steer.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        position = best_pos[index]
        src_vec = entity_vec(model, r.item.intermediate, wanted, names)
        tgt_vec = entity_vec(model, r.item.swap_to, wanted, names)
        if src_vec is None or tgt_vec is None:
            continue
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        vecJ = {l: lens.lens_vectors(model, ids, l).float() for l in BAND}
        diff = {l: tgt_vec[l] - src_vec[l] for l in BAND}
        gen = torch.Generator(device="cpu").manual_seed(index)
        subspaces = {"plane_J": {l: orthonormal(vecJ[l]) for l in BAND}}
        for k in TOPK:
            subspaces[f"top{k}"] = {l: orthonormal(torch.cat([vecJ[l], svd_v[l][:k].to(vecJ[l].device)])) for l in BAND}
        subspaces["random256"] = {l: orthonormal(torch.cat([vecJ[l], torch.randn(256, model.cfg.d_model, generator=gen).to(vecJ[l].device)])) for l in BAND}
        arms = {}
        for sname, sub in subspaces.items():
            proj = {l: (diff[l] @ sub[l].T) @ sub[l] for l in BAND}
            for scale in (1.0, 2.0):
                arms[f"{sname}@x{scale:g}"] = add_hooks(model, proj, position, scale)
        for scale in (1.0, 2.0):
            arms[f"full_diff@x{scale:g}"] = add_hooks(model, diff, position, scale)
        bases, coords = {}, {}
        for l in BAND:
            bases[l] = lens.lens_vectors(model, ids, l)
            pinv = torch.linalg.pinv(bases[l].T.float())
            coords[l] = cache[names[l]].float() @ pinv.T
        arms["clamp2d"] = sum((clamp_hooks(model, bases[l], [l], {l: coords[l][:, position:position + 1, :]}, positions=[position]) for l in BAND), [])
        arms["paste_full_N2"] = paste_hooks(model, tgt_vec, position)
        energies = {sname: sum(float((((diff[l] @ sub[l].T) @ sub[l]) ** 2).sum()) for l in BAND) for sname, sub in subspaces.items()}
        energies["full_diff"] = sum(float((diff[l] ** 2).sum()) for l in BAND)
        for arm, hooks in arms.items():
            logp = final_logprobs(model, tokens, hooks)
            top1 = int(logp.argmax().item())
            handle.write(json.dumps({
                "index": index, "name": r.item.name, "arm": arm, "position": position,
                "delta_margin": float((logp[s] - logp[a]) - base),
                "top1_is_swap": top1 == s, "top1_is_answer": top1 == a, "top1_str": model.tokenizer.decode([top1]),
                "kl": kl_divergence(clean, logp),
                "energy": energies.get(arm.split("@")[0]),
            }) + "\n")
        if n % 10 == 0:
            print(f"[q58] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block54] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
