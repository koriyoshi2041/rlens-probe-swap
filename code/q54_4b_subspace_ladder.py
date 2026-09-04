#!/usr/bin/env python
"""Block 49: the block-46 subspace ladder on Qwen3.5-4B.

Block 31/38: pasting the donor's orthogonal complement (relative to the J plane) flips
46% of the band-unflippable items; the plane's own coordinates flip 11%. Two questions:

  (a) Is the missing information in directions the Jacobian transport AMPLIFIES (the lens
      could see it, two dimensions are just too few) or in directions it ATTENUATES (the
      lens cannot see it at all)? We split the complement by the right-singular vectors of
      J_l and paste only the part inside the top-k singular directions (k = 16 ... 1024) or
      only the part outside the top-256.
  (b) How many lens-style dimensions would suffice? Ladder: J plane (2) -> J plane + R's
      target vector (3) -> J and R planes together (4) -> R plane alone (2).

All pastes are dynamic (h <- h + Q Q^T (donor - h) at the best bridge position, Q an
orthonormal basis of the chosen subspace, at every band layer), so they compose the same
way as block 31. Relation donor as in block 31; neutral donor N2 for the two headline arms.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _make_intervention_hook, _resid_post_hook_name

from q35_donor_paste import find_last
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from q30_4b_replication import load_4b
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TOPK = [16, 64, 256, 1024]
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def orthonormal(rows: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(rows.T.float())
    return q.T  # [k, d]


def projector_paste_hooks(model, donors, layers, position, bases_by_layer):
    """h <- h + Q Q^T (donor - h) at `position`, Q = orthonormal basis (rows) per layer."""
    hooks = []
    for layer in layers:
        donor = donors[layer].float()
        q = bases_by_layer[layer].float()

        def transform(selected, donor=donor, q=q):
            h = selected.float()
            d = donor.to(h.device).expand_as(h) - h
            qq = q.to(h.device)
            return h + (d @ qq.T) @ qq

        hooks.append((_resid_post_hook_name(layer), _make_intervention_hook(transform, [position], model.cfg.d_model)))
    return hooks


def donor_vector(model, text, entity, wanted, names):
    tokens = model.to_tokens(text, prepend_bos=False)
    seq = tokens[0].tolist()
    pos = None
    for variant in (" " + entity, entity):
        pos = find_last(seq, list(encode_variant(model.tokenizer, variant).ids))
        if pos is not None:
            break
    if pos is None:
        return None
    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    return {layer: cache[names[layer]][0, pos] for layer in BAND}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block49_4b_subspace_ladder")
    model, lenses = load_4b()
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / "block27_4b" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / "block35_4b_donor_paste" / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    # right-singular vectors of each J_l (directions in residual space ordered by transport gain)
    svd_v = {}
    for layer in BAND:
        j = lenses["J"].jacobians[layer].to(device=model.W_U.device, dtype=torch.float32)
        _, s, vh = torch.linalg.svd(j, full_matrices=False)
        svd_v[layer] = (vh, s)  # rows of vh: right-singular vectors, descending singular values
        print(f"[q51] L{layer} singular values: top {s[0]:.2f}, #16 {s[15]:.2f}, #256 {s[255]:.3f}, #1024 {s[1023]:.4f}, last {s[-1]:.2e}", flush=True)
    handle = (out_dir / "subspace_ladder.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s_id = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        if index not in best_pos or str(index) not in templates:
            continue
        position = best_pos[index]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        clean = final_logprobs(model, tokens)
        base = float(clean[s_id] - clean[a])
        donors = {"relation": donor_vector(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, wanted, names),
                  "N2": donor_vector(model, f"Fact: Consider the following: {r.item.swap_to}", r.item.swap_to, wanted, names)}
        vec = {k: {layer: lenses[k].lens_vectors(model, ids, layer).float() for layer in BAND} for k in ("J", "R")}
        # subspace bases per layer
        subspaces = {}
        subspaces["J_plane"] = {l: orthonormal(vec["J"][l]) for l in BAND}
        subspaces["R_plane"] = {l: orthonormal(vec["R"][l]) for l in BAND}
        subspaces["J_plane+R_t"] = {l: orthonormal(torch.cat([vec["J"][l], vec["R"][l][1:2]])) for l in BAND}
        subspaces["J_plane+R_s"] = {l: orthonormal(torch.cat([vec["J"][l], vec["R"][l][0:1]])) for l in BAND}
        subspaces["J+R_4d"] = {l: orthonormal(torch.cat([vec["J"][l], vec["R"][l]])) for l in BAND}
        for k in TOPK:
            subspaces[f"J_plane+top{k}"] = {l: orthonormal(torch.cat([vec["J"][l], svd_v[l][0][:k].to(vec["J"][l].device)])) for l in BAND}
        subspaces["J_plane+bottom(rest)"] = None  # handled as full minus top256 below
        full_eye = {l: None for l in BAND}
        # energy split of the complement (relation donor): how much of ||d_orth||^2 lies in top-k singular directions?
        energy = {}
        if donors["relation"] is not None:
            for k in TOPK:
                fr = []
                for l in BAND:
                    d = donors["relation"][l].float() - cache[names[l]][0, position].float()
                    qj = subspaces["J_plane"][l].to(d.device)
                    d_orth = d - (d @ qj.T) @ qj
                    vk = svd_v[l][0][:k].to(d.device)
                    fr.append(float(((d_orth @ vk.T) ** 2).sum() / (d_orth @ d_orth).clamp_min(1e-9)))
                energy[f"frac_orth_energy_top{k}"] = sum(fr) / len(fr)
            # random-subspace reference for k=256
            gen = torch.Generator(device="cpu").manual_seed(index)
            fr = []
            for l in BAND:
                d = donors["relation"][l].float() - cache[names[l]][0, position].float()
                qj = subspaces["J_plane"][l].to(d.device)
                d_orth = d - (d @ qj.T) @ qj
                rnd = orthonormal(torch.randn(256, d.shape[0], generator=gen).to(d.device))
                fr.append(float(((d_orth @ rnd.T) ** 2).sum() / (d_orth @ d_orth).clamp_min(1e-9)))
            energy["frac_orth_energy_random256"] = sum(fr) / len(fr)
        for dname, dvec in donors.items():
            if dvec is None:
                continue
            arms = {}
            for sname, sub in subspaces.items():
                if sub is None:
                    continue
                if dname == "N2" and sname not in ("J_plane", "J+R_4d", "J_plane+top256"):
                    continue
                arms[sname] = projector_paste_hooks(model, dvec, BAND, position, sub)
            if dname == "relation":
                # complement outside the top-256 singular directions: full paste minus (plane + top256)
                bottom = {}
                for l in BAND:
                    top = subspaces["J_plane+top256"][l]
                    bottom[l] = ("bottom", top)
                def bottom_hooks():
                    hooks = []
                    for l in BAND:
                        donor = dvec[l].float()
                        q = subspaces["J_plane+top256"][l].float()

                        def transform(selected, donor=donor, q=q):
                            h = selected.float()
                            d = donor.to(h.device).expand_as(h) - h
                            qq = q.to(h.device)
                            return h + d - (d @ qq.T) @ qq

                        hooks.append((_resid_post_hook_name(l), _make_intervention_hook(transform, [position], model.cfg.d_model)))
                    return hooks
                arms["outside_top256"] = bottom_hooks()
                arms["full"] = projector_paste_hooks(model, dvec, BAND, position, {l: torch.eye(model.cfg.d_model, device=model.W_U.device) for l in BAND})
            for arm, hooks in arms.items():
                logp = final_logprobs(model, tokens, hooks)
                top1 = int(logp.argmax().item())
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "donor": dname, "arm": arm, "position": position,
                    "delta_margin": float((logp[s_id] - logp[a]) - base),
                    "top1_is_swap": top1 == s_id, "top1_is_answer": top1 == a, "kl": kl_divergence(clean, logp),
                    **(energy if arm == "full" else {}),
                }) + "\n")
        if n % 10 == 0:
            print(f"[q51] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    print("[block46] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
