#!/usr/bin/env python
"""Block 72: is "the lens subspace is relayed" more than "Jacobian top directions have large effect"?

Block 64/68: pasting the donor difference through J plane + R top-256 propagates (single L8
paste realises 33-50% of the donor difference by L20) while a random 256-D subspace does not.
Objection: the top right-singular directions of the Jacobian are by construction the
directions with the largest downstream effect, so this could be tautological. Non-trivial
comparisons at matched dimension (256 + the J plane), same protocol as block 64:
  J+R256      top-256 right-singular directions of R_l (reference)
  J+J256      top-256 of J_l
  J+WU256     top-256 eigen-directions of W_U W_U^T (what the unembedding reads; no downstream
              computation involved)
  J+PCA256    top-256 principal components of the residual stream at layer l over all positions
              of the 59 prompts (the high-variance subspace)
  J+entdiff   the span of the OTHER 58 items' donor differences at layer l (leave-one-out, <=58
              dims): a task-specific "entity difference" subspace
  J+rand58    random 58-D control for entdiff
  J+rand256   random 256-D control (reference)
For each: multi-layer projector paste (flip, injected energy) and single-L8 paste (flip, realised
share of the donor difference at each band layer). If PCA256 or WU256 relay as well as R256, the
lens spectrum is not special beyond "large-effect directions"; if entdiff (58 dims) relays and
flips as well as R256, the missing content lives in a shared entity-difference subspace.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q35_donor_paste import find_last
from q51_subspace_ladder import orthonormal
from q68_propagation import recording_projector_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
K = 256


def donor_at(model, text, entity, wanted, names):
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
    return {layer: cache[names[layer]][0, pos].float() for layer in BAND}


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block72_subspace_controls")
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
    dev = model.W_U.device
    # subspaces that do not depend on the item
    svd = {}
    for kind in ("J", "R"):
        svd[kind] = {}
        for layer in BAND:
            j = lenses[kind].jacobians[layer].to(device=dev, dtype=torch.float32)
            _, _, vh = torch.linalg.svd(j, full_matrices=False)
            svd[kind][layer] = vh[:K]
    wu = model.W_U.float()
    evals, evecs = torch.linalg.eigh(wu @ wu.T)  # ascending
    wu_top = evecs[:, -K:].T.contiguous()  # [K, d]
    # pass 1: clean residuals at all positions (for PCA) and per-item donor differences (for entdiff)
    resid = {layer: [] for layer in BAND}
    info = {}
    for index in eligible:
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        for layer in BAND:
            resid[layer].append(cache[names[layer]][0].float())
        donor = donor_at(model, templates[str(index)].format(r.item.swap_to), r.item.swap_to, wanted, names)
        if donor is None:
            continue
        p = best_pos[index]
        info[index] = {"tokens": tokens, "r": r, "clean": {l: cache[names[l]][0, p].float() for l in BAND}, "donor": donor, "position": p}
    pca = {}
    for layer in BAND:
        x = torch.cat(resid[layer])
        x = x - x.mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        pca[layer] = vh[:K]
    diffs = {l: {i: info[i]["donor"][l] - info[i]["clean"][l] for i in info} for l in BAND}
    print(f"[q72] subspaces ready; items {len(info)}; residual rows per layer {sum(t.shape[0] for t in resid[BAND[0]])}", flush=True)
    handle = (out_dir / "subspace_controls.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(sorted(info)):
        it = info[index]
        r, tokens, position, donor = it["r"], it["tokens"], it["position"], it["donor"]
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        clean = final_logprobs(model, tokens)
        base = float(clean[s] - clean[a])
        bases = {l: lens.lens_vectors(model, ids, l).float() for l in BAND}
        gen = torch.Generator(device="cpu").manual_seed(9000 + index)
        others = [i for i in info if i != index]
        ent = {l: torch.stack([diffs[l][i] for i in others]) for l in BAND}
        n_ent = len(others)
        subspaces = {
            "J+R256": {l: orthonormal(torch.cat([bases[l], svd["R"][l]])) for l in BAND},
            "J+J256": {l: orthonormal(torch.cat([bases[l], svd["J"][l]])) for l in BAND},
            "J+WU256": {l: orthonormal(torch.cat([bases[l], wu_top])) for l in BAND},
            "J+PCA256": {l: orthonormal(torch.cat([bases[l], pca[l]])) for l in BAND},
            "J+entdiff": {l: orthonormal(torch.cat([bases[l], ent[l]])) for l in BAND},
            "J+rand58": {l: orthonormal(torch.cat([bases[l], torch.randn(n_ent, model.cfg.d_model, generator=gen).to(dev)])) for l in BAND},
            "J+rand256": {l: orthonormal(torch.cat([bases[l], torch.randn(K, model.cfg.d_model, generator=gen).to(dev)])) for l in BAND},
        }
        d = {l: donor[l] - it["clean"][l] for l in BAND}
        rec = {"index": index, "name": r.item.name, "position": position, "n_entdiff": n_ent, "arms": {}}
        for sub_name, sub in subspaces.items():
            static = [float((((d[l] @ sub[l].T) @ sub[l]) ** 2).sum().item()) for l in BAND]
            store = {}
            logp = final_logprobs(model, tokens, recording_projector_hooks(model, donor, BAND, position, sub, store))
            multi = {"delta_margin": float((logp[s] - logp[a]) - base), "top1_is_swap": int(logp.argmax().item()) == s,
                     "injected": [store[l] for l in BAND], "static": static}
            hooks = recording_projector_hooks(model, donor, [BAND[0]], position, sub, {})
            with model.hooks(fwd_hooks=hooks):
                logits, icache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
            lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            realised = []
            for l in BAND:
                delta = icache[names[l]][0, position].float() - it["clean"][l]
                realised.append(float((delta @ d[l]).item() / max(float((d[l] ** 2).sum().item()), 1e-9)))
            single = {"delta_margin": float((lp[s] - lp[a]) - base), "top1_is_swap": int(lp.argmax().item()) == s, "realised": realised}
            rec["arms"][sub_name] = {"multi": multi, "single_L8": single}
        handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q72] {n + 1}/{len(info)} " + " ".join(f"{k}:{v['multi']['top1_is_swap']:d}/{v['single_L8']['realised'][-1]:.2f}" for k, v in rec["arms"].items()), flush=True)
    handle.close()
    print("[block72] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
