#!/usr/bin/env python
"""Block 63: why do 27 never-flipped items ignore the 2-D lens plane at the bridge position?

Block 62: at the single best bridge position the over-driven 2-D clamp (4x) injects a
median 118 energy into 27 of the 35 never-flipped items and the output does not move
(KL 0.04), while a tenth of the donor's off-plane difference (64 energy) rescues 11.
Offline (block 51 profile) those 27 have a small coordinate gap (1.8 vs 4.5 for flipped
items) and fewer readable band layers at that position (7 vs 12). Two readings:
  H-weak    the plane carries little entity-specific content there (small gap) but the
            plane direction itself is read downstream like anywhere else;
  H-unread  the downstream sensitivity at that position lies outside the plane: the
            direction the clamp pushes along is not what L17-24 reads for these items.
Test 1 (no intervention). g_l = d(logp_s - logp_a)/dh_l[pos], l in L8-20:
  sensitivity share in the J plane  ||P_J g_l||^2/||g_l||^2  (chance 2/4096 = 0.0005),
  the same for the R plane, a random 2-D plane, the R top-256 subspace and a random 256-D
  subspace (chance 0.0625); first-order predictions sum_l <g_l, delta_l> of clamp2d@4,
  orth@0.1, full@0.25; cos(g_l, d_l) for the donor difference d_l.
Test 2. Does the off-plane paste change WHERE the final position attends? L19/L23 full-
attention patterns from the final position onto the bridge position (L23 h8 and all-head
sums) under clean, clamp2d@4, orth@0.1, orth@0.25, full@0.25.
Predictions. H-unread: the in-plane sensitivity share of the 27 is at chance while
flipped/rescued items sit above it, and the first-order predictions already separate the
arms. If attention onto the bridge rises under orth but not under the clamp, the
complement also carries the key-side "attend to me" signal.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q30_4b_replication import load_4b
from q35_donor_paste import find_last
from q51_subspace_ladder import orthonormal
from q66_energy_of_arms import scaled_paste_hooks
from rlens.data import load_items
from rlens.forward import final_logprobs
from rlens.interventions import clamp_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
FOURB = os.environ.get("MATS_MODEL", "9b") == "4b"  # block67: same test on Qwen3.5-4B; full-attention layers discovered from the hook dict
ATTN_LAYERS = (19, 23)
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"


def margin_gradients(model, tokens, names, s, a):
    """Gradient of logp_s - logp_a at the final position w.r.t. every band residual (full sequence)."""
    captured = {}

    def grab(name):
        def fn(act, hook):
            if not act.requires_grad:
                act.requires_grad_(True)
            act.retain_grad()
            captured[name] = act
            return act
        return fn

    with torch.enable_grad():
        with model.hooks(fwd_hooks=[(names[layer], grab(names[layer])) for layer in BAND]):
            logits = model(tokens, return_type="logits")
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        (lp[s] - lp[a]).backward()
    grads = {layer: captured[names[layer]].grad[0].detach().float() for layer in BAND}
    acts = {layer: captured[names[layer]][0].detach().float() for layer in BAND}
    return grads, acts


def share(g, q):
    """Fraction of ||g||^2 inside the row-space of the orthonormal basis q [k, d]."""
    return float(((g @ q.T) ** 2).sum().item() / max(float((g ** 2).sum().item()), 1e-12))


def attention_onto(model, tokens, hooks, position, s, a, layers=ATTN_LAYERS):
    wanted = {f"blocks.{l}.attn.hook_pattern" for l in layers}
    with model.hooks(fwd_hooks=hooks):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
    lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
    out = {"margin": float(lp[s] - lp[a]), "top1_is_swap": int(lp.argmax().item()) == s}
    for l in layers:
        pat = cache[f"blocks.{l}.attn.hook_pattern"][0, :, -1, :].float()  # [heads, key]
        out[f"L{l}_heads_to_bridge"] = pat[:, position].tolist()
        out[f"L{l}_sum_to_bridge"] = float(pat[:, position].sum().item())
        out[f"L{l}_max_head_to_bridge"] = float(pat[:, position].max().item())
        out[f"L{l}_sum_to_last"] = float(pat[:, -1].sum().item())
        if l == 23:
            out["L23_h8_to_bridge"] = float(pat[8, position].item())
    return out


def main() -> int:
    out_dir = results_dir("block67_4b_why_plane_ignored" if FOURB else "block63_why_plane_ignored")
    if FOURB:
        model, lenses = load_4b()
    else:
        model = load_model()
        lenses = load_lenses()
    for p in model.parameters():
        p.requires_grad_(False)
    lens = lenses["J"]
    attn_layers = tuple(l for l in range(model.cfg.n_layers) if f"blocks.{l}.attn.hook_pattern" in model.hook_dict and l > BAND[-1] - 4) if FOURB else ATTN_LAYERS
    print(f"[q67] attention layers read: {attn_layers}", flush=True)
    items = load_items()
    templates = json.loads(TEMPLATES.read_text(encoding="utf-8"))
    clean_rows = json.loads((RESULTS_DIR / ("block27_4b" if FOURB else "block01") / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    best_pos = {}
    for line in (RESULTS_DIR / ("block35_4b_donor_paste" if FOURB else "block31_donor_paste") / "donor_paste.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["position_set"] == "best" and r["arm"] == "clamp2d":
            best_pos[r["index"]] = int(r["position"])
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values())
    svd_R = {}
    for layer in BAND:
        j = lenses["R"].jacobians[layer].to(device=model.W_U.device, dtype=torch.float32)
        _, _, vh = torch.linalg.svd(j, full_matrices=False)
        svd_R[layer] = orthonormal(vh[:256])
    handle = (out_dir / "why_plane_ignored.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        if index not in best_pos or str(index) not in templates:
            continue
        position = best_pos[index]
        grads, acts = margin_gradients(model, tokens, names, s, a)
        with torch.inference_mode():
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
            gen = torch.Generator(device="cpu").manual_seed(1000 + index)
            rec = {"index": index, "name": r.item.name, "position": position, "n_prompt": int(tokens.shape[1]), "layers": []}
            pred = {"clamp2d@4": 0.0, "orth@0.1": 0.0, "full@0.25": 0.0}
            bases = {}
            for layer in BAND:
                basis = lens.lens_vectors(model, ids, layer).float()
                bases[layer] = basis
                rbasis = lenses["R"].lens_vectors(model, ids, layer).float()
                g = grads[layer][position]
                g_last = grads[layer][-1]
                h = acts[layer][position]
                pinv = torch.linalg.pinv(basis.T)
                c = h @ pinv.T
                d_clamp4 = 4.0 * ((c[[1, 0]] - c) @ basis)
                d = donor[layer].float() - h
                in_plane = (d @ pinv.T) @ basis
                d_orth = d - in_plane
                qJ = orthonormal(basis)
                qR = orthonormal(rbasis)
                q_rand2 = orthonormal(torch.randn(2, model.cfg.d_model, generator=gen).to(g.device))
                q_rand256 = orthonormal(torch.randn(256, model.cfg.d_model, generator=gen).to(g.device))
                q_R256 = svd_R[layer].to(g.device)
                pred["clamp2d@4"] += float((g @ d_clamp4).item())
                pred["orth@0.1"] += float((g @ (0.1 * d_orth)).item())
                pred["full@0.25"] += float((g @ (0.25 * d)).item())
                rec["layers"].append({
                    "layer": layer, "grad_norm": float(g.norm().item()), "grad_norm_last": float(g_last.norm().item()),
                    "share_J": share(g, qJ), "share_R": share(g, qR), "share_rand2": share(g, q_rand2),
                    "share_R256": share(g, q_R256), "share_rand256": share(g, q_rand256),
                    "share_J_last": share(g_last, qJ),
                    "cos_g_d": float(torch.nn.functional.cosine_similarity(g[None], d[None]).item()),
                    "cos_g_dorth": float(torch.nn.functional.cosine_similarity(g[None], d_orth[None]).item()),
                    "cos_g_clamp": float(torch.nn.functional.cosine_similarity(g[None], d_clamp4[None]).item()),
                    "gap": float((c[1] - c[0]).abs().item()), "clamp4_energy": float((d_clamp4 ** 2).sum().item()),
                    "orth01_energy": float(((0.1 * d_orth) ** 2).sum().item()),
                })
            rec["pred"] = pred
            # attention onto the bridge under the arms
            coords = {l: acts[l][None] @ torch.linalg.pinv(bases[l].T).T for l in BAND}
            arms = {"clean": []}
            hooks = []
            for l in BAND:
                cc = coords[l][:, position:position + 1, :]
                target = cc + 4.0 * (cc[..., [1, 0]] - cc)
                hooks += clamp_hooks(model, bases[l], [l], {l: target}, exchange=False, positions=[position])
            arms["clamp2d@4"] = hooks
            arms["orth@0.1"] = scaled_paste_hooks(model, donor, BAND, position, 0.1, bases, "orth")
            arms["orth@0.25"] = scaled_paste_hooks(model, donor, BAND, position, 0.25, bases, "orth")
            arms["full@0.25"] = scaled_paste_hooks(model, donor, BAND, position, 0.25, None, "full")
            rec["attention"] = {arm: attention_onto(model, tokens, hooks, position, s, a, attn_layers) for arm, hooks in arms.items()}
            for arm in arms:
                rec["attention"][arm]["delta_margin"] = rec["attention"][arm]["margin"] - rec["attention"]["clean"]["margin"]
            handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q67] {n + 1}/{len(eligible)} pred {pred}", flush=True)
    handle.close()
    print("[block67] done" if FOURB else "[block63] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
