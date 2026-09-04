#!/usr/bin/env python
"""Block 74: the right null for "the lens plane is read": random-TOKEN lens planes; and a symmetric head search.

Audit 3 (M-3): the final-margin gradient g_l = d(logp_s - logp_a)/dh_l and the lens plane
span{J_l^T W_U[:,int], J_l^T W_U[:,swap]} both live in the row space of J_l^T, so a random 2-D
plane of R^4096 (share 0.0005) is the wrong null for "the plane holds 1.4-5.9% of the
sensitivity". The right null: lens planes of OTHER token pairs. Per item, at the bridge
position and every band layer:
  share of ||g_l||^2 in the item's own J plane;
  the same in 20 planes built from other items' (intermediate, swap_to) pairs and in 20 planes
  from random vocabulary token pairs;
  the donor-difference energy share in the own plane vs the two nulls (audit: "富集 48 倍").
Also (audit MINOR: asymmetric head search) a symmetric transport-head search on BOTH models:
for every full-attention layer of the model, every head's attention from the final position
onto the bridge position under clean and under the complement paste (orth@0.25).
MATS_MODEL=4b runs the same on Qwen3.5-4B (results dir block74_4b_...).
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
from q67_why_plane_ignored import margin_gradients, share
from rlens.data import load_items
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import encode_variant

BAND = list(range(8, 21))
FOURB = os.environ.get("MATS_MODEL", "9b") == "4b"
TEMPLATES = pathlib.Path(__file__).resolve().parent / "data" / "single_hop_templates.json"
N_NULL = 20


def main() -> int:
    out_dir = results_dir("block74_4b_random_token_planes" if FOURB else "block74_random_token_planes")
    if FOURB:
        model, lenses = load_4b()
    else:
        model = load_model()
        lenses = load_lenses()
    for p in model.parameters():
        p.requires_grad_(False)
    lens = lenses["J"]
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
    attn_layers = [l for l in range(model.cfg.n_layers) if f"blocks.{l}.attn.hook_pattern" in model.hook_dict]
    pattern_names = {f"blocks.{l}.attn.hook_pattern" for l in attn_layers}
    print(f"[q74] full-attention layers: {attn_layers}", flush=True)
    # token pairs of all eligible items (for the "other items" null)
    pairs = {}
    for index in eligible:
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        pairs[index] = (r.src.first_id, r.tgt.first_id)
    vocab = model.cfg.d_vocab_out if hasattr(model.cfg, "d_vocab_out") else model.W_U.shape[1]
    handle = (out_dir / "random_token_planes.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        if index not in best_pos or str(index) not in templates:
            continue
        r = resolve_item(model.tokenizer, items[index], modes[index]["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
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
            gen = torch.Generator(device="cpu").manual_seed(4000 + index)
            others = [pairs[i] for i in eligible if i != index]
            pick = torch.randperm(len(others), generator=gen)[:N_NULL].tolist()
            other_pairs = [others[k] for k in pick]
            rand_pairs = [(int(x), int(y)) for x, y in torch.randint(0, int(vocab), (N_NULL, 2), generator=gen).tolist()]
            rec = {"index": index, "name": r.item.name, "position": position, "layers": []}
            bases = {}
            for layer in BAND:
                g = grads[layer][position]
                d = dcache[names[layer]][0, dpos].float() - acts[layer][position]
                own = lens.lens_vectors(model, ids, layer).float()
                bases[layer] = own
                q_own = orthonormal(own)
                row = {"layer": layer, "share_own": share(g, q_own), "dshare_own": share(d, q_own)}
                for tag, plist in (("other", other_pairs), ("rand", rand_pairs)):
                    sh, dsh = [], []
                    for (t1, t2) in plist:
                        q = orthonormal(lens.lens_vectors(model, [t1, t2], layer).float())
                        sh.append(share(g, q)); dsh.append(share(d, q))
                    row[f"share_{tag}"] = sh; row[f"dshare_{tag}"] = dsh
                rec["layers"].append(row)
            # symmetric head search
            orth = scaled_paste_hooks(model, {l: dcache[names[l]][0, dpos] for l in BAND}, BAND, position, 0.25, bases, "orth")
            heads = {}
            for arm, hooks in (("clean", []), ("orth@0.25", orth)):
                with model.hooks(fwd_hooks=hooks):
                    _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in pattern_names)
                heads[arm] = {str(l): cache[f"blocks.{l}.attn.hook_pattern"][0, :, -1, position].float().tolist() for l in attn_layers}
            rec["heads"] = heads
            handle.write(json.dumps(rec) + "\n")
        if n % 10 == 0:
            print(f"[q74] {n + 1}/{len(eligible)} share_own L12 {rec['layers'][4]['share_own']:.4f} other-null median {sorted(rec['layers'][4]['share_other'])[N_NULL // 2]:.4f}", flush=True)
    handle.close()
    print("[block74] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
