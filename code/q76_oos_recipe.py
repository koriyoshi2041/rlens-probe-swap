#!/usr/bin/env python
"""Block 76: OUT-OF-SAMPLE test of the editing recipe and of the pre-intervention predictors on
the v2 non-geographic items (block 55 protocol; 28 eligible).

Recipe (block 69/75): choose the edit position by where L23 h8 attends in the clean run (p_h8)
instead of by lens readability (p_best); apply the 2-D clamp there (x1, x4); optionally force
h8's attention onto p (donor-free routing) or transplant the donor's keys (donor routing).
Also the donor complement paste at both positions (neutral-carrier donor as in block 55).
Predictors recorded per item before any intervention: h8 attention mass at p_best and p_h8,
readability hits at both, the band-mean coordinate gap at both, and the unembedding cosine.
Everything is scored with the block-55 eligibility and the same prompt modes.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from q60_newitems_prediction import CARRIER, donor_vec
from q66_energy_of_arms import scaled_paste_hooks
from q69_key_value import capture, hook_name, patch_position
from q70_position_choice import clamp_at
from q75_donor_free_recipe import forcing_hooks, run
from rlens.data import SwapItem
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import resolve_item

BAND = list(range(8, 21))
ATTN_LAYERS = (19, 23)
DATA = pathlib.Path(__file__).resolve().parent / "data" / os.environ.get("MATS_NEWITEMS", "new_items_nongeo_v2.json")
RES_NEW = RESULTS_DIR / ("block55_newitems_v2" if "v2" in DATA.name else "block55_newitems")


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block76_oos_recipe")
    model = load_model()
    lens = load_lenses()["J"]
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    results55 = {r["name"]: r for r in json.loads((RES_NEW / "newitems.json").read_text(encoding="utf-8"))}
    items = [SwapItem(index=i, name=it["name"], category=it["category"], prompt=it["prompt"], intermediate=it["intermediate"],
                      answer=it["answer"], swap_to=it["swap_to"], swap_answer=it["swap_answer"]) for i, it in enumerate(payload["items"])]
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values()) | {hook_name(23, "hook_pattern")}
    handle = (out_dir / "oos_recipe.jsonl").open("w", encoding="utf-8")
    done = 0
    for item in items:
        res = results55.get(item.name)
        if not res or not res["eligible"]:
            continue
        r = resolve_item(model.tokenizer, item, res["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        T = int(tokens.shape[1])
        a, s = r.answer.first_id, r.swap_answer.first_id
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        pat = cache[hook_name(23, "hook_pattern")][0, 8, -1, :].float()
        p_h8 = int(pat[: T - 1].argmax().item())
        ranks, _ = rank_readout(model, lens, tokens, ids, BAND)
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        p_best = int(max(range(len(hits)), key=lambda p: (hits[p], p)))
        bases = {l: lens.lens_vectors(model, ids, l) for l in BAND}
        coords = {l: cache[names[l]].float() @ torch.linalg.pinv(bases[l].T.float()).T for l in BAND}
        gap = {tag: float(np.mean([abs(float(coords[l][0, p, 1] - coords[l][0, p, 0])) for l in BAND])) for tag, p in (("best", p_best), ("h8", p_h8))}
        donor = donor_vec(model, item.swap_to, set(names.values()), names)
        rec = {"name": item.name, "category": item.category, "n_prompt": T, "p_best": p_best, "p_h8": p_h8,
               "tok_best": model.tokenizer.decode([tokens[0, p_best].item()]), "tok_h8": model.tokenizer.decode([tokens[0, p_h8].item()]),
               "h8_mass": {"best": float(pat[p_best]), "h8": float(pat[p_h8]), "final": float(pat[-1])},
               "hits": {"best": int(hits[p_best]), "h8": int(hits[p_h8])}, "gap": gap,
               "cos": res["cos_entitydiff_answerdiff"], "band_clamp_flip_block55": res["clamp_J_flip"], "band_clamp_dm_block55": res["clamp_J_delta_margin"],
               "arms": {}}
        arms = {"clean": []}
        for tag, p in (("h8", p_h8), ("best", p_best)):
            c4 = clamp_at(model, bases, coords, [p], 4.0)
            arms[f"clamp@1_{tag}"] = clamp_at(model, bases, coords, [p], 1.0)
            arms[f"clamp@4_{tag}"] = c4
            arms[f"F[h8@23,1]+clamp@4_{tag}"] = forcing_hooks(p, "h8@23", 1.0) + c4
            arms[f"F[all@23,1]+clamp@4_{tag}"] = forcing_hooks(p, "all@23", 1.0) + c4
            if donor is not None:
                S = capture(model, tokens, scaled_paste_hooks(model, donor, BAND, p, 0.25, bases, "orth"))
                arms[f"K<-donor+clamp@4_{tag}"] = [(hook_name(l, "hook_k"), patch_position(S[hook_name(l, "hook_k")], p)) for l in ATTN_LAYERS] + c4
                arms[f"orth@0.25_{tag}"] = scaled_paste_hooks(model, donor, BAND, p, 0.25, bases, "orth")
                arms[f"full@0.25_{tag}"] = scaled_paste_hooks(model, donor, BAND, p, 0.25, None, "full")
        arms["clamp@1_all"] = clamp_at(model, bases, coords, list(range(T)), 1.0)
        arms["clamp@4_all"] = clamp_at(model, bases, coords, list(range(T)), 4.0)
        results = {}
        for arm, hooks in arms.items():
            p = p_h8 if arm.endswith("_h8") or arm.endswith("_all") else p_best
            results[arm] = run(model, tokens, hooks, p, s, a)
        for arm in results:
            results[arm]["delta_margin"] = results[arm]["margin"] - results["clean"]["margin"]
        rec["arms"] = results
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1
        print(f"[q76] {item.name} p_best={p_best} p_h8={p_h8} clamp@4_best={results['clamp@4_best']['delta_margin']:+.2f} clamp@4_h8={results['clamp@4_h8']['delta_margin']:+.2f} F+clamp_h8={results['F[h8@23,1]+clamp@4_h8']['delta_margin']:+.2f}", flush=True)
    handle.close()
    print(f"[block76] done ({done} items)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
