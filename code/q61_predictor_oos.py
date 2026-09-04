#!/usr/bin/env python
"""Block 56: out-of-sample test of the PRE-INTERVENTION predictor of editability.

Block 37 found, on the original 59 items, that L23 head 8's clean attention from the final
position onto the best bridge position predicts the clamp's Δmargin (Spearman +0.43),
whereas the unembedding cosine (block 55) does not generalise. This block computes, for
the v2 non-geographic items that are eligible, the same clean-run quantities BEFORE any
intervention: L23.h8 attention to the best bridge position, the summed L23 attention of all
heads to it, the intermediate's best lens rank in the band, and the coordinate gap; then
reports their Spearman with the block-55 clamp Δmargin / flip. No new interventions.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch
from transformer_lens.tools.analysis.jacobian_lens import _resid_post_hook_name

from rlens.data import SwapItem
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import resolve_item

BAND = list(range(8, 21))
DATA = pathlib.Path(__file__).resolve().parent / "data" / os.environ.get("MATS_NEWITEMS", "new_items_nongeo_v2.json")
RES_NEW = RESULTS_DIR / ("block55_newitems_v2" if "v2" in DATA.name else "block55_newitems")


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.inference_mode()
def main() -> int:
    out_dir = results_dir("block56_predictor_oos")
    model = load_model()
    lens = load_lenses()["J"]
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    results = {r["name"]: r for r in json.loads((RES_NEW / "newitems.json").read_text(encoding="utf-8"))}
    items = [SwapItem(index=i, name=it["name"], category=it["category"], prompt=it["prompt"], intermediate=it["intermediate"],
                      answer=it["answer"], swap_to=it["swap_to"], swap_answer=it["swap_answer"]) for i, it in enumerate(payload["items"])]
    names = {layer: _resid_post_hook_name(layer) for layer in BAND}
    wanted = set(names.values()) | {"blocks.23.attn.hook_pattern"}
    rows = []
    for item in items:
        res = results.get(item.name)
        if not res or not res["eligible"]:
            continue
        r = resolve_item(model.tokenizer, item, res["mode"])
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        ids = [r.src.first_id, r.tgt.first_id]
        _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in wanted)
        ranks, _ = rank_readout(model, lens, tokens, ids, BAND)
        hits = (ranks[:-1, :, 0] <= 10).sum(dim=0).numpy()
        bp = int(max(range(len(hits)), key=lambda p: (hits[p], p)))
        pat = cache["blocks.23.attn.hook_pattern"][0, :, -1, :].float()  # [heads, key]
        gaps = []
        for layer in BAND:
            basis = lens.lens_vectors(model, ids, layer)
            pinv = torch.linalg.pinv(basis.T.float())
            c = cache[names[layer]][0, bp].float() @ pinv.T
            gaps.append(float((c[1] - c[0]).abs()))
        rows.append({
            "name": item.name, "best_pos": bp, "hits": int(hits[bp]),
            "h8_attn_to_bridge": float(pat[8, bp]), "h9_attn_to_bridge": float(pat[9, bp]), "all_heads_attn_to_bridge": float(pat[:, bp].sum()),
            "best_rank_intermediate": int(ranks[:-1, :, 0].min()), "coord_gap_mean": float(np.mean(gaps)),
            "cos": res["cos_entitydiff_answerdiff"], "clamp_J_delta_margin": res["clamp_J_delta_margin"], "clamp_J_flip": res["clamp_J_flip"],
            "paste_full_flip": res.get("paste_full_flip"),
        })
    (out_dir / "predictors.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    dm = [r["clamp_J_delta_margin"] for r in rows]; fl = [float(r["clamp_J_flip"]) for r in rows]
    print(f"[q61] n={len(rows)}")
    for key in ("h8_attn_to_bridge", "h9_attn_to_bridge", "all_heads_attn_to_bridge", "best_rank_intermediate", "coord_gap_mean", "cos", "hits"):
        x = [r[key] for r in rows]
        print(f"  Spearman({key:26s}, ΔM) = {spearman(x, dm):+.2f}   (flip) = {spearman(x, fl):+.2f}")
    print("[block56] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
