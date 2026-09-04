#!/usr/bin/env python
"""Block 04: the experiments that turn the corrections into positive contributions.

1. cross_readout  -- after a swap, what does the band actually read? Cross-lens
                     (R reads J's intervention and vice versa) so no lens grades itself.
2. layer_profile  -- remove / install / swap at every single layer: where does erasing
                     help, where does it hurt?
3. width_sweep    -- energy growth vs band width and alpha: the compounding law.
4. knowledge2     -- reruns the single-hop check with the six repaired templates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.readout import rank_readout
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item

PROFILE_LAYERS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
WIDTH_STARTS = 8
WIDTHS = [1, 2, 4, 8, 13]
WIDTH_ALPHAS = [0.5, 1.0, 1.25, 1.5, 2.0]
WIDTH_ITEMS = 8


@torch.inference_mode()
def cross_readout(model, lenses, resolved, eligible, out_dir):
    """Does the swap change what the band reads, and does the reader lens matter?"""
    band = cfg.BANDS[cfg.PRIMARY_BAND]
    handle = (out_dir / "cross_readout.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        tracked = r.tracked_ids  # src, tgt, answer, swap_answer
        arms = {"clean": None}
        for kind in ("J", "R"):
            for mode in ("swap", "install"):
                arms[f"{kind}_{mode}"] = (kind, mode)
        for arm, spec in arms.items():
            hooks = []
            if spec is not None:
                kind, mode = spec
                for layer in band:
                    basis = lenses[kind].lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                    hooks += coordinate_map_hooks(model, basis, [layer], mode=mode, alpha=1.0)
            for reader in ("J", "R"):
                with model.hooks(fwd_hooks=hooks) if hooks else torch.no_grad():
                    ranks, _ = rank_readout(model, lenses[reader], tokens, tracked, band)
                # ranks: [len(band)+1, pos, 4]; best over positions per layer
                best = ranks[:-1].min(dim=1).values.numpy()  # [len(band), 4]
                handle.write(json.dumps({
                    "index": index, "name": r.item.name, "arm": arm, "reader": reader,
                    "layers": band,
                    "best_rank_intermediate": best[:, 0].tolist(),
                    "best_rank_swap_to": best[:, 1].tolist(),
                    "best_rank_answer": best[:, 2].tolist(),
                    "best_rank_swap_answer": best[:, 3].tolist(),
                }) + "\n")
        if n % 10 == 0:
            print(f"[cross_readout] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()


@torch.inference_mode()
def layer_profile(model, lenses, resolved, eligible, out_dir):
    """remove vs install vs swap, one layer at a time."""
    handle = (out_dir / "layer_profile.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        for layer in PROFILE_LAYERS:
            for kind, lens in lenses.items():
                basis = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                for mode in ("swap", "remove", "install"):
                    stats = EnergyStats()
                    hooks = coordinate_map_hooks(model, basis, [layer], mode=mode, alpha=1.0, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    handle.write(json.dumps({
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "layer": layer, "lens": kind, "mode": mode,
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                        "kl": kl_divergence(clean, logp), "dh_norm": float(stats.total_dh2 ** 0.5),
                    }) + "\n")
        if n % 10 == 0:
            print(f"[layer_profile] {n + 1}/{len(eligible)} elapsed={time.time() - t0:.0f}s", flush=True)
    handle.close()


@torch.inference_mode()
def width_sweep(model, lenses, resolved, eligible, out_dir):
    """Energy growth as a function of band width and alpha: |1-2 alpha|^width."""
    rows = []
    for index in eligible[:WIDTH_ITEMS]:
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        for width in WIDTHS:
            layers = list(range(WIDTH_STARTS, WIDTH_STARTS + width))
            for kind, lens in lenses.items():
                bases = {layer: lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer) for layer in layers}
                for alpha in WIDTH_ALPHAS:
                    stats = EnergyStats()
                    hooks = []
                    for layer in layers:
                        hooks += coordinate_map_hooks(model, bases[layer], [layer], mode="swap", alpha=alpha, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    per_layer = stats.as_rows()
                    rows.append({
                        "index": index, "name": r.item.name, "width": width, "lens": kind, "alpha": alpha,
                        "total_dh2": stats.total_dh2,
                        "first_layer_dh2": per_layer[0]["sum_dh2"], "last_layer_dh2": per_layer[-1]["sum_dh2"],
                        "growth_last_over_first": per_layer[-1]["sum_dh2"] / per_layer[0]["sum_dh2"],
                        "max_rel_last": per_layer[-1]["max_rel"],
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "kl": kl_divergence(clean, logp),
                        "top1_is_swap": int(logp.argmax().item()) == s,
                    })
    (out_dir / "width_sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n=== band width vs alpha: median per-layer energy growth (last/first) ===")
    print(f"{'width':>6s} " + " ".join(f"a={a:<6}" for a in WIDTH_ALPHAS))
    for width in WIDTHS:
        cells = []
        for alpha in WIDTH_ALPHAS:
            sub = [r["growth_last_over_first"] for r in rows if r["width"] == width and r["alpha"] == alpha and r["lens"] == "J"]
            cells.append(f"{np.median(sub):8.3g}" if sub else "       -")
        theory = [f"{abs(1 - 2 * a) ** (2 * (width - 1)):8.3g}" for a in WIDTH_ALPHAS]
        print(f"{width:6d} " + " ".join(cells))
        print(f"{'theory':>6s} " + " ".join(theory))
    return rows


@torch.inference_mode()
def knowledge2(model, items, eligible, out_dir):
    from rlens.stages import knowledge

    rows = knowledge.run(model, items, eligible, out_dir)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="width,profile,cross,knowledge")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",")]
    out_dir = results_dir("block04")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    if "width" in stages:
        width_sweep(model, lenses, resolved, eligible, out_dir)
    if "knowledge" in stages:
        knowledge2(model, items, eligible, out_dir)
    if "profile" in stages:
        layer_profile(model, lenses, resolved, eligible, out_dir)
    if "cross" in stages:
        cross_readout(model, lenses, resolved, eligible, out_dir)
    print("[block04] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
