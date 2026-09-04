#!/usr/bin/env python
"""Exploration after the main run: alternative explanations and the remove/install decomposition.

A. Does the intervention direction itself already encode the answer? (Neel's
   "direct answer direction" alternative explanation, never tested until now.)
B. Are the R matrices simply lower-rank / smoother than J early on?
C. remove-only vs install-only vs full swap over all eligible items.
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

import numpy as np
import torch

from rlens import config_block02 as cfg
from rlens.data import load_items
from rlens.forward import final_logprobs, kl_divergence
from rlens.interventions import EnergyStats, coordinate_map_hooks
from rlens.model import load_lenses, load_model
from rlens.paths import RESULTS_DIR, results_dir
from rlens.stages.common import eligible_hybrid, hybrid_rows, resolve_item
from rlens.tokens import rank_of

PROBE_LAYERS = [3, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24]


@torch.inference_mode()
def part_a_direction_readout(model, lenses, items, resolved, eligible, out_dir) -> List[Dict]:
    """Read out the intervention direction (v_t - v_s) through each lens.

    Adding delta at layer l changes the final logits by ~ W_U^T J_l delta, so the
    ranking of that vector says which tokens the perturbation promotes *directly*,
    before any model computation. If swap_answer already ranks high there, a
    "targeted rewrite" is partly a direct logit push.
    """
    rows = []
    for kind, lens in lenses.items():
        for layer in PROBE_LAYERS:
            for index in eligible:
                r = resolved[index]
                vecs = lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer)
                direction = vecs[1] - vecs[0]  # effective push when the source is present
                matrix = lens.jacobians[layer].to(device=direction.device, dtype=torch.float32)
                logits = model.W_U.float().T @ (matrix @ direction)  # [d_vocab]
                rows.append(
                    {
                        "lens": kind,
                        "layer": layer,
                        "index": index,
                        "name": r.item.name,
                        "rank_swap_to": rank_of(logits, r.tgt.first_id),
                        "rank_intermediate": rank_of(logits, r.src.first_id),
                        "rank_swap_answer": rank_of(logits, r.swap_answer.first_id),
                        "rank_answer": rank_of(logits, r.answer.first_id),
                    }
                )
    (out_dir / "explore_direction_readout.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n=== A. lens readout of the intervention direction (v_t - v_s): median rank over items ===")
    print(f"{'lens':5s} {'L':>3s} {'swap_to':>9s} {'intermed':>9s} {'swap_ans':>9s} {'answer':>8s}  {'%swap_ans<=100':>15s}")
    for kind in lenses:
        for layer in PROBE_LAYERS:
            sub = [r for r in rows if r["lens"] == kind and r["layer"] == layer]
            med = lambda k: float(np.median([r[k] for r in sub]))
            frac = float(np.mean([r["rank_swap_answer"] <= 100 for r in sub]))
            print(f"{kind:5s} {layer:3d} {med('rank_swap_to'):9.0f} {med('rank_intermediate'):9.0f} {med('rank_swap_answer'):9.0f} {med('rank_answer'):8.0f}  {frac:15.2f}")
    return rows


@torch.inference_mode()
def part_b_spectra(lenses, out_dir) -> List[Dict]:
    """Effective rank of each transport matrix: is R just a smoother operator early on?"""
    rows = []
    print("\n=== B. spectra of the transport matrices ===")
    print(f"{'L':>3s} {'J eff_rank':>11s} {'R eff_rank':>11s} {'J top1share':>12s} {'R top1share':>12s} {'J s1/s10':>9s} {'R s1/s10':>9s}")
    for layer in PROBE_LAYERS:
        stats = {}
        for kind, lens in lenses.items():
            matrix = lens.jacobians[layer].cuda().float()
            sv = torch.linalg.svdvals(matrix)
            p = (sv**2) / (sv**2).sum()
            stats[kind] = {
                "effective_rank": float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()),
                "top1_share": float(p[0].item()),
                "s1_over_s10": float((sv[0] / sv[9]).item()),
                "stable_rank": float(((sv**2).sum() / sv[0] ** 2).item()),
            }
            del matrix, sv, p
            torch.cuda.empty_cache()
        rows.append({"layer": layer, **{k: v for k, v in stats.items()}})
        print(f"{layer:3d} {stats['J']['effective_rank']:11.1f} {stats['R']['effective_rank']:11.1f} {stats['J']['top1_share']:12.4f} {stats['R']['top1_share']:12.4f} {stats['J']['s1_over_s10']:9.2f} {stats['R']['s1_over_s10']:9.2f}")
    (out_dir / "explore_spectra.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


@torch.inference_mode()
def part_c_decomposition(model, lenses, resolved, eligible, out_dir) -> List[Dict]:
    """swap = remove + install, exactly. Which half carries the behavioural effect?"""
    rows = []
    handle = (out_dir / "explore_decomposition.jsonl").open("w", encoding="utf-8")
    for n, index in enumerate(eligible):
        r = resolved[index]
        tokens = model.to_tokens(r.prompt, prepend_bos=False)
        clean = final_logprobs(model, tokens)
        a, s = r.answer.first_id, r.swap_answer.first_id
        for band_name, layers in cfg.BANDS.items():
            for kind, lens in lenses.items():
                bases = {layer: lens.lens_vectors(model, [r.src.first_id, r.tgt.first_id], layer) for layer in layers}
                for mode in ("swap", "remove", "install"):
                    stats = EnergyStats()
                    hooks = []
                    for layer in layers:
                        hooks += coordinate_map_hooks(model, bases[layer], [layer], mode=mode, alpha=1.0, stats=stats)
                    logp = final_logprobs(model, tokens, hooks)
                    top1 = int(logp.argmax().item())
                    row = {
                        "index": index, "name": r.item.name, "category": r.item.category,
                        "band": band_name, "lens": kind, "mode": mode,
                        "delta_logp_answer": float(logp[a] - clean[a]),
                        "delta_logp_swap_answer": float(logp[s] - clean[s]),
                        "delta_margin": float((logp[s] - logp[a]) - (clean[s] - clean[a])),
                        "top1_is_swap": top1 == s, "top1_is_answer": top1 == a,
                        "top1_str": model.tokenizer.decode([top1]),
                        "kl": kl_divergence(clean, logp), "total_dh2": stats.total_dh2,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if n % 10 == 0:
            print(f"[decomp] {n + 1}/{len(eligible)}", flush=True)
    handle.close()
    return rows


def main() -> int:
    out_dir = results_dir("block03_explore")
    model = load_model()
    lenses = load_lenses()
    items = load_items()
    clean_rows = json.loads((RESULTS_DIR / "block01" / "clean_rows.json").read_text(encoding="utf-8"))
    eligible = eligible_hybrid(clean_rows)
    modes = hybrid_rows(clean_rows)
    resolved = {i: resolve_item(model.tokenizer, items[i], modes[i]["mode"]) for i in eligible}
    part_a_direction_readout(model, lenses, items, resolved, eligible, out_dir)
    part_b_spectra(lenses, out_dir)
    part_c_decomposition(model, lenses, resolved, eligible, out_dir)
    print("\n[explore] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
