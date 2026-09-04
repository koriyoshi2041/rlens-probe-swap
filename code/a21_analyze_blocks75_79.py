#!/usr/bin/env python3
"""Offline analysis of blocks 75 (donor-free recipe), 76 (OOS recipe + predictors), 77 (PCA necessity + growth
localisation), 78 (early-band routing), 79 (clean-run head ablation, 9B and 4B). CPU only.
Usage: a21_analyze_blocks75_79.py <results_root>"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402

BAND = list(range(8, 21))


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if pos and neg else float("nan")


def groups_9b(root):
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    return zero, resc4


def block75(root):
    p = root / "block75_donor_free_recipe" / "donor_free.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, resc4 = groups_9b(root)
    groups = {"all": rows, "flipped by band clamp": [r for r in rows if r["index"] not in zero], "zero-flip (35)": [r for r in rows if r["index"] in zero],
              "zero: clamp2d@4 fails (27)": [r for r in rows if r["index"] in zero and r["index"] not in resc4]}
    arms = [a for a in rows[0]["arms"] if a != "clean"]
    print("===== block75: donor-free recipe (9B) — flip | top1=answer | median ΔM | h8 attention onto the edit position =====")
    for g, rs in groups.items():
        print(f"  --- [{g}] n={len(rs)} ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs if arm in r["arms"]]
            if not a:
                continue
            print(f"    {arm:30s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ans {np.mean([q['top1_is_answer'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h8→p {med([q['h8_to_p'] for q in a]):.2f}")
    # paired: forcing + clamp vs clamp alone at p_h8, flippable
    fl = groups["flipped by band clamp"]
    for arm in ("F[h8@23,1]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "K<-donor+clamp@4_h8"):
        d = [r["arms"][arm]["delta_margin"] - r["arms"]["clamp@4_h8"]["delta_margin"] for r in fl if arm in r["arms"]]
        print(f"  flippable: {arm} − clamp@4_h8: median {med(d):+.2f}, positive in {np.mean([x > 0 for x in d]):.2f} (n={len(d)})")


def block76(root):
    p = root / "block76_oos_recipe" / "oos_recipe.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    print(f"\n===== block76: OUT-OF-SAMPLE (v2 non-geographic, n={len(rows)}) — flip | median ΔM =====")
    print(f"  p_h8 == p_best: {np.mean([r['p_h8'] == r['p_best'] for r in rows]):.2f}; block-55 band clamp flip {np.mean([r['band_clamp_flip_block55'] for r in rows]):.2f}")
    for arm in [a for a in rows[0]["arms"] if a != "clean"]:
        a = [r["arms"][arm] for r in rows if arm in r["arms"]]
        if a:
            print(f"    {arm:28s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  (n={len(a)})")
    for target in ("clamp@4_best", "clamp@4_h8", "clamp@1_all"):
        lab = [r["arms"][target]["top1_is_swap"] for r in rows]
        y = [r["arms"][target]["delta_margin"] for r in rows]
        tag = "best" if target.endswith("best") else "h8"
        feats = {"h8 mass @site": [r["h8_mass"][tag] for r in rows], "gap @site": [r["gap"][tag] for r in rows], "hits @site": [r["hits"][tag] for r in rows],
                 "gap×h8 @site": [r["gap"][tag] * r["h8_mass"][tag] for r in rows], "cos (unembed)": [r["cos"] for r in rows]}
        from scipy.stats import spearmanr
        print(f"  predictors for {target} (pos={sum(lab)}/{len(lab)}): " + " | ".join(f"{k} AUC {auc(v, lab):.2f} ρ {spearmanr(v, y).correlation:+.2f}" for k, v in feats.items()))


def block77(root):
    p = root / "block77_growth_pca" / "growth_pca.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, _ = groups_9b(root)
    print(f"\n===== block77 (a): necessity ladder with PCA removal (full paste ×0.25 at p_best; n={len(rows)}) =====")
    for arm in rows[0]["necessity"]:
        a = [r["necessity"][arm] for r in rows]
        z = [r["necessity"][arm] for r in rows if r["index"] in zero]
        print(f"    {arm:20s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  flipZ {np.mean([q['top1_is_swap'] for q in z]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  KL {med([q['kl'] for q in a]):.2f}  E {med([q['energy'] for q in a]):.0f}")
    print("===== block77 (b): growth localisation after the L8-only paste — per-layer sublayer contribution to the donor difference (median) =====")
    for sub in rows[0]["growth"]:
        g = [r["growth"][sub] for r in rows]
        print(f"  [{sub}] flip {np.mean([q['top1_is_swap'] for q in g]):.2f}; realised at L8: on d8 {med([q['realised_L8_on_d8'] for q in g]):.2f}, on d20 {med([q['realised_L8_on_d20'] for q in g]):.2f}")
        layers = [q["layer"] for q in g[0]["layers"]]
        for key in ("attn_on_dl", "mlp_on_dl", "attn_on_d20", "mlp_on_d20", "realised_d20"):
            vals = [med([q["layers"][i][key] for q in g]) for i in range(len(layers))]
            print(f"      {key:13s} " + " ".join(f"L{l}:{v:+.3f}" for l, v in zip(layers, vals)))
        sa = [sum(q["layers"][i]["attn_on_d20"] for i in range(len(layers))) for q in g]
        sm = [sum(q["layers"][i]["mlp_on_d20"] for i in range(len(layers))) for q in g]
        print(f"      sum over L9-20 on d20: attention {med(sa):+.3f}  mlp {med(sm):+.3f}   (realised L20 on d20 {med([q['layers'][-1]['realised_d20'] for q in g]):.3f})")


def block78(root):
    p = root / "block78_early_band_routing" / "early_routing.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    print(f"\n===== block78: early band (L3-8, all positions) J vs R — routing (n={len(rows)}) =====")
    print(f"  {'arm':16s} {'flip':>5s} {'ΔM':>7s} {'h8→best':>8s} {'h8→h8pos':>9s} {'h8→final':>9s} {'L23Σbest':>9s} {'L19Σbest':>9s} {'argmax==best':>12s}")
    for arm in rows[0]["arms"]:
        a = [r["arms"][arm] for r in rows]
        print(f"  {arm:16s} {np.mean([q['top1_is_swap'] for q in a]):5.2f} {med([q['delta_margin'] for q in a]):+7.2f} {med([q['h8_best'] for q in a]):8.3f} {med([q['h8_h8pos'] for q in a]):9.3f} {med([q['h8_final'] for q in a]):9.3f} {med([q['L23_sum_best'] for q in a]):9.2f} {med([q['L19_sum_best'] for q in a]):9.2f} {np.mean([q['h8_argmax'] == r['p_best'] for q, r in zip(a, rows)]):12.2f}")
    for sc in ("1", "2"):
        d = [r["arms"][f"R@{sc}"]["delta_margin"] - r["arms"][f"J@{sc}"]["delta_margin"] for r in rows]
        dh = [r["arms"][f"R@{sc}"]["h8_best"] - r["arms"][f"J@{sc}"]["h8_best"] for r in rows]
        dkv = [r["arms"][f"KV<-R@{sc}_best"]["delta_margin"] - r["arms"][f"KV<-J@{sc}_best"]["delta_margin"] for r in rows]
        print(f"  paired R−J @{sc}: ΔM {med(d):+.2f} (R>J in {np.mean([x > 0 for x in d]):.2f}); h8→best {med(dh):+.3f}; KV-transplant ΔM {med(dkv):+.2f}")


def block79(root, tag):
    p = root / f"block79{tag}_clean_head_ablation" / "head_ablation.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    print(f"\n===== block79{tag}: clean-run mean ablation of L23 heads at the final position (n items={len({r['index'] for r in rows})}) =====")
    print(f"  {'arm':10s} {'two-hop acc':>12s} {'Δlogp':>7s} | {'single-hop acc':>15s} {'Δlogp':>7s}")
    for arm in ("h8", "h9", "h0", "h8h9h0", "rand3@23", "all@23", "all@19"):
        line = f"  {arm:10s}"
        for kind in ("two_hop", "single_hop"):
            s = [r for r in rows if r["arm"] == arm and r["prompt_kind"] == kind and r["clean_correct"]]
            line += f" {np.mean([r['still_correct'] for r in s]):12.2f} {med([r['delta_logp_answer'] for r in s]):+7.2f} |"
        print(line)
    for kind in ("two_hop", "single_hop"):
        s = [r for r in rows if r["arm"] == "h8" and r["prompt_kind"] == kind]
        print(f"  clean accuracy {kind}: {np.mean([r['clean_correct'] for r in s]):.2f}")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block75(root)
    block76(root)
    block77(root)
    block78(root)
    block79(root, "")
    block79(root, "_4b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
