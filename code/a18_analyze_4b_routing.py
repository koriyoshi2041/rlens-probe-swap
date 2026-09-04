#!/usr/bin/env python3
"""Offline analysis of blocks 67 (4B gradient split + routing) and 68 (4B propagation). CPU only.
Usage: a18_analyze_4b_routing.py <results_root>"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402

BAND = list(range(8, 21))
SHOW = (8, 9, 10, 12, 15, 18, 20)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if pos and neg else float("nan")


def groups_4b(root):
    rep_dir = root / "block27_4b_band8_20" if (root / "block27_4b_band8_20" / "replication.jsonl").exists() else root / "block27_4b"
    rep = load_records(rep_dir / "replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    e65 = load_records(root / "block65_4b_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e65 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    resco = {r["index"] for r in e65 if r["arm"] == "orth@0.25" and r["top1_is_swap"]}
    return zero, resc4, resco


def block67(root):
    p = root / "block67_4b_why_plane_ignored" / "why_plane_ignored.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, resc4, resco = groups_4b(root)
    groups = {
        "flipped by 4B band clamp": [r for r in rows if r["index"] not in zero],
        "zero: clamp2d@4 rescues": [r for r in rows if r["index"] in zero and r["index"] in resc4],
        "zero: clamp2d@4 fails": [r for r in rows if r["index"] in zero and r["index"] not in resc4],
        "zero: orth@0.25 rescues": [r for r in rows if r["index"] in zero and r["index"] in resco],
    }
    print(f"===== block67: Qwen3.5-4B gradient sensitivity split at the bridge position (n={len(rows)}) =====")
    print(f"  {'group':26s} n  {'J':>7s} {'rand2':>7s} {'J/rand2':>8s} | {'R256':>7s} {'rand256':>8s} {'ratio':>6s} | {'gap':>5s} | clamp4 pred | orth0.1 pred")
    for g, rs in groups.items():
        if not rs:
            continue
        def bm(key):
            return [float(np.mean([q[key] for q in r["layers"]])) for r in rs]
        sj, s2, s256, sr256 = bm("share_J"), bm("share_rand2"), bm("share_R256"), bm("share_rand256")
        print(f"  {g:26s} {len(rs):2d} {med(sj):7.4f} {med(s2):7.4f} {med([a / max(b, 1e-9) for a, b in zip(sj, s2)]):8.1f} | {med(s256):7.3f} {med(sr256):8.3f} {med([a / max(b, 1e-9) for a, b in zip(s256, sr256)]):6.2f} | {med(bm('gap')):5.2f} | {med([r['pred']['clamp2d@4'] for r in rs]):11.2f} | {med([r['pred']['orth@0.1'] for r in rs]):12.2f}")
    # routing: find the (layer, head) whose attention onto the bridge rises most under orth@0.25 vs clean
    layers = sorted({int(k[1:].split("_")[0]) for k in rows[0]["attention"]["clean"] if k.endswith("_heads_to_bridge")})
    best = None
    for l in layers:
        n_heads = len(rows[0]["attention"]["clean"][f"L{l}_heads_to_bridge"])
        for h in range(n_heads):
            rise = med([r["attention"]["orth@0.25"][f"L{l}_heads_to_bridge"][h] - r["attention"]["clean"][f"L{l}_heads_to_bridge"][h] for r in rows])
            if best is None or rise > best[0]:
                best = (rise, l, h)
    rise, L, H = best
    print(f"  routing: attention layers {layers}; head with the largest median rise under orth@0.25: L{L} h{H} (+{rise:.3f})")
    arms = ("clean", "clamp2d@4", "orth@0.1", "orth@0.25", "full@0.25")
    for key_name, f in ((f"L{L} h{H} onto bridge", lambda r, arm: r["attention"][arm][f"L{L}_heads_to_bridge"][H]), (f"L{L} all heads onto bridge", lambda r, arm: r["attention"][arm][f"L{L}_sum_to_bridge"])):
        print(f"  [{key_name}]")
        for g, rs in groups.items():
            if not rs:
                continue
            print(f"    {g:26s} " + "  ".join(f"{arm} {med([f(r, arm) for r in rs]):.3f}" for arm in arms))
    # pre-intervention AUC for single-position clamp2d@4 success and for band-clamp flips
    lab4 = [r["index"] in resc4 or r["index"] not in zero for r in rows]  # clamp2d@4 flips (all items)
    e65 = load_records(root / "block65_4b_energy_of_arms" / "energy_of_arms.jsonl")
    c4 = {r["index"]: r["top1_is_swap"] for r in e65 if r["arm"] == "clamp2d@4"}
    lab4 = [c4.get(r["index"], False) for r in rows]
    labb = [r["index"] not in zero for r in rows]
    h_att = [r["attention"]["clean"][f"L{L}_heads_to_bridge"][H] for r in rows]
    gap = [float(np.mean([q["gap"] for q in r["layers"]])) for r in rows]
    fo = [r["pred"]["clamp2d@4"] for r in rows]
    print(f"  AUC predicting single-position clamp2d@4 flip: L{L}h{H} attention {auc(h_att, lab4):.2f} | gap {auc(gap, lab4):.2f} | first-order {auc(fo, lab4):.2f}")
    print(f"  AUC predicting band-clamp flip:               L{L}h{H} attention {auc(h_att, labb):.2f} | gap {auc(gap, labb):.2f} | first-order {auc(fo, labb):.2f}")
    zr = [r for r in rows if r["index"] in zero]
    for arm in ("clamp2d@4", "orth@0.25"):
        d_resc = [r["attention"][arm][f"L{L}_sum_to_bridge"] - r["attention"]["clean"][f"L{L}_sum_to_bridge"] for r in zr if r["attention"][arm]["top1_is_swap"]]
        d_not = [r["attention"][arm][f"L{L}_sum_to_bridge"] - r["attention"]["clean"][f"L{L}_sum_to_bridge"] for r in zr if not r["attention"][arm]["top1_is_swap"]]
        print(f"  Δ(L{L} attention onto bridge) under {arm:10s}: rescued med {med(d_resc):+.3f} (n={len(d_resc)}) vs not {med(d_not):+.3f} (n={len(d_not)})")


def block68(root):
    p = root / "block68_4b_propagation" / "propagation.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, _, _ = groups_4b(root)
    print(f"\n===== block68: Qwen3.5-4B propagation (n={len(rows)}) =====")
    subs = ("J_plane", "J+R256", "J+rand256", "full")
    print("  (a) multi-layer projector paste: injected/static per layer (median); flip | flipZ")
    for sub in subs:
        rs = [r["arms"][sub]["multi"] for r in rows]
        rz = [r["arms"][sub]["multi"] for r in rows if r["index"] in zero]
        ratios = [med([m["injected"][BAND.index(l)] / max(m["static"][BAND.index(l)], 1e-9) for m in rs]) for l in SHOW]
        print(f"  {sub:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f} " + " ".join(f"{x:4.2f}" for x in ratios) + f"   {med([sum(m['injected']) for m in rs]):8.1f} / {med([sum(m['static']) for m in rs]):8.1f}")
    print("  (b) paste at L8 only: realised share of the donor difference downstream (median); in-subspace version")
    for sub in subs:
        rs = [r["arms"][sub]["single_L8"] for r in rows]
        rz = [r["arms"][sub]["single_L8"] for r in rows if r["index"] in zero]
        f = [med([m["realised"][BAND.index(l)] for m in rs]) for l in SHOW]
        fs = [med([m["realised_sub"][BAND.index(l)] for m in rs]) for l in SHOW]
        print(f"  {sub:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f} " + " ".join(f"{x:4.2f}" for x in f) + "   | " + " ".join(f"{x:4.2f}" for x in fs))


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block67(root)
    block68(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
