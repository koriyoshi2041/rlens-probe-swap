#!/usr/bin/env python3
"""Offline analysis of block 73 (transport-layer inputs at the bridge position) and block 74 (random-token
lens-plane null + symmetric head search, 9B and 4B). CPU only.  Usage: a21_analyze_blocks73_74.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def block73(root):
    p = root / "block73_routing_all_layers" / "routing_all_layers.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    check = json.load(open(root / "block73_routing_all_layers" / "self_check.json"))
    print(f"===== block73: transport layers L17-24 fed from the bridge position (n={len(rows)}; identity_ok={check['identity_ok']}) =====")
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    groups = {"all": rows, "flipped by band clamp": [r for r in rows if r["index"] not in zero], "zero: clamp2d@4 fails (27)": [r for r in rows if r["index"] in zero and r["index"] not in resc4]}
    arms = ["clean", "clamp2d@4", "orth@0.25", "orth-restore", "K19,23", "KV19,23", "lin_in", "K19,23+lin_in", "KV19,23+lin_in", "transport<-clamp",
            "K19,23+clamp", "KV19,23+clamp", "lin_in+clamp", "K19,23+lin_in+clamp", "KV19,23+lin_in+clamp", "PAT_h8@23+clamp"]
    for g, rs in groups.items():
        print(f"  --- [{g}] n={len(rs)}: flip | rescued count | median ΔM | h8→bridge ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs if arm in r["arms"]]
            if not a:
                continue
            print(f"    {arm:22s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  n {sum(q['top1_is_swap'] for q in a):2d}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h8 {med([q['h8'] for q in a]):.3f}")
    # share of the paste effect flowing through the transport layers' inputs at p
    sh = [r["arms"]["KV19,23+lin_in"]["delta_margin"] / r["arms"]["orth@0.25"]["delta_margin"] for r in rows if r["arms"]["orth@0.25"]["delta_margin"] > 1]
    rest = [r["arms"]["orth-restore"]["delta_margin"] / r["arms"]["orth@0.25"]["delta_margin"] for r in rows if r["arms"]["orth@0.25"]["delta_margin"] > 1]
    print(f"  share of orth ΔM via transport inputs at p (KV19,23+lin_in / orth): median {med(sh):.2f}; residual with those inputs restored to clean: median {med(rest):.2f} (n={len(sh)})")
    fl = [r for r in rows if r["index"] not in zero]
    d = [r["arms"]["PAT_h8@23+clamp"]["delta_margin"] - r["arms"]["clamp2d@4"]["delta_margin"] for r in fl]
    print(f"  PAT_h8@23+clamp − clamp on flippable: >0 in {sum(x > 0 for x in d)}/{len(d)}, median {med(d):+.2f}; flips {np.mean([r['arms']['PAT_h8@23+clamp']['top1_is_swap'] for r in fl]):.2f} vs clamp {np.mean([r['arms']['clamp2d@4']['top1_is_swap'] for r in fl]):.2f}")


def block74(root, tag):
    d = root / f"block74{tag}_random_token_planes"
    p = d / "random_token_planes.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    print(f"\n===== block74{tag}: random-token lens-plane null (n={len(rows)}) =====")
    own = [np.mean([q["share_own"] for q in r["layers"]]) for r in rows]
    other = [np.mean([np.median(q["share_other"]) for q in r["layers"]]) for r in rows]
    rand = [np.mean([np.median(q["share_rand"]) for q in r["layers"]]) for r in rows]
    print(f"  gradient share: own plane {med(own):.4f} | other-item planes (median of 20) {med(other):.4f} | random-token planes {med(rand):.4f} | ratio own/other {med([a / max(b, 1e-9) for a, b in zip(own, other)]):.1f}x  own/rand {med([a / max(b, 1e-9) for a, b in zip(own, rand)]):.1f}x")
    down = [np.mean([q["dshare_own"] for q in r["layers"]]) for r in rows]
    dother = [np.mean([np.median(q["dshare_other"]) for q in r["layers"]]) for r in rows]
    drand = [np.mean([np.median(q["dshare_rand"]) for q in r["layers"]]) for r in rows]
    print(f"  donor-difference share: own {med(down):.4f} | other-item planes {med(dother):.4f} | random-token planes {med(drand):.4f} | ratio own/other {med([a / max(b, 1e-9) for a, b in zip(down, dother)]):.1f}x  own/rand {med([a / max(b, 1e-9) for a, b in zip(down, drand)]):.1f}x")
    # percentile of own share within the other-item null, per item
    pct = [np.mean([np.mean([o < q["share_own"] for o in q["share_other"]]) for q in r["layers"]]) for r in rows]
    print(f"  own gradient share above the other-item null: median percentile {med(pct):.2f}; items with median percentile >= 0.95: {sum(x >= 0.95 for x in pct)}/{len(pct)}")
    # symmetric head search
    layers = sorted(int(l) for l in rows[0]["heads"]["clean"])
    best = []
    for l in layers:
        nh = len(rows[0]["heads"]["clean"][str(l)])
        for h in range(nh):
            rise = med([r["heads"]["orth@0.25"][str(l)][h] - r["heads"]["clean"][str(l)][h] for r in rows])
            clean = med([r["heads"]["clean"][str(l)][h] for r in rows])
            best.append((rise, l, h, clean))
    best.sort(reverse=True)
    print(f"  head search over layers {layers}: top-5 by median rise under orth@0.25: " + "; ".join(f"L{l} h{h} +{rise:.2f} (clean {c:.2f})" for rise, l, h, c in best[:5]))


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block73(root)
    block74(root, "")
    block74(root, "_4b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
