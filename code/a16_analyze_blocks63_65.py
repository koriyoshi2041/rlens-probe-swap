#!/usr/bin/env python3
"""Offline analysis of blocks 63 (gradient sensitivity split), 64 (propagation) and 65 (4B energy of arms).
CPU only.  Usage: a16_analyze_blocks63_65.py <results_root>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402

BAND = list(range(8, 21))
SHOW = (8, 9, 10, 12, 15, 18, 20)


def groups_9b(root):
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    resco = {r["index"] for r in e62 if r["arm"] == "orth@0.1" and r["top1_is_swap"]}
    real = {(r["index"], r["arm"]): r["delta_margin"] for r in e62}
    return zero, resc4, resco, real


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def block63(root):
    p = root / "block63_why_plane_ignored" / "why_plane_ignored.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, resc4, resco, real = groups_9b(root)
    groups = {
        "flipped by band clamp": [r for r in rows if r["index"] not in zero],
        "zero: clamp2d@4 rescues": [r for r in rows if r["index"] in zero and r["index"] in resc4],
        "zero: clamp2d@4 fails (27)": [r for r in rows if r["index"] in zero and r["index"] not in resc4],
        "zero: orth@0.1 rescues": [r for r in rows if r["index"] in zero and r["index"] in resco],
        "zero: neither": [r for r in rows if r["index"] in zero and r["index"] not in resc4 and r["index"] not in resco],
    }
    print("===== block63: final-margin gradient at the bridge position, band-mean per item, group medians =====")
    print("  share of ||g||^2 inside: J plane | R plane | random 2-D | R top-256 | random 256-D   (chance 0.0005 / 0.0625)")
    print(f"  {'group':28s} n  {'J':>7s} {'R':>7s} {'rand2':>7s} {'J/rand2':>8s} | {'R256':>7s} {'rand256':>8s} {'ratio':>6s} | {'|g|':>6s} {'gap':>5s}")
    for g, rs in groups.items():
        if not rs:
            continue
        def bm(key):
            return [float(np.mean([q[key] for q in r["layers"]])) for r in rs]
        sj, sr, s2, s256, sr256 = bm("share_J"), bm("share_R"), bm("share_rand2"), bm("share_R256"), bm("share_rand256")
        ratio2 = [a / max(b, 1e-9) for a, b in zip(sj, s2)]
        ratio256 = [a / max(b, 1e-9) for a, b in zip(s256, sr256)]
        print(f"  {g:28s} {len(rs):2d} {med(sj):7.4f} {med(sr):7.4f} {med(s2):7.4f} {med(ratio2):8.1f} | {med(s256):7.3f} {med(sr256):8.3f} {med(ratio256):6.2f} | {med(bm('grad_norm')):6.3f} {med(bm('gap')):5.2f}")
    print("  --- first-order prediction sum_l <g_l, delta_l> vs realised delta-margin (block62), group medians ---")
    print(f"  {'group':28s} {'clamp4 pred':>11s} {'real':>6s} | {'orth0.1 pred':>12s} {'real':>6s} | {'full0.25 pred':>13s} {'real':>6s}")
    for g, rs in groups.items():
        if not rs:
            continue
        line = f"  {g:28s}"
        for arm, w in (("clamp2d@4", 11), ("orth@0.1", 12), ("full@0.25", 13)):
            pred = [r["pred"][arm] for r in rs]
            rl = [real.get((r["index"], arm)) for r in rs]
            line += f" {med(pred):{w}.2f} {med(rl):6.2f} |"
        print(line)
    print("  --- cosine(g_l, direction), band-mean, group medians: donor diff d | its complement | clamp direction ---")
    for g, rs in groups.items():
        if not rs:
            continue
        def bm(key):
            return [float(np.mean([q[key] for q in r["layers"]])) for r in rs]
        print(f"  {g:28s} {med(bm('cos_g_d')):+.3f} {med(bm('cos_g_dorth')):+.3f} {med(bm('cos_g_clamp')):+.3f}")
    print("  --- attention from the final position onto the bridge position (L23 h8 / L23 all heads / L19 all heads), group medians ---")
    arms = ("clean", "clamp2d@4", "orth@0.1", "orth@0.25", "full@0.25")
    for key in ("L23_h8_to_bridge", "L23_sum_to_bridge", "L19_sum_to_bridge"):
        print(f"  [{key}]")
        for g, rs in groups.items():
            if not rs:
                continue
            print(f"    {g:28s} " + "  ".join(f"{arm} {med([r['attention'][arm][key] for r in rs]):.3f}" for arm in arms))
    # does the change in attention track rescue, within the zero-flip set, arm orth@0.1?
    zr = [r for r in rows if r["index"] in zero]
    for arm in ("clamp2d@4", "orth@0.1", "orth@0.25"):
        d_resc = [r["attention"][arm]["L23_sum_to_bridge"] - r["attention"]["clean"]["L23_sum_to_bridge"] for r in zr if r["attention"][arm]["top1_is_swap"]]
        d_not = [r["attention"][arm]["L23_sum_to_bridge"] - r["attention"]["clean"]["L23_sum_to_bridge"] for r in zr if not r["attention"][arm]["top1_is_swap"]]
        print(f"  Δ(L23 attention onto bridge) under {arm:10s}: rescued med {med(d_resc):+.3f} (n={len(d_resc)}) vs not {med(d_not):+.3f} (n={len(d_not)})")
    # consistency: block63 realised vs block62 realised for the same arm
    for arm in ("clamp2d@4", "orth@0.1"):
        diffs = [abs(r["attention"][arm]["delta_margin"] - real[(r["index"], arm)]) for r in rows if (r["index"], arm) in real]
        print(f"  consistency block63 vs block62 delta-margin {arm}: median |diff| {med(diffs):.3f}, max {max(diffs):.3f}")


def block64(root):
    p = root / "block64_propagation" / "propagation.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, _, _, _ = groups_9b(root)
    print("\n===== block64: propagation of pasted components at the bridge position =====")
    subs = ("J_plane", "J+R256", "J+rand256", "full")
    print("  (a) multi-layer projector paste: injected/static energy ratio per layer (median over items); 1 = nothing propagated")
    print(f"  {'subspace':10s} {'flip':>5s} {'flipZ':>5s} " + " ".join(f"L{l:>2d}" for l in SHOW) + "   total injected / total static")
    for sub in subs:
        rs = [r["arms"][sub]["multi"] for r in rows]
        rz = [r["arms"][sub]["multi"] for r in rows if r["index"] in zero]
        ratios = []
        for l in SHOW:
            i = BAND.index(l)
            ratios.append(med([m["injected"][i] / max(m["static"][i], 1e-9) for m in rs]))
        tot_i = med([sum(m["injected"]) for m in rs]); tot_s = med([sum(m["static"]) for m in rs])
        print(f"  {sub:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f} " + " ".join(f"{x:4.2f}" for x in ratios) + f"   {tot_i:8.1f} / {tot_s:8.1f}")
    print("  (b) paste at L8 only: realised fraction of the donor difference downstream, <h_int-h_clean, d_l>/||d_l||^2 (median over items)")
    print(f"  {'subspace':10s} {'flip':>5s} {'flipZ':>5s} " + " ".join(f"L{l:>2d}" for l in SHOW) + "   | in-subspace version " + " ".join(f"L{l:>2d}" for l in SHOW))
    for sub in subs:
        rs = [r["arms"][sub]["single_L8"] for r in rows]
        rz = [r["arms"][sub]["single_L8"] for r in rows if r["index"] in zero]
        f = [med([m["realised"][BAND.index(l)] for m in rs]) for l in SHOW]
        fs = [med([m["realised_sub"][BAND.index(l)] for m in rs]) for l in SHOW]
        print(f"  {sub:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f} " + " ".join(f"{x:4.2f}" for x in f) + "   | " + " ".join(f"{x:4.2f}" for x in fs))
    print("  (b') energy moved downstream after the L8-only paste, relative to the L8 injection (median): how much the model amplifies/keeps it")
    for sub in subs:
        rs = [r["arms"][sub]["single_L8"] for r in rows]
        line = [med([m["moved_energy"][BAND.index(l)] / max(m["moved_energy"][0], 1e-9) for m in rs]) for l in SHOW]
        print(f"  {sub:10s} " + " ".join(f"{x:5.2f}" for x in line))


def block65(root):
    p = root / "block65_4b_energy_of_arms" / "energy_of_arms.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    rep = load_records(root / "block27_4b_band8_20" / "replication.jsonl") if (root / "block27_4b_band8_20" / "replication.jsonl").exists() else load_records(root / "block27_4b" / "replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    idx = sorted({r["index"] for r in rows})
    print(f"\n===== block65: Qwen3.5-4B, same arms at the bridge position (n={len(idx)}, never-flipped by the 4B band clamp: {len([i for i in idx if i in zero])}) =====")
    share = med([r["donor_diff_inplane_energy"] / r["donor_diff_energy"] for r in rows if r["arm"] == "clamp2d@4"])
    print(f"  donor-difference in-plane share median {share:.4f} (chance 2/{2560} = {2/2560:.4f})")
    print(f"  {'arm':12s} {'flipAll':>7s} {'E_all':>7s} | {'flipZ':>6s} {'E_zero':>7s} {'nresc':>5s} {'nZ':>3s}")
    order = ["clamp2d@1", "clamp2d@1.5", "clamp2d@2", "clamp2d@3", "clamp2d@4", "orth@0.1", "orth@0.25", "orth@0.5", "orth@1", "full@0.25", "full@0.5", "full@1", "rand16", "rand64", "rand256", "Rtop256"]
    for arm in order:
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        z = [r for r in s if r["index"] in zero]
        print(f"  {arm:12s} {np.mean([r['top1_is_swap'] for r in s]):7.2f} {med([r['energy'] for r in s]):7.1f} | {np.mean([r['top1_is_swap'] for r in z]) if z else float('nan'):6.2f} {med([r['energy'] for r in z]):7.1f} {sum(r['top1_is_swap'] for r in z):5d} {len(z):3d}")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block63(root)
    block64(root)
    block65(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
