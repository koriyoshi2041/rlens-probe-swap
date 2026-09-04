#!/usr/bin/env python3
"""Offline analysis of block62 (block58 rerun with injected INPUT energy). CPU only.
Usage: a15_analyze_energy_arms.py <results_root>
Per arm, on all 59 items and on the 35 zero-flip items: flip rate, median injected energy,
median energy among rescued / not rescued, median KL; plus the donor-difference energy and its in-plane share."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ORDER = ["clamp2d@1", "clamp2d@1.5", "clamp2d@2", "clamp2d@3", "clamp2d@4", "orth@0.1", "orth@0.25", "orth@0.5", "orth@1",
         "full@0.25", "full@0.5", "full@1", "rand16", "rand64", "rand256", "Rtop256"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    sup = load(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    first = {r["index"]: r for r in rows}
    de = np.array([r["donor_diff_energy"] for r in first.values()]); di = np.array([r["donor_diff_inplane_energy"] for r in first.values()])
    print(f"donor-difference energy at the bridge position (sum over L8-20): median {np.median(de):.1f}; in-plane share median {np.median(di / de):.4f} (2 of 4096 dims = {2/4096:.4f})")
    print(f"{'arm':12s} | {'flipAll':>7s} {'E_all':>7s} {'KL_all':>6s} | {'flipZ':>5s} {'E_zero':>7s} {'E_resc':>7s} {'E_not':>7s} {'nresc':>5s} {'KLz_mean':>8s}")
    for arm in ORDER:
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        z = [r for r in s if r["index"] in zero]
        resc = [r for r in z if r["top1_is_swap"]]; notr = [r for r in z if not r["top1_is_swap"]]
        med = lambda xs, k: np.median([x[k] for x in xs]) if xs else float("nan")
        print(f"{arm:12s} | {np.mean([r['top1_is_swap'] for r in s]):7.2f} {med(s, 'energy'):7.1f} {med(s, 'kl'):6.2f} | {np.mean([r['top1_is_swap'] for r in z]):5.2f} {med(z, 'energy'):7.1f} {med(resc, 'energy'):7.1f} {med(notr, 'energy'):7.1f} {len(resc):5d} {np.mean([r['kl'] for r in z]):8.2f}")
    # matched-energy readout: for each clamp scale, find the complement scale with the closest median energy on the zero set
    print("\nzero-flip set, arms sorted by median injected energy:")
    stats = []
    for arm in ORDER:
        z = [r for r in rows if r["arm"] == arm and r["index"] in zero]
        if z:
            stats.append((np.median([r["energy"] for r in z]), arm, np.mean([r["top1_is_swap"] for r in z]), sum(r["top1_is_swap"] for r in z)))
    for e, arm, f, n in sorted(stats):
        print(f"  E={e:8.1f}  {arm:12s} flip {f:.2f} ({n}/{len(zero)})")
    # per-item: does the clamp at 4x inject more or less energy than orth@0.1 on the same item?
    c4 = {r["index"]: r for r in rows if r["arm"] == "clamp2d@4"}; o1 = {r["index"]: r for r in rows if r["arm"] == "orth@0.1"}
    common = [i for i in zero if i in c4 and i in o1]
    ratio = np.array([c4[i]["energy"] / max(o1[i]["energy"], 1e-9) for i in common])
    print(f"\nper-item energy ratio clamp2d@4 / orth@0.1 on zero-flip items: median {np.median(ratio):.2f}, IQR [{np.percentile(ratio, 25):.2f}, {np.percentile(ratio, 75):.2f}]; items where clamp@4 injects MORE than orth@0.1: {int((ratio > 1).sum())}/{len(common)}")
    both = [i for i in common if c4[i]["top1_is_swap"] and o1[i]["top1_is_swap"]]; only_c = [i for i in common if c4[i]["top1_is_swap"] and not o1[i]["top1_is_swap"]]; only_o = [i for i in common if o1[i]["top1_is_swap"] and not c4[i]["top1_is_swap"]]
    print(f"rescued by both: {len(both)}, clamp@4 only: {len(only_c)}, orth@0.1 only: {len(only_o)}, neither: {len(common) - len(both) - len(only_c) - len(only_o)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
