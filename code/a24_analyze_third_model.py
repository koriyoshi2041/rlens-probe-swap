#!/usr/bin/env python3
"""Offline analysis of block 82 (Qwen3-4B mechanism checks); block 81 is analysed with a01_analyze_4b.py.
Usage: a24_analyze_third_model.py <results_root>"""
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


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    meta = json.load(open(root / "block81_qwen3_4b" / "meta.json"))
    rows = load_records(root / "block82_qwen3_4b_mechanism" / "mechanism.jsonl")
    rep = load_records(root / "block81_qwen3_4b" / "replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    print(f"===== block82: Qwen3-4B (dense) mechanism checks; band {meta['band'][0]}–{meta['band'][-1]}, n={len(rows)}, never-flipped by band clamp {len([r for r in rows if r['index'] in zero])} =====")
    print("  (1) single bridge position: flip | top1=answer | median ΔM | median KL   [all | never-flipped]")
    for arm in rows[0]["arms"]:
        a = [r["arms"][arm] for r in rows]
        z = [r["arms"][arm] for r in rows if r["index"] in zero]
        print(f"    {arm:12s} {np.mean([q['top1_is_swap'] for q in a]):.2f} {np.mean([q['top1_is_answer'] for q in a]):.2f} {med([q['delta_margin'] for q in a]):+6.2f} {med([q['kl'] for q in a]):5.2f}   | {np.mean([q['top1_is_swap'] for q in z]) if z else float('nan'):.2f}")
    # (2) head search
    layers = sorted(int(k) for k in rows[0]["clean_pattern_to_p"].keys())
    best = []
    for l in layers:
        nh = len(rows[0]["clean_pattern_to_p"][str(l)])
        for h in range(nh):
            rise = med([r["orth_pattern_to_p"][str(l)][h] - r["clean_pattern_to_p"][str(l)][h] for r in rows])
            cl = med([r["clean_pattern_to_p"][str(l)][h] for r in rows])
            best.append((rise, l, h, cl))
    best.sort(reverse=True)
    print("  (2) heads with the largest median rise of attention onto the bridge position under the complement paste:")
    for rise, l, h, cl in best[:8]:
        print(f"      L{l} h{h}: rise +{rise:.3f} (clean {cl:.2f})")
    L, H = best[0][1], best[0][2]
    clean_h = [r["clean_pattern_to_p"][str(L)][H] for r in rows]
    lab = [r["arms"]["clamp@4"]["top1_is_swap"] for r in rows]
    pos = [s for s, l in zip(clean_h, lab) if l]; neg = [s for s, l in zip(clean_h, lab) if not l]
    auc = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]) if pos and neg else float("nan")
    print(f"      top head L{L} h{H}: clean attention onto p predicts single-position clamp@4 flip with AUC {auc:.2f}; median clean attention flipped {med(pos):.2f} vs not {med(neg):.2f}")
    # (3) sliding-window stage necessity
    print("  (3) clean-run mean ablation of attention outputs at the final position, 8-layer windows: two-hop acc | single-hop acc (over clean-correct items)")
    starts = [w["start"] for w in rows[0]["stage"]["two_hop"]["windows"]]
    for i, s in enumerate(starts):
        line = f"    L{s:2d}–{s + 7:2d}:"
        for kind in ("two_hop", "single_hop"):
            rs = [r for r in rows if r["stage"][kind]["clean_correct"]]
            acc = np.mean([r["stage"][kind]["windows"][i]["still_correct"] for r in rs])
            dl = med([r["stage"][kind]["windows"][i]["delta_logp_answer"] for r in rs])
            line += f"  {kind} {acc:.2f} ({dl:+.2f})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
