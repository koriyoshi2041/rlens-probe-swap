#!/usr/bin/env python3
"""Offline analysis of block 80 (4B donor-free recipe). CPU only. Usage: a23_analyze_block80.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import cluster_map, fact_pairs, load_records, paired_difference  # noqa: E402


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load_records(root / "block80_4b_donor_free_recipe" / "donor_free.jsonl")
    rep_dir = root / "block27_4b_band8_20" if (root / "block27_4b_band8_20" / "replication.jsonl").exists() else root / "block27_4b"
    rep = load_records(rep_dir / "replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    groups = {"all": rows, "flipped by 4B band clamp": [r for r in rows if r["index"] not in zero], "zero-flip": [r for r in rows if r["index"] in zero]}
    arms = ["clamp@4_best", "F[h8@23,1]+clamp@4_best", "F[h8h9h0@23,1]+clamp@4_best", "F[all@23,1]+clamp@4_best", "K<-donor+clamp@4_best", "F[h8@23,1]_best",
            "clamp@4_h8", "F[h8@23,1]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "K<-donor+clamp@4_h8", "clamp@4_all"]
    print(f"===== block80: Qwen3.5-4B donor-free recipe (n={len(rows)}; p_h8==p_best {np.mean([r['p_h8'] == r['p_best'] for r in rows]):.2f}) =====")
    for g, rs in groups.items():
        print(f"  --- [{g}] n={len(rs)}: flip | top1=answer | median ΔM | h8→p ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs if arm in r["arms"]]
            if a:
                print(f"    {arm:30s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ans {np.mean([q['top1_is_answer'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h8→p {med([q['h8_to_p'] for q in a]):.2f}")
    for g in ("flipped by 4B band clamp", "all"):
        rs = groups[g]
        idx = sorted(r["index"] for r in rs)
        cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
        for a, b in (("F[h8@23,1]+clamp@4_best", "clamp@4_best"), ("F[all@23,1]+clamp@4_best", "clamp@4_best"), ("K<-donor+clamp@4_best", "clamp@4_best"), ("clamp@4_all", "clamp@4_best")):
            A = {r["index"]: r["arms"][a]["delta_margin"] for r in rs if a in r["arms"] and b in r["arms"]}
            B = {r["index"]: r["arms"][b]["delta_margin"] for r in rs if a in r["arms"] and b in r["arms"]}
            d = paired_difference(A, B, cl)
            print(f"  [{g}] {a} − {b}: {d['mean_diff']:+.2f} [{d['lo95']:+.2f},{d['hi95']:+.2f}] win {d['win_rate']:.2f} (n={len(A)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
