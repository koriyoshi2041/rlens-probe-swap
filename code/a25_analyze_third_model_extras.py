#!/usr/bin/env python3
"""Offline analysis of blocks 83 (Qwen3-4B donor-free recipe), 84 (Qwen3-4B L23 head ablation) and 85 (fp32 lens refit
replication vs the bf16 one, block 81). CPU only. Usage: a25_analyze_third_model_extras.py <results_root>"""
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


def zero_set(root, block):
    rep = load_records(root / block / "replication.jsonl")
    return {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}


def block83(root):
    p = root / "block83_qwen3_4b_donor_free_recipe" / "donor_free.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero = zero_set(root, "block81_qwen3_4b")
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    groups = {"all": rows, "flipped by band clamp": [r for r in rows if r["index"] not in zero], "zero-flip": [r for r in rows if r["index"] in zero]}
    arms = ["clamp@4_best", "F[h8@23,1]+clamp@4_best", "F[all@23,1]+clamp@4_best", "K<-donor+clamp@4_best", "F[h8@23,1]_best", "clamp@4_h8", "F[h8@23,1]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "clamp@4_all"]
    print(f"===== block83: Qwen3-4B donor-free recipe (head L23 h10; n={len(rows)}; p_h8==p_best {np.mean([r['p_h8'] == r['p_best'] for r in rows]):.2f}) =====")
    for g, rs in groups.items():
        print(f"  --- [{g}] n={len(rs)}: flip | top1=answer | median ΔM | head→p ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs if arm in r["arms"]]
            if a:
                print(f"    {arm:28s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ans {np.mean([q['top1_is_answer'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h→p {med([q['h8_to_p'] for q in a]):.2f}")
    for g in ("flipped by band clamp", "all"):
        rs = groups[g]
        idx = sorted(r["index"] for r in rs)
        cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
        for a, b in (("F[h8@23,1]+clamp@4_best", "clamp@4_best"), ("F[all@23,1]+clamp@4_best", "clamp@4_best"), ("K<-donor+clamp@4_best", "clamp@4_best")):
            A = {r["index"]: r["arms"][a]["delta_margin"] for r in rs if a in r["arms"]}
            B = {r["index"]: r["arms"][b]["delta_margin"] for r in rs if a in r["arms"]}
            d = paired_difference(A, B, cl)
            print(f"  [{g}] {a} − {b}: {d['mean_diff']:+.2f} [{d['lo95']:+.2f},{d['hi95']:+.2f}] win {d['win_rate']:.2f} (n={len(A)})")


def block84(root):
    p = root / "block84_qwen3_4b_clean_head_ablation_zero" / "head_ablation.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    print(f"\n===== block84: Qwen3-4B clean-run ZERO ablation of L23 heads at the final position (n items={len({r['index'] for r in rows})}) =====")
    for arm in sorted({r["arm"] for r in rows}):
        line = f"  {arm:10s}"
        for kind in ("two_hop", "single_hop"):
            s = [r for r in rows if r["arm"] == arm and r["prompt_kind"] == kind and r["clean_correct"]]
            line += f" {kind} acc {np.mean([r['still_correct'] for r in s]):.2f} Δlogp {med([r['delta_logp_answer'] for r in s]):+.2f} |"
        print(line)


def block85(root):
    p = root / "block85_qwen3_4b_fp32" / "replication.jsonl"
    if not p.exists():
        return
    print("\n===== block85: Qwen3-4B lens refit in fp32 — width ladder vs the bf16 lens (block81) =====")
    for name, block in (("bf16 (block81)", "block81_qwen3_4b"), ("fp32 (block85)", "block85_qwen3_4b_fp32")):
        rep = load_records(root / block / "replication.jsonl")
        meta = json.load(open(root / block / "meta.json"))
        widths = sorted({int(r["width"]) for r in rep if r["stage"] == "ladder"})
        inv = [np.mean([r["top1_is_swap"] for r in rep if r["stage"] == "ladder" and r["lens"] == "J" and r["arm"] == "involution" and int(r["width"]) == w]) for w in widths]
        cl = [np.mean([r["top1_is_swap"] for r in rep if r["stage"] == "ladder" and r["lens"] == "J" and r["arm"] == "clamp" and int(r["width"]) == w]) for w in widths]
        band = [r for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["control"] == "full"]
        fi = np.mean([r["top1_is_swap"] for r in band if r["arm"] == "involution"]); fc = np.mean([r["top1_is_swap"] for r in band if r["arm"] == "clamp"])
        print(f"  {name}: band {meta['band'][0]}–{meta['band'][-1]} n={meta['n_eligible']}  band flips involution {fi:.2f} clamp {fc:.2f}")
        print("    involution " + " ".join(f"{x:.2f}" for x in inv))
        print("    clamp      " + " ".join(f"{x:.2f}" for x in cl))


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block83(root)
    block84(root)
    block85(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
