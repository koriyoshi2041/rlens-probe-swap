#!/usr/bin/env python3
"""Fact-level cluster-bootstrap CIs for the new paired contrasts (C19-C21, block 66/69/73/75/76). CPU only.
Usage: a22_ci_new_claims.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import cluster_map, fact_pairs, load_records, paired_difference  # noqa: E402


def contrasts(root):
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    out = []
    def add(block, file, arm_a, arm_b, subset_name, keep):
        p = root / block / file
        if not p.exists():
            return
        rows = [r for r in load_records(p) if keep(r)]
        A = {r["index"]: r["arms"][arm_a]["delta_margin"] for r in rows if arm_a in r["arms"] and arm_b in r["arms"]}
        B = {r["index"]: r["arms"][arm_b]["delta_margin"] for r in rows if arm_a in r["arms"] and arm_b in r["arms"]}
        idx = sorted(A)
        cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
        d = paired_difference(A, B, cl)
        fa = np.mean([r["arms"][arm_a]["top1_is_swap"] for r in rows if r["index"] in A]); fb = np.mean([r["arms"][arm_b]["top1_is_swap"] for r in rows if r["index"] in A])
        out.append((block, f"{arm_a} − {arm_b}", subset_name, len(idx), d["mean_diff"], d["lo95"], d["hi95"], d["win_rate"], fa, fb))
    allk = lambda r: True
    flip = lambda r: r["index"] not in zero
    zk = lambda r: r["index"] in zero
    # C21 site choice (block 69)
    diffpos = lambda r: r["p_h8"] != r["p_best"]
    for a, b in (("orth@0.25_h8", "orth@0.25_best"), ("clamp@4_h8", "clamp@4_best"), ("clamp@1_h8", "clamp@1_best"), ("clamp@4_all", "clamp@4_best")):
        add("block69_position_choice", "position_choice.jsonl", a, b, "all 59", allk)
        add("block69_position_choice", "position_choice.jsonl", a, b, "zero-flip 35", zk)
        add("block69_position_choice", "position_choice.jsonl", a, b, "p_h8≠p_best", diffpos)
    # C20 routing-assisted clamp (block 66 / 69 / 73)
    add("block66_key_value", "key_value.jsonl", "K@19,23+clamp2d@4", "clamp2d@4", "flippable 24", flip)
    add("block66_key_value", "key_value.jsonl", "K@19,23+clamp2d@4", "clamp2d@4", "all 59", allk)
    add("block69_position_choice", "position_choice.jsonl", "K<-foil+clamp@4_best", "clamp@4_best", "flippable 24", flip)
    add("block73_routing_all_layers", "routing_all_layers.jsonl", "PAT_h8@23+clamp", "clamp2d@4", "flippable 24", flip)
    # block 75 donor-free forcing
    for a in ("F[h8@23,1]+clamp@4_h8", "F[h8@23,0.8]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "K<-donor+clamp@4_h8"):
        add("block75_donor_free_recipe", "donor_free.jsonl", a, "clamp@4_h8", "flippable 24", flip)
        add("block75_donor_free_recipe", "donor_free.jsonl", a, "clamp@4_h8", "all 59", allk)
    add("block75_donor_free_recipe", "donor_free.jsonl", "F[h8@23,1]+clamp@4_best", "clamp@4_best", "flippable 24", flip)
    return out


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    print(f"{'block':28s} {'contrast':48s} {'subset':13s} {'n':>3s} {'mean Δ':>7s} {'95% CI':>18s} {'win':>5s} {'flip a':>7s} {'flip b':>7s}")
    for block, c, sub, n, m, lo, hi, w, fa, fb in contrasts(root):
        print(f"{block:28s} {c:48s} {sub:13s} {n:3d} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}] {w:5.2f} {fa:7.2f} {fb:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
