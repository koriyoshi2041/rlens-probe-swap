#!/usr/bin/env python3
"""Offline analysis of block 66 (key/value/pattern patching at the transport layers). CPU only.
Usage: a17_analyze_block66.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402

ARMS = ["clean", "clamp2d@4", "orth@0.25", "K@19,23", "V@19,23", "KV@19,23", "Q@19,23", "KV@23", "KV@19", "PAT@19,23", "PAT_h8@23", "Z@19,23",
        "K@19,23+clamp2d@4", "PAT@19,23+clamp2d@4", "V@19,23+clamp2d@4", "KV@19,23+clamp2d@4", "KV@19,23<-clamp", "Z@19,23<-clamp"]


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load_records(root / "block66_key_value" / "key_value.jsonl")
    check = json.load(open(root / "block66_key_value" / "self_check.json"))
    print("self-check:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in check.items()})
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    groups = {"all": rows, "flipped by band clamp": [r for r in rows if r["index"] not in zero], "zero-flip (35)": [r for r in rows if r["index"] in zero],
              "zero: clamp2d@4 fails (27)": [r for r in rows if r["index"] in zero and r["index"] not in resc4]}
    for g, rs in groups.items():
        print(f"\n===== block66 [{g}] n={len(rs)}: flip rate | median Δmargin | median L23 h8 attention onto bridge | L23 all heads =====")
        for arm in ARMS:
            if arm not in rs[0]["arms"]:
                continue
            a = [r["arms"][arm] for r in rs]
            print(f"  {arm:22s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ΔM {np.median([q['delta_margin'] for q in a]):+6.2f}  h8 {np.median([q['L23_h8_to_bridge'] for q in a]):.3f}  L23 {np.median([q['L23_sum_to_bridge'] for q in a]):.2f}  L19 {np.median([q['L19_sum_to_bridge'] for q in a]):.2f}")
    # pairwise on the 27: how many does each combination rescue that clamp2d@4 alone does not
    rs = groups["zero: clamp2d@4 fails (27)"]
    print("\n  rescues among the 27 (clamp2d@4 alone rescues none by construction):")
    for arm in ("K@19,23", "PAT@19,23", "V@19,23", "KV@19,23", "K@19,23+clamp2d@4", "PAT@19,23+clamp2d@4", "V@19,23+clamp2d@4", "KV@19,23+clamp2d@4", "orth@0.25", "Z@19,23"):
        print(f"    {arm:22s} {sum(r['arms'][arm]['top1_is_swap'] for r in rs):2d}/27   toward original answer (top1 = answer & ΔM<0): {sum((not r['arms'][arm]['top1_is_swap']) and r['arms'][arm]['delta_margin'] < -0.5 for r in rs):2d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
