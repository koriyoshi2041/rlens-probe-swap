#!/usr/bin/env python3
"""Offline analysis of block26 (entity-report probe). CPU only.

Usage: python3 a02_analyze_entity_probe.py <results_dir>
Questions answered:
  * how often does each probe phrasing yield a readable entity (top-1 is the
    intermediate token in the clean run)?
  * among items whose answer flips under the clamp, how often does the entity
    report also flip (top-1 becomes swap_to) / move (margin sign)?
  * across all items, correlation between the answer-margin change and the
    entity-margin change (is "rewriting the thought" the same variable as
    "changing the answer"?).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rlens.analysis import load_records, select  # noqa: E402


def spearman(x, y):
    from scipy.stats import spearmanr  # type: ignore

    return spearmanr(x, y).correlation


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    rows = load_records(d / "entity_probe.jsonl")
    idx = sorted({int(r["index"]) for r in rows})
    print(f"n items = {len(idx)}")
    for probe in ("q_noans", "q_cleanans", "q_swapans", "ref_noans"):
        clean = {r["index"]: r for r in select(rows, arm="clean", probe=probe)}
        readable = [i for i, r in clean.items() if r["entity_top1_is_intermediate"]]
        print(f"\n=== probe '{probe}': clean top-1 == intermediate for {len(readable)}/{len(clean)} items ===")
        for lens in ("J", "R"):
            arm = {r["index"]: r for r in select(rows, arm=lens, probe=probe)}
            flipped = [i for i in idx if arm[i]["top1_is_swap_answer"]]
            unflipped = [i for i in idx if not arm[i]["top1_is_swap_answer"]]
            def ent_stats(group):
                if not group:
                    return "n=0"
                to_swap = np.mean([arm[i]["entity_top1_is_swap_to"] for i in group])
                still_src = np.mean([arm[i]["entity_top1_is_intermediate"] for i in group])
                dmarg = [arm[i]["entity_margin"] - clean[i]["entity_margin"] for i in group]
                sign_flip = np.mean([arm[i]["entity_margin"] > 0 for i in group])
                return (f"n={len(group)} entity_top1=swap_to {to_swap:.2f} | still intermediate {still_src:.2f} | "
                        f"other {1 - to_swap - still_src:.2f} | median Δ(entity margin) {np.median(dmarg):+.2f} | entity margin>0 {sign_flip:.2f}")
            print(f"  {lens} answer-flipped   : {ent_stats(flipped)}")
            print(f"  {lens} answer-unflipped : {ent_stats(unflipped)}")
            # readable-only view (clean probe names the intermediate)
            fr = [i for i in flipped if i in readable]
            ur = [i for i in unflipped if i in readable]
            print(f"  {lens} [readable subset] flipped   : {ent_stats(fr)}")
            print(f"  {lens} [readable subset] unflipped : {ent_stats(ur)}")
            da = np.array([arm[i]["answer_margin"] - clean[i]["answer_margin"] for i in idx])
            de = np.array([arm[i]["entity_margin"] - clean[i]["entity_margin"] for i in idx])
            try:
                print(f"  {lens} Spearman(Δanswer margin, Δentity margin) over all items = {spearman(da, de):+.2f}; readable subset = {spearman(da[[idx.index(i) for i in readable]], de[[idx.index(i) for i in readable]]):+.2f}")
            except Exception as e:  # scipy missing
                print(f"  (spearman unavailable: {e}); Pearson = {np.corrcoef(da, de)[0, 1]:+.2f}")
            # the dissociation list, explicit
            both = [arm[i]["name"] for i in fr if arm[i]["entity_top1_is_swap_to"]]
            ans_only = [arm[i]["name"] for i in fr if arm[i]["entity_top1_is_intermediate"]]
            other = [arm[i]["name"] for i in fr if not arm[i]["entity_top1_is_swap_to"] and not arm[i]["entity_top1_is_intermediate"]]
            print(f"  {lens} readable+flipped: both change {len(both)} {both}")
            print(f"  {lens} readable+flipped: answer only {len(ans_only)} {ans_only}")
            print(f"  {lens} readable+flipped: entity other {len(other)} {other}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
