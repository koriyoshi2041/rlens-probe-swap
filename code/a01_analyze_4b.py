#!/usr/bin/env python3
"""Offline analysis of block27 (Qwen3.5-4B replication). CPU only; run locally on the synced results.

Usage: python3 a01_analyze_4b.py <results_dir> [<results_dir> ...]
Prints, per results dir: band flip rates (involution vs clamp, J/R/logit, full vs
orthogonalized), the width-parity table, the odd-even paired contrast, and the
orthogonalized retention fraction with cluster-bootstrap CIs.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rlens.analysis import bootstrap_mean, by_index, cluster_map, fact_pairs, load_records, paired_difference, select  # noqa: E402


def main() -> int:
    for d in sys.argv[1:]:
        d = pathlib.Path(d)
        rows = load_records(d / "replication.jsonl")
        meta = json.loads((d / "meta.json").read_text())
        idx = sorted({int(r["index"]) for r in rows})
        audit_rows = json.loads((d.parent / "block01" / "q01_data_audit_rows.json").read_text())
        items = {int(r["index"]): r for r in audit_rows}
        clusters = cluster_map(idx, fact_pairs({i: items[i] for i in idx if i in items}))  # fact-level clusters (audit 2 fix)
        print(f"\n===== {d.name}: band={meta['band']} n={len(idx)} =====")
        print("--- band arms (flip rate = top-1 becomes swap_answer; ΔM = mean Δmargin [95% CI]) ---")
        for lens in ("J", "R", "logit"):
            for arm in ("involution", "clamp"):
                for control in ("full", "ortho_rescaled"):
                    sub = select(rows, stage="band", lens=lens, arm=arm, control=control)
                    if not sub:
                        continue
                    m = by_index(sub, "delta_margin")
                    mean, lo, hi = bootstrap_mean(m, clusters)
                    flip = np.mean([r["top1_is_swap"] for r in sub])
                    neither = np.mean([not r["top1_is_swap"] and not r["top1_is_answer"] for r in sub])
                    energy = np.median([r["total_dh2"] for r in sub])
                    print(f"  {lens:5s} {arm:10s} {control:15s} flip={flip:.2f} neither={neither:.2f} ΔM={mean:+.2f} [{lo:+.2f},{hi:+.2f}] median_energy={energy:.3g}")
        for lens in ("J", "R"):
            full = by_index(select(rows, stage="band", lens=lens, arm="clamp", control="full"), "delta_margin")
            orth = by_index(select(rows, stage="band", lens=lens, arm="clamp", control="ortho_rescaled"), "delta_margin")
            if full and orth:
                ret = np.mean(list(orth.values())) / np.mean(list(full.values()))
                diff = paired_difference(orth, full, clusters)
                print(f"  {lens} clamp ortho retention = {ret:.2f}; ortho-full = {diff['mean_diff']:+.2f} [{diff['lo95']:+.2f},{diff['hi95']:+.2f}]")
            ci = by_index(select(rows, stage="band", lens=lens, arm="clamp", control="full"), "delta_margin")
            iv = by_index(select(rows, stage="band", lens=lens, arm="involution", control="full"), "delta_margin")
            if ci and iv:
                diff = paired_difference(ci, iv, clusters)
                print(f"  {lens} clamp - involution = {diff['mean_diff']:+.2f} [{diff['lo95']:+.2f},{diff['hi95']:+.2f}] win={diff['win_rate']:.2f}")
        print("--- width ladder: flip rate by width (involution / clamp) ---")
        widths = sorted({int(r["width"]) for r in rows if r["stage"] == "ladder"})
        for lens in ("J", "R"):
            line_i, line_c = [], []
            for w in widths:
                si = select(rows, stage="ladder", lens=lens, arm="involution", width=w)
                sc = select(rows, stage="ladder", lens=lens, arm="clamp", width=w)
                line_i.append(f"{np.mean([r['top1_is_swap'] for r in si]):.2f}")
                line_c.append(f"{np.mean([r['top1_is_swap'] for r in sc]):.2f}")
            print(f"  {lens} widths {widths}")
            print(f"    involution {' '.join(line_i)}")
            print(f"    clamp      {' '.join(line_c)}")
            # odd - even paired contrast on Δmargin, per item: mean over odd widths minus mean over even widths
            for arm in ("involution", "clamp"):
                per_item_odd, per_item_even = {}, {}
                for r in select(rows, stage="ladder", lens=lens, arm=arm):
                    tgt = per_item_odd if int(r["width"]) % 2 == 1 else per_item_even
                    tgt.setdefault(int(r["index"]), []).append(float(r["delta_margin"]))
                odd = {i: float(np.mean(v)) for i, v in per_item_odd.items()}
                even = {i: float(np.mean(v)) for i, v in per_item_even.items()}
                if odd and even:
                    diff = paired_difference(odd, even, clusters)
                    print(f"    {arm:10s} odd-even ΔM = {diff['mean_diff']:+.2f} [{diff['lo95']:+.2f},{diff['hi95']:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
