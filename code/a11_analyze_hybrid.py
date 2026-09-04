#!/usr/bin/env python3
"""Offline analysis of block43 (hybrid J/R bases). CPU only.  Usage: python3 a11_analyze_hybrid.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import per_item, sel  # noqa: E402
from rlens.analysis import cluster_map, fact_pairs, load_records, paired_difference  # noqa: E402


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load_records(root / "block43_hybrid_basis" / "hybrid_basis.jsonl")
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    idx = sorted({r["index"] for r in rows})
    cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
    for band in ("early_L3_8", "primary_L8_20"):
        print(f"===== {band} =====")
        ref = per_item(sel(rows, band=band, arm="J_s+J_t"))
        for arm in ("J_s+J_t", "R_s+R_t", "R_s+J_t", "J_s+R_t"):
            s = sel(rows, band=band, arm=arm)
            d = paired_difference(per_item(s), ref, cl)
            print(f"  {arm:8s} flip {np.mean([r['top1_is_swap'] for r in s]):.2f}  ΔM med {np.median([r['delta_margin'] for r in s]):+.2f}  vs J: {d['mean_diff']:+.2f} [{d['lo95']:+.2f},{d['hi95']:+.2f}]  Δlogp(ans) med {np.median([r['delta_logp_answer'] for r in s]):+.2f}  Δlogp(swap) med {np.median([r['delta_logp_swap_answer'] for r in s]):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
