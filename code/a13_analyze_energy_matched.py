#!/usr/bin/env python3
"""Offline analysis of block48 (energy-matched J / R / hybrid clamps). CPU only.  Usage: a13_analyze_energy_matched.py <results_root>"""
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
    rows = load_records(root / "block48_energy_matched_hybrid" / "energy_matched.jsonl")
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    idx = sorted({r["index"] for r in rows})
    cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
    for band in ("early_L3_8", "primary_L8_20"):
        print(f"===== {band} =====")
        ref = per_item(sel(rows, band=band, arm="J_s+J_t", version="as_is"))
        e_ref = np.median([r["energy_as_is"] for r in sel(rows, band=band, arm="J_s+J_t", version="as_is")])
        print(f"  {'arm':8s} {'version':18s} {'flip':>5s} {'ΔM med':>7s} {'KL med':>7s} {'energy/J':>9s}   ΔM − J(as-is) [95% CI]")
        for arm in ("J_s+J_t", "R_s+R_t", "R_s+J_t", "J_s+R_t"):
            for version in ("as_is", "match_positions", "match_total", "match_total_to_R"):
                s = sel(rows, band=band, arm=arm, version=version)
                if not s:
                    continue
                d = paired_difference(per_item(s), ref, cl)
                e = np.median([r["energy_as_is"] for r in s]) / e_ref if version == "as_is" else float("nan")
                print(f"  {arm:8s} {version:18s} {np.mean([r['top1_is_swap'] for r in s]):5.2f} {np.median([r['delta_margin'] for r in s]):7.2f} {np.median([r['kl'] for r in s]):7.3f} {e:9.2f}   {d['mean_diff']:+.2f} [{d['lo95']:+.2f},{d['hi95']:+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
