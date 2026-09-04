#!/usr/bin/env python3
"""Offline analysis of block34 (pile-10k / skip-4 refit). CPU only.

Usage: python3 a10_analyze_refit.py <results_root>
"""
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
    d = root / "block34_refit_pile"
    rows = load_records(d / "refit_eval.jsonl")
    geo = json.loads((d / "geometry.json").read_text())
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}
    idx = sorted({r["index"] for r in rows})
    cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
    print("geometry vs published J (mean over L8-20):")
    for lens in sorted({g["lens"] for g in geo}):
        gs = [g for g in geo if g["lens"] == lens]
        print(f"  {lens:16s} cos {np.mean([g['cos_to_published'] for g in gs]):.3f}  norm ratio {np.mean([g['norm_ratio_to_published'] for g in gs]):.2f}")
    print("behaviour (band L8-20):")
    for lens in sorted({r["lens"] for r in rows}):
        for arm in ("clamp", "involution"):
            s = sel(rows, lens=lens, arm=arm)
            print(f"  {lens:16s} {arm:10s} flip {np.mean([r['top1_is_swap'] for r in s]):.2f}  ΔM mean {np.mean([r['delta_margin'] for r in s]):+.2f}")
    pub = per_item(sel(rows, lens="published_J", arm="clamp"))
    for lens in sorted({r["lens"] for r in rows}):
        if lens == "published_J":
            continue
        d_ = paired_difference(per_item(sel(rows, lens=lens, arm="clamp")), pub, cl)
        print(f"  {lens} − published_J (clamp ΔM): {d_['mean_diff']:+.2f} [{d_['lo95']:+.2f},{d_['hi95']:+.2f}] win {d_['win_rate']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
