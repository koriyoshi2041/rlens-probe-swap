#!/usr/bin/env python3
"""Offline analysis of block46 (subspace ladder for the donor paste). CPU only.  Usage: a12_analyze_subspace.py <results_root>"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import sel  # noqa: E402
from rlens.analysis import load_records  # noqa: E402


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load_records(root / "block46_subspace_ladder" / "subspace_ladder.jsonl")
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    bf = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    order = ["J_plane", "R_plane", "J_plane+R_s", "J_plane+R_t", "J+R_4d", "J_plane+top16", "J_plane+top64", "J_plane+top256", "J_plane+top1024", "outside_top256", "full"]
    for donor in ("relation", "N2"):
        print(f"===== donor {donor} (single best bridge position, L8-20) =====")
        print(f"  {'subspace':18s} {'flip all':>9s} {'flip unflippable':>17s} {'neither':>8s} {'ΔM med':>7s} {'KL med':>7s}")
        for arm in order:
            s = sel(rows, donor=donor, arm=arm)
            if not s:
                continue
            u = [r for r in s if not bf.get(r["index"], False)]
            neither = np.mean([not r["top1_is_swap"] and not r["top1_is_answer"] for r in s])
            print(f"  {arm:18s} {np.mean([r['top1_is_swap'] for r in s]):9.2f} {np.mean([r['top1_is_swap'] for r in u]):17.2f} {neither:8.2f} {np.median([r['delta_margin'] for r in s]):7.2f} {np.median([r['kl'] for r in s]):7.2f}")
    full = [r for r in sel(rows, donor="relation", arm="full") if "frac_orth_energy_top16" in r]
    if full:
        print("\n  energy of the plane-orthogonal donor difference inside the top-k right-singular directions of J_l (mean over band, median over items):")
        for k in ("top16", "top64", "top256", "top1024", "random256"):
            print(f"    {k:10s} {np.median([r[f'frac_orth_energy_{k}'] for r in full]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
