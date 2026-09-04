#!/usr/bin/env python3
"""Offline analysis of block30 (answer-signal emergence by layer and sublayer). CPU only.

Usage: python3 a05_analyze_emergence.py <results_dir>
Per lens and per group (flipped / not flipped under the band clamp): median per-layer
contribution to the answer-contrast direction from attention, MLP and the clamp's direct
delta, the cumulative curves, and the logit-lens margin (hooked - clean) by layer.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rlens.analysis import load_records, select  # noqa: E402


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    rows = load_records(d / "answer_emergence.jsonl")
    for lens in ("J", "R"):
        sub = select(rows, lens=lens)
        layers = [p["layer"] for p in sub[0]["layers"]]
        groups = {"flipped": [r for r in sub if r["top1_is_swap"]], "unflipped": [r for r in sub if not r["top1_is_swap"]]}
        print(f"\n===== {lens}: n={len(sub)} flipped={len(groups['flipped'])} =====")
        for gname, g in groups.items():
            A = np.array([[p["attn_answer"] for p in r["layers"]] for r in g])
            M = np.array([[p["mlp_answer"] for p in r["layers"]] for r in g])
            D = np.array([[p["direct_answer"] for p in r["layers"]] for r in g])
            R = np.array([[p["resid_answer_delta"] for p in r["layers"]] for r in g])
            LL = np.array([[p["logit_lens_margin_hooked"] - p["logit_lens_margin_clean"] for p in r["layers"]] for r in g])
            print(f"--- {gname} (n={len(g)}): per-layer median answer-direction contribution (units: nat-like projection) ---")
            print(f"  {'L':>3s} {'attn':>7s} {'mlp':>7s} {'direct':>7s} {'cumA':>7s} {'cumM':>7s} {'cumD':>7s} {'resid':>7s} {'LLmargin':>9s}")
            cumA = cumM = cumD = 0.0
            for i, l in enumerate(layers):
                a, m, dd = np.median(A[:, i]), np.median(M[:, i]), np.median(D[:, i])
                cumA += a; cumM += m; cumD += dd
                print(f"  {l:3d} {a:7.2f} {m:7.2f} {dd:7.2f} {cumA:7.2f} {cumM:7.2f} {cumD:7.2f} {np.median(R[:, i]):7.2f} {np.median(LL[:, i]):9.2f}")
            tA, tM, tD = A.sum(axis=1), M.sum(axis=1), D.sum(axis=1)
            print(f"  totals (median over items): attn {np.median(tA):+.2f}  mlp {np.median(tM):+.2f}  direct {np.median(tD):+.2f}  | ΔM measured {np.median([r['delta_margin'] for r in g]):+.2f}")
            print(f"  share of computed answer signal from MLP: {np.median(tM / np.where(np.abs(tA + tM) > 1e-6, tA + tM, np.nan)):.2f} (median of per-item ratio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
