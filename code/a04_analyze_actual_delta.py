#!/usr/bin/env python3
"""Offline analysis of block28 (clamp first-order prediction with actual deltas). CPU only.

Usage: python3 a04_analyze_actual_delta.py <results_dir>
Reports, per lens: the old (clean-delta) and new (actual-delta) first-order sums vs the
measured clamp effect; the self-check that re-injecting the captured deltas at scale 1
reproduces the clamp; the dose-response of the same direction field at scales .25/.5/1;
and the per-layer share of the actual prediction (where does the first-order effect come from).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rlens.analysis import load_records, select  # noqa: E402


def q(x, p):
    return float(np.percentile(x, p))


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    rows = load_records(d / "actual_delta.jsonl")
    for lens in ("J", "R"):
        sub = select(rows, lens=lens)
        actual = np.array([r["actual_delta_margin"] for r in sub])
        pc = np.array([r["predicted_clean_sum"] for r in sub])
        pa = np.array([r["predicted_actual_sum"] for r in sub])
        re1 = np.array([r["additive_reinjection_delta_margin"]["1.0"] for r in sub])
        re05 = np.array([r["additive_reinjection_delta_margin"]["0.5"] for r in sub])
        re025 = np.array([r["additive_reinjection_delta_margin"]["0.25"] for r in sub])
        print(f"\n===== lens {lens}: n={len(sub)} =====")
        print(f"measured clamp ΔM          median {np.median(actual):+.2f}  mean {actual.mean():+.2f}")
        print(f"old prediction (clean Δ)   median {np.median(pc):+.2f}  mean {pc.mean():+.2f}   ratio-of-medians actual/pred = {np.median(actual)/np.median(pc):.2f}")
        print(f"new prediction (actual Δ)  median {np.median(pa):+.2f}  mean {pa.mean():+.2f}   ratio-of-medians actual/pred = {np.median(actual)/np.median(pa):.2f}")
        ratio = actual / np.where(np.abs(pa) > 1e-6, pa, np.nan)
        print(f"per-item actual/new-pred   median {np.nanmedian(ratio):.2f}  IQR [{q(ratio[~np.isnan(ratio)],25):.2f},{q(ratio[~np.isnan(ratio)],75):.2f}]  n(pred<=0)={int((pa<=0).sum())}")
        print(f"self-check reinjection@1.0 vs clamp: max|diff| {np.max(np.abs(re1-actual)):.3f}  median|diff| {np.median(np.abs(re1-actual)):.3f}")
        print(f"dose-response of actual Δ field: scale .25 median {np.median(re025):+.2f} | .5 {np.median(re05):+.2f} | 1.0 {np.median(re1):+.2f}   (linear would be {np.median(re1)*0.25:+.2f} / {np.median(re1)*0.5:+.2f})")
        print(f"  ratio measured/linear-from-pred at .25: {np.median(re025)/ (0.25*np.median(pa)):.2f}; at .5: {np.median(re05)/(0.5*np.median(pa)):.2f}; at 1: {np.median(re1)/np.median(pa):.2f}")
        rho = np.corrcoef(np.argsort(np.argsort(pa)), np.argsort(np.argsort(actual)))[0, 1]
        print(f"Spearman(new pred, measured) = {rho:+.2f}")
        layers = sorted(int(k) for k in sub[0]["predicted_actual_per_layer"])
        share_a = np.array([[r["predicted_actual_per_layer"][str(l)] for l in layers] for r in sub])
        share_c = np.array([[r["predicted_clean_per_layer"][str(l)] for l in layers] for r in sub])
        en_a = np.array([[r["energy_actual_per_layer"][str(l)] for l in layers] for r in sub])
        print("per-layer median: actual-Δ prediction | clean-Δ prediction | actual energy share")
        tot_e = en_a.sum(axis=1, keepdims=True)
        for i, l in enumerate(layers):
            print(f"  L{l:02d}  {np.median(share_a[:, i]):+6.2f} | {np.median(share_c[:, i]):+6.2f} | {np.median(en_a[:, i]/tot_e[:, 0]):.2f}")
        flipped = np.array([r["top1_is_swap"] for r in sub])
        if flipped.any() and (~flipped).any():
            print(f"flipped group: measured {np.median(actual[flipped]):+.2f} vs new pred {np.median(pa[flipped]):+.2f}; unflipped: {np.median(actual[~flipped]):+.2f} vs {np.median(pa[~flipped]):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
