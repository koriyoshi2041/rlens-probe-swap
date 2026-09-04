#!/usr/bin/env python3
"""Offline analysis of blocks 38-40 (neutral donor, stage necessity, early suppression). CPU only.

Usage: python3 a08_analyze_controls.py <results_root>
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import sel  # noqa: E402
from rlens.analysis import load_records  # noqa: E402


def flip(rows):
    return float(np.mean([r["top1_is_swap"] for r in rows])) if rows else float("nan")


def med(rows, k="delta_margin"):
    return float(np.median([r[k] for r in rows])) if rows else float("nan")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    unflip = lambda r: not band_flip.get(r["index"], False)

    p = root / "block38_neutral_donor" / "neutral_donor.jsonl"
    if p.exists():
        rows = load_records(p)
        print("===== block38 neutral donor (single best position, L8-20, J plane) =====")
        print(f"  {'donor':18s} {'mode':5s} {'all flip':>9s} {'unflippable flip':>17s} {'neither':>8s} {'ΔM med':>7s} {'KL med':>7s}")
        for donor in sorted({r["donor"] for r in rows}):
            for mode in ("full", "orth"):
                s = sel(rows, donor=donor, mode=mode)
                u = [r for r in s if unflip(r)]
                neither = np.mean([not r["top1_is_swap"] and not r["top1_is_answer"] for r in s])
                print(f"  {donor:18s} {mode:5s} {flip(s):9.2f} {flip(u):17.2f} {neither:8.2f} {med(s):7.2f} {med(s, 'kl'):7.2f}   (n={len(s)})")
        # item-level agreement between neutral and relation donors
        for mode in ("full", "orth"):
            rel = {r["index"]: r["top1_is_swap"] for r in sel(rows, donor="relation_target", mode=mode)}
            for key in ("N1_target", "N2_target"):
                neu = {r["index"]: r["top1_is_swap"] for r in sel(rows, donor=key, mode=mode)}
                both = sum(1 for i in rel if rel[i] and neu.get(i)); rel_only = sum(1 for i in rel if rel[i] and not neu.get(i)); neu_only = sum(1 for i in rel if not rel[i] and neu.get(i))
                print(f"  {mode}: relation vs {key}: both {both}, relation-only {rel_only}, neutral-only {neu_only}")

    p = root / "block39_stage_necessity" / "stage_necessity.jsonl"
    if p.exists():
        rows = load_records(p)
        print("\n===== block39 stage necessity (J clamp L8-20 all positions + restore a sublayer group to CLEAN) =====")
        base = sel(rows, arm="clamp_only")
        print(f"  {'arm':22s} {'flip':>6s} {'ΔM med':>8s} {'flip on band-flippable':>24s} {'KL med':>7s}")
        for arm in sorted({r["arm"] for r in rows}, key=lambda a: (a != "clamp_only", a)):
            s = sel(rows, arm=arm)
            f = [r for r in s if band_flip.get(r["index"], False)]
            print(f"  {arm:22s} {flip(s):6.2f} {med(s):8.2f} {flip(f):24.2f} {med(s, 'kl'):7.2f}")

    p = root / "block40_early_suppression" / "early_suppression.jsonl"
    if p.exists():
        rows = load_records(p)
        print("\n===== block40 early band L3-8: does extra suppression let J catch up with R? =====")
        print(f"  {'lens':4s} {'arm':18s} {'flip':>6s} {'ΔM med':>8s} {'Δlogp(ans) med':>15s} {'Δlogp(swap) med':>16s}")
        for lens in ("J", "R"):
            for arm in ("clamp", "clamp+ablate_src", "clamp+ablate_rand", "clamp+ablate_tgt", "ablate_src_only"):
                s = sel(rows, lens=lens, arm=arm)
                print(f"  {lens:4s} {arm:18s} {flip(s):6.2f} {med(s):8.2f} {med(s, 'delta_logp_answer'):15.2f} {med(s, 'delta_logp_swap_answer'):16.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
