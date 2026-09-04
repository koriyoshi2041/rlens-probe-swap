#!/usr/bin/env python3
"""Offline analysis of block41 (head necessity under the clamp) and block42 (native role in clean runs). CPU only.

Usage: python3 a09_analyze_heads_causal.py <results_root>
"""
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
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    p = root / "block41_head_necessity" / "head_necessity.jsonl"
    if p.exists():
        rows = load_records(p)
        print("===== block41: clamp L8-20 + one head's z restored to clean at the final position =====")
        print(f"  {'arm':16s} {'flip':>6s} {'flip|band-flippable':>20s} {'ΔM med':>8s} {'KL med':>7s}")
        for arm in sorted({r["arm"] for r in rows}, key=lambda a: (a != "clamp_only", a)):
            s = sel(rows, arm=arm)
            f = [r for r in s if band_flip.get(r["index"], False)]
            print(f"  {arm:16s} {np.mean([r['top1_is_swap'] for r in s]):6.2f} {np.mean([r['top1_is_swap'] for r in f]):20.2f} {np.median([r['delta_margin'] for r in s]):8.2f} {np.median([r['kl'] for r in s]):7.2f}")
    p = root / "block42_head_native_role" / "head_native_role.jsonl"
    if p.exists():
        rows = load_records(p)
        print("\n===== block42: CLEAN run, head z zeroed at the final position: does the model's own answer survive? =====")
        for kind in ("two_hop", "single_hop"):
            print(f"  -- {kind} prompts --")
            print(f"  {'arm':12s} {'clean acc':>9s} {'acc after':>10s} {'Δlogp(ans) med':>15s} {'Δmargin med':>12s}")
            for arm in sorted({r["arm"] for r in rows}):
                s = sel(rows, prompt_kind=kind, arm=arm)
                cc = [r for r in s if r["clean_correct"]]
                print(f"  {arm:12s} {np.mean([r['clean_correct'] for r in s]):9.2f} {np.mean([r['still_correct'] for r in cc]) if cc else float('nan'):10.2f} {np.median([r['delta_logp_answer'] for r in cc]) if cc else float('nan'):15.2f} {np.median([r['delta_margin_swap_minus_answer'] for r in cc]) if cc else float('nan'):12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
