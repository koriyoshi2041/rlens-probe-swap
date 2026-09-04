#!/usr/bin/env python3
"""Offline analysis of block31 (donor paste vs 2-D clamp). CPU only.

Usage: python3 a06_analyze_donor.py <results_dir>
Flip rates and Δmargin per arm x position set, split by whether the item flips under the
full-band J clamp (block12), plus the item-level 2x2 (clamp2d fails / paste_full succeeds).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "audit_recompute"))
from audit_lib import load_jsonl, sel  # noqa: E402
from rlens.analysis import load_records, select  # noqa: E402


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    rows = load_records(d / "donor_paste.jsonl")
    skipped = json.loads((d / "skipped.json").read_text()) if (d / "skipped.json").exists() else []
    print(f"rows {len(rows)}; items {len({r['index'] for r in rows})}; skipped {len(skipped)}: {[s['name'] for s in skipped]}")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    arms = ("clamp2d", "paste_sub", "paste_orth", "paste_full")
    for pos in ("best", "last"):
        print(f"\n=== position set: {pos} ===")
        print(f"  {'group':12s} " + " ".join(f"{a:>22s}" for a in arms))
        for gname, cond in (("all", lambda i: True), ("band-flippable", lambda i: band_flip.get(i, False)), ("band-unflippable", lambda i: not band_flip.get(i, False))):
            cells = []
            for arm in arms:
                s = [r for r in select(rows, position_set=pos, arm=arm) if cond(r["index"])]
                if not s:
                    cells.append(f"{'n=0':>22s}")
                    continue
                cells.append(f"flip {np.mean([r['top1_is_swap'] for r in s]):.2f} ΔM {np.median([r['delta_margin'] for r in s]):+5.2f} n={len(s):2d}")
            print(f"  {gname:12s} " + " ".join(f"{c:>22s}" for c in cells))
        # item-level 2x2 between clamp2d and paste_full
        c2 = {r["index"]: r["top1_is_swap"] for r in select(rows, position_set=pos, arm="clamp2d")}
        pf = {r["index"]: r["top1_is_swap"] for r in select(rows, position_set=pos, arm="paste_full")}
        po = {r["index"]: r["top1_is_swap"] for r in select(rows, position_set=pos, arm="paste_orth")}
        both = sum(1 for i in c2 if c2[i] and pf.get(i)); only_full = sum(1 for i in c2 if not c2[i] and pf.get(i)); only_c2 = sum(1 for i in c2 if c2[i] and not pf.get(i)); neither = sum(1 for i in c2 if not c2[i] and not pf.get(i))
        print(f"  2x2 clamp2d vs paste_full: both {both}, paste_full only {only_full}, clamp2d only {only_c2}, neither {neither}")
        print(f"  among band-unflippable items: paste_full flips {sum(1 for i in pf if pf[i] and not band_flip.get(i, False))}, paste_orth flips {sum(1 for i in po if po[i] and not band_flip.get(i, False))}, clamp2d flips {sum(1 for i in c2 if c2[i] and not band_flip.get(i, False))}")
        kl = {arm: np.median([r["kl"] for r in select(rows, position_set=pos, arm=arm)]) for arm in arms}
        print(f"  median KL by arm: {kl}")
    hits = [r["best_pos_hits"] for r in select(rows, position_set="best", arm="clamp2d")]
    toks = {}
    for r in select(rows, position_set="best", arm="clamp2d"):
        toks[r["pos_token"]] = toks.get(r["pos_token"], 0) + 1
    print(f"\nbest position: hits median {np.median(hits):.0f}/13; most common tokens {sorted(toks.items(), key=lambda kv: -kv[1])[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
