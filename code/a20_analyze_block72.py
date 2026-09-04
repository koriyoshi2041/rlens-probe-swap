#!/usr/bin/env python3
"""Offline analysis of block 72 (subspace controls: lens vs W_U vs PCA vs entity-difference span). CPU only.
Usage: a20_analyze_block72.py <results_root>"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402

BAND = list(range(8, 21))
SHOW = (8, 9, 10, 12, 15, 18, 20)
ARMS = ("J+R256", "J+J256", "J+WU256", "J+PCA256", "J+entdiff", "J+rand58", "J+rand256")


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    rows = load_records(root / "block72_subspace_controls" / "subspace_controls.jsonl")
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    print(f"===== block72: subspace controls at the bridge position (n={len(rows)}; entdiff dims {rows[0]['n_entdiff']}) =====")
    print("  multi-layer projector paste: flip | flipZ | injected total | static total | injected/static at " + ",".join(f"L{l}" for l in SHOW))
    for arm in ARMS:
        rs = [r["arms"][arm]["multi"] for r in rows]
        rz = [r["arms"][arm]["multi"] for r in rows if r["index"] in zero]
        ratios = [med([m["injected"][BAND.index(l)] / max(m["static"][BAND.index(l)], 1e-9) for m in rs]) for l in SHOW]
        print(f"  {arm:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f} {med([sum(m['injected']) for m in rs]):8.1f} {med([sum(m['static']) for m in rs]):8.1f}   " + " ".join(f"{x:4.2f}" for x in ratios))
    print("  single L8 paste: flip | flipZ | realised share of the donor difference at " + ",".join(f"L{l}" for l in SHOW))
    for arm in ARMS:
        rs = [r["arms"][arm]["single_L8"] for r in rows]
        rz = [r["arms"][arm]["single_L8"] for r in rows if r["index"] in zero]
        f = [med([m["realised"][BAND.index(l)] for m in rs]) for l in SHOW]
        print(f"  {arm:10s} {np.mean([m['top1_is_swap'] for m in rs]):5.2f} {np.mean([m['top1_is_swap'] for m in rz]):5.2f}   " + " ".join(f"{x:4.2f}" for x in f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
