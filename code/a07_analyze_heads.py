#!/usr/bin/env python3
"""Offline analysis of block37 (per-head attribution at the full-attention layers). CPU only.

Usage: python3 a07_analyze_heads.py <results_dir>
For flipped vs unflipped items: per layer, the median per-head answer-direction
contribution (clamped - clean, final position), the top heads by median contribution and
by how many items they lead, the share of the layer's total carried by its top-3 heads,
and whether those heads' attention onto the best bridge position changes under the clamp.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rlens.analysis import load_records  # noqa: E402


def main() -> int:
    d = pathlib.Path(sys.argv[1])
    rows = load_records(d / "head_attribution.jsonl")
    layers = sorted(int(k) for k in rows[0]["layers"])
    for gname, g in (("flipped", [r for r in rows if r["top1_is_swap"]]), ("unflipped", [r for r in rows if not r["top1_is_swap"]])):
        print(f"\n===== {gname}: n={len(g)} =====")
        for layer in layers:
            key = str(layer)
            D = np.array([r["layers"][key]["head_delta_answer"] for r in g])  # [items, heads]
            med = np.median(D, axis=0)
            tot = D.sum(axis=1)
            order = np.argsort(-med)
            lead = np.bincount(np.argmax(D, axis=1), minlength=D.shape[1])
            top3_share = np.median(np.sort(D, axis=1)[:, -3:].sum(axis=1) / np.where(np.abs(tot) > 1e-6, tot, np.nan))
            ac = np.array([r["layers"][key]["attn_to_best_clean"] for r in g])
            ah = np.array([r["layers"][key]["attn_to_best_hooked"] for r in g])
            print(f"  L{layer}: layer total (median over items) {np.median(tot):+.2f}; top-3-head share of total (median) {top3_share:.2f}")
            print("    top heads by median Δ(answer dir): " + ", ".join(f"h{h}:{med[h]:+.2f} (leads {lead[h]} items; attn→bridge {np.median(ac[:, h]):.2f}→{np.median(ah[:, h]):.2f})" for h in order[:5]))
    # consistency: are the same heads on top for flipped items across the two groups' ranking?
    return 0


if __name__ == "__main__":
    sys.exit(main())
