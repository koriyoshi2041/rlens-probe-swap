"""Fig. 3 - where the lens edit sits between cheap baselines and a full activation patch."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, rate_ci, sel
from style import BLUE, BLUE_LIGHT, GRAY, INK2, RED, apply, ci_bars, save, xgrid

P = "primary_L8_20"


def make() -> None:
    apply()
    b2 = load_jsonl("block02/block02_records.jsonl")
    lad = load_jsonl("block10_ladder/ladder.jsonl")
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    sub = load_jsonl("block10_ladder/subspace_patch.jsonl")
    pat = load_jsonl("block07_causal/patch.jsonl")
    rows = [
        ("random plane, swap", sel(b2, condition="swap_gauss", lens="J", band=P, alpha=1.0, positions="all"), GRAY),
        ("mismatched entity pair, swap", sel(b2, condition="swap_shuffled", lens="J", band=P, alpha=1.0, positions="all"), GRAY),
        ("logit lens, clamp", sel(lad, lens="logit", arm="clamp", scale=1.0), GRAY),
        ("J-lens, published swap", sel(b2, condition="swap_raw", lens="J", band=P, alpha=1.0, positions="all"), RED),
        ("J-lens clamp, direct push removed", sel(c9, band=P, lens="J", pair="entity", arm="ortho_rescaled"), BLUE_LIGHT),
        ("J-lens clamp", sel(c9, band=P, lens="J", pair="entity", arm="full"), BLUE),
        ("J-lens clamp, scale 2", sel(lad, lens="J", arm="clamp", scale=2.0), BLUE),
        ("J-lens clamp, 14 minimal pairs", sel(sub, lens="J", arm="clamp_exchange"), BLUE),
        ("full activation patch, 14 minimal pairs", sel(pat, arm="patch_full", lens="J"), GRAY),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ys = np.arange(len(rows))[::-1]
    for y, (label, rs, col) in zip(ys, rows):
        m, lo, hi, n = rate_ci(rs)
        ax.barh(y, m, height=0.62, color=col, zorder=2)
        ci_bars(ax, y, m, lo, hi, color=INK2, horizontal=True)
        ax.text(max(hi, m) + 0.02, y, f"{m:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("flip rate, band L8–20, all prompt positions")
    xgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "Fig3_baseline_ladder")


if __name__ == "__main__":
    make()
