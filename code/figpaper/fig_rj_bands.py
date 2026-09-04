"""Fig. 8 - clamp flip rates, J-lens vs R-lens, early band vs workspace band, two models."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, rate_ci, sel
from style import BLUE, INK2, ORANGE, apply, ci_bars, save, ygrid


def make() -> None:
    apply()
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    r38 = load_jsonl("block27_4b_band3_8/replication.jsonl")
    r820 = load_jsonl("block27_4b_band8_20/replication.jsonl")
    groups = [
        ("Qwen3.5-9B\nL3–8", lambda lens: sel(c9, band="early_L3_8", lens=lens, pair="entity", arm="full")),
        ("Qwen3.5-9B\nL8–20", lambda lens: sel(c9, band="primary_L8_20", lens=lens, pair="entity", arm="full")),
        ("Qwen3.5-4B\nL3–8", lambda lens: sel(r38, stage="band", lens=lens, arm="clamp", control="full")),
        ("Qwen3.5-4B\nL8–20", lambda lens: sel(r820, stage="band", lens=lens, arm="clamp", control="full")),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    w = 0.36
    for g, (label, rows_of) in enumerate(groups):
        for k, (lens, col) in enumerate((("J", BLUE), ("R", ORANGE))):
            m, lo, hi, n = rate_ci(rows_of(lens))
            x = g + (k - 0.5) * w
            ax.bar(x, m, width=w * 0.92, color=col, label=f"{lens}-lens" if g == 0 else None, zorder=2)
            ci_bars(ax, x, m, lo, hi, color=INK2)
            ax.text(x, hi + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("clamp flip rate")
    ax.set_ylim(0, 0.64)
    ax.legend(loc="upper left")
    ygrid(ax)
    save(fig, "Fig8_R_vs_J_bands")


if __name__ == "__main__":
    make()
