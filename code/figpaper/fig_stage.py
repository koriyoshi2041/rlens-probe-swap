"""Fig. 15 - stage necessity for the edit: clamp on L8–20 with one sublayer group restored to its clean output."""
from __future__ import annotations

import matplotlib.pyplot as plt

from data import load_jsonl, rate_ci, sel
from style import BLUE, INK2, VIOLET, YELLOW, apply, ci_bars, save, ygrid

ARMS = [
    ("clamp_only", "none\n(clamp only)", BLUE),
    ("attn_9_16@final", "attention\nL9–16", VIOLET),
    ("attn_17_24@final", "attention\nL17–24", VIOLET),
    ("attn_25_31@final", "attention\nL25–31", VIOLET),
    ("mlp_17_24@final", "MLP\nL17–24", YELLOW),
    ("mlp_26_29@final", "MLP\nL26–29", YELLOW),
]


def make() -> None:
    apply()
    rows = load_jsonl("block39_stage_necessity/stage_necessity.jsonl")
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for k, (arm, label, col) in enumerate(ARMS):
        m, lo, hi, n = rate_ci(sel(rows, arm=arm))
        ax.bar(k, m, width=0.64, color=col, zorder=2)
        ci_bars(ax, k, m, lo, hi, color=INK2)
        ax.text(k, hi + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[1] for a in ARMS])
    ax.set_xlabel("sublayer group restored to clean at the final position")
    ax.set_ylabel("flip rate (n = 59)")
    ax.set_ylim(0, 0.62)
    ygrid(ax)
    save(fig, "Fig15_stage_necessity")


if __name__ == "__main__":
    make()
