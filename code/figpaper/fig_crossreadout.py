"""Fig. 4 - the internal representation is swapped: best lens rank of each entity, clean vs after the swap."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, sel
from style import BLUE, GRAY, INK2, RULE, apply, save


def make() -> None:
    apply()
    cr = load_jsonl("block04/cross_readout.jsonl")
    clean = {r["index"]: r for r in sel(cr, arm="clean", reader="J")}
    swapped = {r["index"]: r for r in sel(cr, arm="R_swap", reader="J")}
    idx = sorted(set(clean) & set(swapped))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    lim = (0.7, 4000)
    ax.plot(lim, lim, color=RULE, lw=0.8, zorder=1)
    for key, col, label, mk in (("best_rank_intermediate", GRAY, "original entity", "o"), ("best_rank_swap_to", BLUE, "target entity", "s")):
        x = np.array([min(clean[i][key]) for i in idx], float)
        y = np.array([min(swapped[i][key]) for i in idx], float)
        jx = x * np.exp(rng.uniform(-0.06, 0.06, len(x)))
        jy = y * np.exp(rng.uniform(-0.06, 0.06, len(y)))
        ax.scatter(jx, jy, s=16, marker=mk, color=col, alpha=0.85, edgecolor="white", linewidth=0.5, label=label, zorder=3)
        ax.scatter([np.median(x)], [np.median(y)], s=70, marker=mk, color=col, edgecolor=INK2, linewidth=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("best rank in band L8–20, clean run")
    ax.set_ylabel("best rank after the swap\n(R-lens swap, read with the J-lens)")
    ax.legend(loc="upper left")
    save(fig, "Fig4_cross_readout")


if __name__ == "__main__":
    make()
