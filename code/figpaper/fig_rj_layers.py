"""Fig. 7 - R-lens minus J-lens, single-layer swap, layer by layer (paired, 95% cluster-bootstrap CI)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, paired_ci, per_item, sel
from style import GRAY, ORANGE, RULE, apply, save, ygrid


def make() -> None:
    apply()
    sw = load_jsonl("block03_sweep/sweep.jsonl")
    layers = sorted({r["layer"] for r in sw})
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    for alpha, col, dx in ((1.0, ORANGE, 0.0), (0.5, GRAY, 0.18)):
        ms, los, his = [], [], []
        for l in layers:
            d = paired_ci(per_item(sel(sw, lens="R", layer=l, alpha=alpha)), per_item(sel(sw, lens="J", layer=l, alpha=alpha)))
            ms.append(d["mean_diff"])
            los.append(d["lo95"])
            his.append(d["hi95"])
        x = np.array(layers) + dx
        ax.errorbar(x, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="o", color=col, ms=3.5, elinewidth=0.9, capsize=0, label=f"single-layer swap, α = {alpha:g}", zorder=3)
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("R − J, Δmargin (nat)")
    ax.legend(loc="lower left")
    ygrid(ax)
    save(fig, "Fig7_R_minus_J_by_layer")


if __name__ == "__main__":
    make()
