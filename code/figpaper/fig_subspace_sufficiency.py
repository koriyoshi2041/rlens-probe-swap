"""Fig. 10 - sufficiency ladder: donor paste restricted to a subspace, from the 2-D plane to the full vector."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import flip_rate, load_jsonl, sel, zero_set_9b
from style import BLUE, GRAY, INK2, apply, save, ygrid

ARMS = [("J_plane", "lens\nplane (2)"), ("J+R_4d", "J and R\nplanes (4)"), ("J_plane+top16", "+ top 16\nlens dirs"), ("J_plane+top64", "+ top 64"), ("J_plane+top256", "+ top 256"), ("J_plane+top1024", "+ top 1024"), ("outside_top256", "outside\ntop 256"), ("full", "full vector\n(4096)")]


def make() -> None:
    apply()
    rows = sel(load_jsonl("block46_subspace_ladder/subspace_ladder.jsonl"), donor="relation")
    zero = zero_set_9b()
    groups = [("all items", lambda i: True, BLUE), ("items the band clamp never flips", lambda i: i in zero, GRAY)]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    w = 0.38
    x = np.arange(len(ARMS))
    for k, (label, keep, col) in enumerate(groups):
        vals = [flip_rate([r for r in sel(rows, arm=a) if keep(int(r["index"]))]) for a, _ in ARMS]
        n = len({int(r["index"]) for r in rows if keep(int(r["index"]))})
        xs = x + (k - 0.5) * w
        ax.bar(xs, vals, width=w * 0.92, color=col, label=f"{label} (n = {n})", zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in ARMS], fontsize=7)
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left")
    ygrid(ax)
    save(fig, "Fig10_subspace_sufficiency")


if __name__ == "__main__":
    make()
