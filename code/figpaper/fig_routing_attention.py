"""Fig. 12 - before any edit: how much the transport head (L23 h8) attends to the bridge position, by item group."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, rescued_by_clamp4, zero_set_9b
from style import BLUE, GRAY, GRAY_DARK, INK, apply, save, ygrid


def make() -> None:
    apply()
    r63 = {int(r["index"]): r for r in load_jsonl("block63_why_plane_ignored/why_plane_ignored.jsonl")}
    zero = zero_set_9b()
    resc = rescued_by_clamp4()
    groups = [
        ("band clamp\nflips", [i for i in r63 if i not in zero], BLUE),
        ("no band-clamp flip,\nclamp ×4 rescues", [i for i in r63 if i in zero and i in resc], GRAY_DARK),
        ("no band-clamp flip,\nclamp ×4 fails", [i for i in r63 if i in zero and i not in resc], GRAY),
    ]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    labels = []
    for k, (label, ids, col) in enumerate(groups):
        ys = np.array([r63[i]["attention"]["clean"]["L23_h8_to_bridge"] for i in ids])
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(ys)), ys, s=14, color=col, alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
        ax.hlines(np.median(ys), k - 0.28, k + 0.28, color=INK, lw=1.4, zorder=4)
        labels.append(f"{label}\n(n = {len(ys)})")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("clean-run attention of L23 h8\nto the bridge position")
    ax.set_ylim(0, 1.0)
    ygrid(ax)
    save(fig, "Fig12_routing_attention")


if __name__ == "__main__":
    make()
