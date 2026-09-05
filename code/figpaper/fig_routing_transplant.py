"""Fig. 13 - what the transport layers receive at the bridge position: keys, values, or the whole pasted state."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, rescued_by_clamp4, zero_set_9b
from style import BLUE, GRAY, GRAY_DARK, INK2, apply, save, ygrid

ARMS = [
    ("clamp2d@4", "2-D clamp\n×4", "b66"),
    ("K@19,23", "donor keys\nonly", "b66"),
    ("K@19,23+clamp2d@4", "donor keys\n+ clamp ×4", "b66"),
    ("KV@19,23", "donor keys\n+ values", "b66"),
    ("transport<-clamp", "L17–24 fed\nclamped state", "b73"),
    ("KV19,23+lin_in", "L17–24 fed\npasted state", "b73"),
    ("orth@0.25", "complement\npaste ×0.25", "b66"),
]


def make() -> None:
    apply()
    src = {
        "b66": {int(r["index"]): r for r in load_jsonl("block66_key_value/key_value.jsonl")},
        "b73": {int(r["index"]): r for r in load_jsonl("block73_routing_all_layers/routing_all_layers.jsonl")},
    }
    zero = zero_set_9b()
    resc = rescued_by_clamp4()
    ids = sorted(src["b66"])
    groups = [
        ("band clamp flips", [i for i in ids if i not in zero], BLUE),
        ("no band-clamp flip, clamp ×4 rescues", [i for i in ids if i in zero and i in resc], GRAY_DARK),
        ("no band-clamp flip, clamp ×4 fails", [i for i in ids if i in zero and i not in resc], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = 0.26
    x = np.arange(len(ARMS))
    for k, (label, gid, col) in enumerate(groups):
        vals = []
        for arm, _, s in ARMS:
            rows = src[s]
            vals.append(float(np.mean([rows[i]["arms"][arm]["top1_is_swap"] for i in gid if i in rows])))
        xs = x + (k - 1) * w
        ax.bar(xs, vals, width=w * 0.92, color=col, label=f"{label} (n = {len(gid)})", zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, "0" if v < 0.005 else f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in ARMS], fontsize=7)
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.2, handlelength=1.0)
    ygrid(ax)
    save(fig, "Fig13_routing_transplant")


if __name__ == "__main__":
    make()
