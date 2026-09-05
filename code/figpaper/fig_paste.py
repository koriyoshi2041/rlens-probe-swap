"""Fig. 9 - single bridge position: the 2-D lens plane vs the donor's complement vs the full donor vector."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import flip_rate, load_jsonl, sel, zero_set_9b
from style import BLUE, GRAY, INK2, apply, save, ygrid

ARMS = [("clamp2d", "2-D lens clamp"), ("paste_sub", "donor,\nlens plane only"), ("paste_orth", "donor,\ncomplement only"), ("paste_full", "donor,\nfull vector")]


def make() -> None:
    apply()
    rows = sel(load_jsonl("block31_donor_paste/donor_paste.jsonl"), position_set="best")
    foil = sel(load_jsonl("block31_donor_paste_foil/donor_paste.jsonl"), position_set="best", arm="paste_full")
    zero = zero_set_9b()
    groups = [("all items", lambda i: True, BLUE), ("items the band clamp does not flip", lambda i: i in zero, GRAY)]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    w = 0.36
    x = np.arange(len(ARMS))
    for k, (label, keep, col) in enumerate(groups):
        vals = [flip_rate([r for r in sel(rows, arm=a) if keep(int(r["index"]))]) for a, _ in ARMS]
        n = len({int(r["index"]) for r in rows if keep(int(r["index"]))})
        xs = x + (k - 0.5) * w
        ax.bar(xs, vals, width=w * 0.92, color=col, label=f"{label} (n = {n})", zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    fv = flip_rate(foil)
    ax.axhline(fv, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3, label=f"mismatched donor, full vector ({fv:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in ARMS])
    ax.set_ylabel("flip rate")
    ax.set_ylim(0, 0.98)
    ax.legend(loc="upper left")
    ygrid(ax)
    save(fig, "Fig9_donor_paste")


if __name__ == "__main__":
    make()
