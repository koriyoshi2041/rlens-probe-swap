"""Fig. 19 - update growth across the band: measured vs the |1 − 2α|^(2(w−1)) prediction of the involution."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from data import RES
from style import BLUE, BLUE_LIGHT, INK, INK2, apply, save, ygrid


def make() -> None:
    apply()
    ws = json.load(open(RES / "block04" / "width_sweep.json"))
    widths = sorted({r["width"] for r in ws})
    alphas = sorted({r["alpha"] for r in ws})
    cols = [BLUE_LIGHT, BLUE, INK][: len(widths)]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for w, col in zip(widths, cols):
        ys = [np.median([r["growth_last_over_first"] for r in ws if r["width"] == w and r["alpha"] == a and r["lens"] == "J"]) for a in alphas]
        th = [(abs(1 - 2 * a) ** (2 * (w - 1)) if w > 1 else 1) if abs(1 - 2 * a) > 1e-9 or w == 1 else np.nan for a in alphas]
        ax.plot(alphas, ys, "-o", color=col, label=f"width {w}, measured")
        ax.plot(alphas, th, ls=(0, (2, 2)), color=col, lw=1.0)
    ax.plot([], [], ls=(0, (2, 2)), color=INK2, lw=1.0, label="prediction |1 − 2α|^(2(w−1)), dotted")
    ax.set_yscale("log")
    ax.set_xlabel("α")
    ax.set_ylabel("‖Δh‖² at the last band layer / at the first")
    ax.legend(loc="upper left", fontsize=6.5)
    ygrid(ax)
    save(fig, "Fig19_alpha_divergence")


if __name__ == "__main__":
    make()
