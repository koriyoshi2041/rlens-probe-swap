"""Fig. 17 - a single paste at L8: how much of the donor difference the model regenerates by itself, by subspace."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl
from style import BLUE, GRAY, INK2, VIOLET, apply, save, ygrid

BAND = list(range(8, 21))


def _median_curve(rows, arm):
    return [float(np.median([r["arms"][arm]["single_L8"]["realised"][i] for r in rows])) for i in range(len(BAND))]


def make() -> None:
    apply()
    r72 = load_jsonl("block72_subspace_controls/subspace_controls.jsonl")
    r64 = load_jsonl("block64_propagation/propagation.jsonl")
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.plot(BAND, _median_curve(r64, "full"), ls=(0, (3, 2)), color=INK2, label="full donor vector")
    ax.plot(BAND, _median_curve(r72, "J+R256"), "-o", color=BLUE, label="lens plane + top-256 lens dirs")
    ax.plot(BAND, _median_curve(r72, "J+PCA256"), "-o", color=VIOLET, label="lens plane + top-256 residual PCs")
    ax.plot(BAND, _median_curve(r72, "J+rand256"), "-o", color=GRAY, label="lens plane + 256 random dirs")
    ax.set_xticks(BAND)
    ax.set_xlabel("layer read (paste applied at L8 only)")
    ax.set_ylabel("realised share of the donor difference\n(median over items)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.8, columnspacing=1.0, handlelength=1.4)
    ygrid(ax)
    save(fig, "Fig17_propagation")


if __name__ == "__main__":
    make()
