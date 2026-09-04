"""Fig. 11 - necessity ladder: full donor paste (scale 0.25) with a k-dimensional subspace removed."""
from __future__ import annotations

import matplotlib.pyplot as plt

from data import flip_rate, load_jsonl, sel
from style import BLUE, GRAY, INK, INK2, VIOLET, apply, save, ygrid


def make() -> None:
    apply()
    r71 = load_jsonl("block71_necessity_ladder/necessity_ladder.jsonl")
    r77 = load_jsonl("block77_growth_pca/growth_pca.jsonl")
    ks = [2, 16, 64, 256, 1024]
    lens_v = [flip_rate(sel(r71, arm=f"full-Jtop{k}")) for k in ks]
    rand_v = [flip_rate(sel(r71, arm=f"full-rand{k}")) for k in ks if k > 2]
    pk = [64, 256, 1024]
    pca_v = [float(sum(r["necessity"][f"full-PCA{k}"]["top1_is_swap"] for r in r77) / len(r77)) for k in pk]
    both = float(sum(r["necessity"]["full-(R256+PCA256)"]["top1_is_swap"] for r in r77) / len(r77))
    full = flip_rate(sel(r71, arm="full"))
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.axhline(full, color=INK2, lw=0.8, ls=(0, (3, 2)), label="nothing removed", zorder=1)
    ax.plot([k for k in ks if k > 2], rand_v, "-o", color=GRAY, label="random k dims", zorder=2)
    ax.plot(ks, lens_v, "-o", color=BLUE, label="top-k lens (J) directions", zorder=3)
    ax.plot(pk, pca_v, "-o", color=VIOLET, label="top-k residual PCs", zorder=3)
    ax.plot([256], [both], marker="*", ms=9, color=INK, ls="none", label="lens-256 ∪ PCs-256", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k dimensions removed from the paste")
    ax.set_ylabel("flip rate (n = 59)")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="lower left")
    ygrid(ax)
    save(fig, "Fig11_subspace_necessity")


if __name__ == "__main__":
    make()
