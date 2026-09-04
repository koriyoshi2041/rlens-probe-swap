"""Fig 6: what the 2-D plane misses at the bridge position, and where that content lives."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import style as S
from .data import band_flip_9b, load_jsonl, rate, sel


def _paste(ax):
    rows = load_jsonl("block31_donor_paste/donor_paste.jsonl")
    bf = band_flip_9b()
    arms = [("clamp2d", "2-D\nclamp"), ("paste_sub", "plane\ncoords"), ("paste_orth", "outside\nplane"), ("paste_full", "full\nvector")]
    x = np.arange(len(arms))
    out = {}
    handles = []
    for k, (gname, keep, col) in enumerate((("band clamp flips (n = 24)", lambda i: bf.get(i, False), S.AQUA),
                                            ("never flips (n = 35)", lambda i: not bf.get(i, False), S.GRAY))):
        vals = [rate([r for r in sel(rows, position_set="best", arm=a) if keep(r["index"])]) for a, _ in arms]
        out[gname] = vals
        xs = x + (k - 0.5) * 0.38
        handles.append(ax.bar(xs, vals, 0.36, color=col, label=gname, zorder=2))
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.0, color=S.INK)
    foil = rate(sel(load_jsonl("block31_donor_paste_foil/donor_paste.jsonl"), position_set="best", arm="paste_full"))
    ax.axhline(foil, color=S.RED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    handles.append(Line2D([0], [0], color=S.RED, lw=0.8, ls=(0, (3, 2)), label=f"mismatched donor, full vector ({foil:.2f})"))
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.2)
    ax.set_xlabel("pasted from the donor run at one bridge position, L8–20")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("flip rate")
    S.title(ax, "the content the second hop needs\nis outside the plane")
    ax.legend(handles=handles, loc="upper left", fontsize=5.6)
    S.ygrid(ax)
    return out, foil


def _sufficiency(ax):
    rows = sel(load_jsonl("block46_subspace_ladder/subspace_ladder.jsonl"), donor="relation")
    bf = band_flip_9b()
    arms = [("J_plane", "2"), ("J+R_4d", "4"), ("J_plane+top16", "+16"), ("J_plane+top64", "+64"), ("J_plane+top256", "+256"), ("J_plane+top1024", "+1024"), ("full", "full")]
    x = np.arange(len(arms))
    out = {}
    for gname, keep, col in (("all items (n = 59)", lambda i: True, S.BLUE), ("never flips (n = 35)", lambda i: not bf.get(i, False), S.GRAY)):
        vals = [rate([r for r in sel(rows, arm=a) if keep(r["index"])]) for a, _ in arms]
        out[gname] = vals
        ax.plot(x, vals, "-o", color=col, ms=4, lw=1.3, label=gname)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.4)
    ax.set_xlabel("dims pasted: plane + top-k of J")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("flip rate")
    S.title(ax, "sufficiency: about 256 lens directions\nmatch the full paste")
    ax.legend(loc="upper left", fontsize=6.0)
    S.ygrid(ax)
    return out


def _necessity(ax):
    r71 = load_jsonl("block71_necessity_ladder/necessity_ladder.jsonl")
    r77 = load_jsonl("block77_growth_pca/growth_pca.jsonl")
    ks = [2, 16, 64, 256, 1024]
    full = rate(sel(r71, arm="full"))
    lens = [rate(sel(r71, arm=f"full-Jtop{k}")) for k in ks]
    rand = [np.nan] + [rate(sel(r71, arm=f"full-rand{k}")) for k in ks[1:]]
    pk = [64, 256, 1024]
    pca = [np.mean([r["necessity"][f"full-PCA{k}"]["top1_is_swap"] for r in r77]) for k in pk]
    both = np.mean([r["necessity"]["full-(R256+PCA256)"]["top1_is_swap"] for r in r77])
    ax.axhline(full, color=S.LIGHTGRAY, lw=0.8)
    ax.text(2.1, full + 0.02, f"nothing removed: {full:.2f}", fontsize=6.0, color=S.INK2)
    ax.plot(ks, rand, "--s", color=S.GRAY, ms=4, lw=1.2, label="− k random dirs")
    ax.plot(ks, lens, "-o", color=S.BLUE, ms=4, lw=1.3, label="− top-k lens dirs")
    ax.plot(pk, pca, "-^", color=S.VIOLET, ms=4.5, lw=1.3, label="− top-k residual PCs")
    ax.plot([256], [both], "*", color=S.INK, ms=9, ls="none", label="− lens-256 ∪ PCA-256")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k removed from the full paste (×0.25)")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("flip rate, all items")
    S.title(ax, "necessity: removing those directions\nkills the paste; random ones do nothing")
    ax.legend(loc="lower left", fontsize=5.8)
    S.ygrid(ax)
    return full, lens, rand, pca, both


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    paste, foil = _paste(ax_a)
    suff = _sufficiency(ax_b)
    full, lens, rand, pca, both = _necessity(ax_c)
    for ax, letter, dx in ((ax_a, "a", -0.14), (ax_b, "b", -0.22), (ax_c, "c", -0.22)):
        S.panel_label(ax, letter, dx=dx, dy=1.12)
    S.headline(fig, "Items the clamp cannot flip lack low-rank content that lives in the residual principal components and the lens's high-gain directions, not in the plane", y=1.04)
    S.save(fig, "Fig6_subspace")
    for g, v in paste.items():
        print(f"  paste {g}: {np.round(v, 2).tolist()}  foil {foil:.2f}")
    for g, v in suff.items():
        print(f"  sufficiency {g}: {np.round(v, 2).tolist()}")
    print(f"  necessity full {full:.2f} lens {np.round(lens, 2).tolist()} rand {np.round(rand, 2).tolist()} pca {np.round(pca, 2).tolist()} both {both:.2f}")


if __name__ == "__main__":
    main()
