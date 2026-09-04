"""Fig 4: the internal representation is rewritten, the answer mostly is not; when it is,
the model also reports the new entity."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

from . import style as S
from .data import RES, band_flip_9b, load_jsonl, sel, spearman

GRAY_LIGHT = "#d3d2cc"


def _cross_readout(ax):
    cr = load_jsonl("block04/cross_readout.jsonl")
    cols = []
    for reader, arm in (("J", "R_swap"), ("R", "J_swap")):
        clean = {r["index"]: r for r in sel(cr, arm="clean", reader=reader)}
        swapped = {r["index"]: r for r in sel(cr, arm=arm, reader=reader)}
        idx = sorted(set(clean) & set(swapped))
        for key, concept in (("best_rank_intermediate", "original entity"), ("best_rank_swap_to", "target entity")):
            a = np.array([np.median(clean[i][key]) for i in idx])          # per item: median over band layers
            b = np.array([np.median(swapped[i][key]) for i in idx])
            pa = np.median([v for i in idx for v in clean[i][key]])        # pooled over items and layers
            pb = np.median([v for i in idx for v in swapped[i][key]])
            cols.append((reader, concept, a, b, pa, pb))
    x = np.arange(len(cols))
    rng = np.random.default_rng(0)
    for k, (reader, concept, a, b, pa, pb) in enumerate(cols):
        jit = rng.uniform(-0.1, 0.1, len(a))
        for j in range(len(a)):
            ax.plot([k - 0.2 + jit[j], k + 0.2 + jit[j]], [a[j], b[j]], color=S.LIGHTGRAY, lw=0.45, alpha=0.75, zorder=1)
        ax.plot([k - 0.2, k + 0.2], [pa, pb], color=S.INK2, lw=1.3, zorder=3)
        ax.plot(k - 0.2, pa, "o", mfc=S.SURFACE, mec=S.INK2, mew=1.0, ms=5.5, zorder=4)
        ax.plot(k + 0.2, pb, "o", color=S.BLUE, ms=5.5, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1.2, 4000)
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{'original' if 'original' in c[1] else 'target'}\nentity\n{c[4]:.0f} → {c[5]:.0f}" for c in cols],
                       fontsize=6.0, linespacing=1.2)
    ax.text(0.25, -0.36, "read by J-lens\n(R did the swap)", transform=ax.transAxes, ha="center", va="top", fontsize=5.9, color=S.INK2)
    ax.text(0.75, -0.36, "read by R-lens\n(J did the swap)", transform=ax.transAxes, ha="center", va="top", fontsize=5.9, color=S.INK2)
    ax.set_ylabel("rank of the concept in the band, L8–20\n(hollow: clean run; filled: after the swap)")
    S.title(ax, "the two concepts change places, as seen\nby the lens that did not do the edit")
    S.ygrid(ax)
    return cols


def _outcomes(ax):
    b2 = load_jsonl("block02/block02_records.jsonl")
    inv = sel(b2, condition="swap_raw", lens="J", band="primary_L8_20", alpha=1.0, positions="all")
    n_inv_flip = int(sum(r["top1_is_swap"] for r in inv))
    n = len(inv)
    mc = json.load(open(RES / "block23_semantic" / "manual_classification.json"))
    strict = int(sum(band_flip_9b().values()))
    sem = len(mc["rewrite_semantic_only"])
    dam = len(mc["damaged"])
    unchanged = n - strict - sem - dam
    rows = [
        (0, "idempotent\nclamp", [("strict flip", strict, S.AQUA), ("semantic rewrite, other token", sem, S.AQUA_LIGHT), ("damaged", dam, S.RED), ("unchanged", unchanged, GRAY_LIGHT)]),
        (1.3, "published\nswap", [("strict flip", n_inv_flip, S.AQUA), ("unchanged", n - n_inv_flip, GRAY_LIGHT)]),
    ]
    for y, name, segs in rows:
        left = 0
        small = 0
        for lab, v, col in segs:
            ax.barh(y, v, left=left, height=0.5, color=col, edgecolor=S.SURFACE, lw=1.0)
            if v >= 8:
                ax.text(left + v / 2, y, f"{lab}\n{v}", ha="center", va="center", fontsize=6.2,
                        color=S.SURFACE if col in (S.AQUA, S.RED) else S.INK, linespacing=1.2)
            else:
                ax.annotate(f"{lab}: {v}", (left + v / 2, y + 0.26), xytext=(0, 6 + 10 * small), textcoords="offset points",
                            ha="center", va="bottom", fontsize=5.9, color=S.INK2,
                            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": S.INK2, "shrinkA": 0, "shrinkB": 0})
                small += 1
            left += v
    ax.set_yticks([0, 1.3])
    ax.set_yticklabels(["idempotent\nclamp", "published\nswap"], fontsize=6.6)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.45, 2.1)
    ax.set_xlabel(f"items (n = {n}); clamp outcomes hand-classified\nfrom 20-token continuations")
    S.title(ax, "what the answer does")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return strict, sem, dam, unchanged, n_inv_flip


def _entity_probe(ax):
    rows = load_jsonl("block26_entity_probe/entity_probe.jsonl")
    clean = {r["index"]: r for r in sel(rows, arm="clean", probe="q_noans")}
    arm = {r["index"]: r for r in sel(rows, arm="J", probe="q_noans")}
    idx = sorted(arm)
    da = np.array([arm[i]["answer_margin"] - clean[i]["answer_margin"] for i in idx])
    de = np.array([arm[i]["entity_margin"] - clean[i]["entity_margin"] for i in idx])
    fl = np.array([bool(arm[i]["top1_is_swap_answer"]) for i in idx])
    ax.scatter(da[~fl], de[~fl], s=13, color=S.GRAY, alpha=0.85, label="answer does not flip", zorder=2, linewidths=0)
    ax.scatter(da[fl], de[fl], s=13, color=S.AQUA, alpha=0.95, label="answer flips", zorder=3, linewidths=0)
    rho = spearman(da, de)
    new = np.mean([arm[i]["entity_top1_is_swap_to"] for i in idx if arm[i]["top1_is_swap_answer"]])
    old = np.mean([arm[i]["entity_top1_is_intermediate"] for i in idx if arm[i]["top1_is_swap_answer"]])
    ax.set_xlabel("Δ answer margin at the prompt (nat)")
    ax.set_ylabel("Δ margin for naming the new entity\nwhen asked what the fact is about (nat)")
    S.title(ax, "when the answer flips, the model also\nsays the fact is about the new entity")
    ax.legend(loc="upper left", fontsize=6.0)
    S.note(ax, f"ρ = {rho:+.2f}\nflipped items: probe names\nthe new entity {new:.0%},\nthe old entity {old:.0%}",
           x=0.97, y=0.04, ha="right", va="bottom", size=6.0)
    S.ygrid(ax)
    return rho, new, old


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    cols = _cross_readout(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    strict, sem, dam, unch, ninv = _outcomes(ax_b)
    ax_c = fig.add_subplot(gs[0, 2])
    rho, new, old = _entity_probe(ax_c)
    for ax, letter, dx in ((ax_a, "a", -0.3), (ax_b, "b", -0.2), (ax_c, "c", -0.32)):
        S.panel_label(ax, letter, dx=dx, dy=1.12)
    S.headline(fig, "The representation is rewritten in almost every item; the answer follows in a minority, and when it does the whole state has moved", y=1.04)
    S.save(fig, "Fig4_rewrite_vs_use")
    for reader, concept, a, b, pa, pb in cols:
        print(f"  reader {reader} {concept}: pooled median rank {pa:.0f} -> {pb:.0f}; per-item {np.median(a):.0f} -> {np.median(b):.0f} (n={len(a)})")
    print(f"  clamp outcomes strict {strict}, semantic {sem}, damaged {dam}, unchanged {unch}; involution flips {ninv}")
    print(f"  entity probe rho {rho:+.2f}, new {new:.2f}, old {old:.2f}")


if __name__ == "__main__":
    main()
