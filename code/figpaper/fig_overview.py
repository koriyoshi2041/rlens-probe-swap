"""Fig. 1 - overview: the two-hop prompt, the lens edit at the bridge position, and the content x routing mechanism.

Left: the setup on the running example. Right: the two conditions a single-position edit needs;
the three flip rates are computed from block 75 (flippable items, Qwen3.5-9B).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle

from data import load_jsonl, zero_set_9b
from style import BLUE, GRAY, GRAY_DARK, GRAY_LIGHT, INK, INK2, RED, VIOLET, WHITE, YELLOW, apply, save

TOKENS = ["Fact:", "The", "language", "spoken", "in", "the", "country", "where", "the", "Amazon", "River", "ends", "is"]
BRIDGE, FINAL = 9, 12
N_LAYERS = 32


def _recipe_rates():
    rows = {int(r["index"]): r for r in load_jsonl("block75_donor_free_recipe/donor_free.jsonl")}
    zero = zero_set_9b()
    ids = [i for i in rows if i not in zero]
    def rate(arm):
        return float(np.mean([rows[i]["arms"][arm]["top1_is_swap"] for i in ids]))
    return rate("clamp@4_best"), rate("F[h8@23,1]_best"), rate("F[h8@23,1]+clamp@4_best"), len(ids)


def _setup_panel(ax):
    for x in range(len(TOKENS)):
        ax.plot([x, x], [0, N_LAYERS - 1], color=GRAY_LIGHT, lw=0.9, zorder=1)
    ax.plot([BRIDGE, BRIDGE], [0, N_LAYERS - 1], color=BLUE, lw=1.6, alpha=0.55, zorder=2)
    ax.plot([FINAL, FINAL], [0, N_LAYERS - 1], color=GRAY_DARK, lw=1.6, alpha=0.55, zorder=2)
    ax.add_patch(Rectangle((-0.5, 8), len(TOKENS), 12, facecolor=BLUE, alpha=0.10, edgecolor="none", zorder=0))
    ax.text(-0.35, 8.7, "lens band L8–20 (coordinates edited)", fontsize=6.8, color=BLUE, va="bottom", ha="left")
    ax.plot([BRIDGE], [14], "o", color=BLUE, ms=5, zorder=6)
    for x, tok in enumerate(TOKENS):
        col, w = (BLUE, "bold") if x == BRIDGE else ((GRAY_DARK, "bold") if x == FINAL else (INK2, "normal"))
        ax.text(x, -1.6, tok, rotation=42, ha="right", va="top", fontsize=6.6, color=col, fontweight=w, rotation_mode="anchor")
    ax.text(BRIDGE, -7.6, "bridge position p\n(lens reads “Brazil”)", ha="center", va="top", fontsize=6.6, color=BLUE)
    ax.text(FINAL, -7.6, "final\nposition", ha="center", va="top", fontsize=6.6, color=GRAY_DARK)
    # transport arrows and MLP amplification
    for y in (19.5, 23.5):
        ax.add_patch(FancyArrowPatch((BRIDGE + 0.15, y), (FINAL - 0.15, y), arrowstyle="-|>", mutation_scale=7, color=VIOLET, lw=1.3, zorder=4, connectionstyle="arc3,rad=-0.18"))
    ax.text(10.1, 24.4, "transport\n(attention L17–24)", ha="center", va="bottom", fontsize=6.3, color=VIOLET)
    ax.add_patch(Rectangle((FINAL - 0.35, 25.7), 0.7, 3.6, facecolor=YELLOW, edgecolor="none", zorder=5))
    ax.text(FINAL + 0.5, 27.5, "MLP\nL26–29", ha="left", va="center", fontsize=6.6, color=INK2)
    ax.text(FINAL + 0.05, N_LAYERS + 0.6, "Spanish", ha="right", va="bottom", fontsize=7.5, color=BLUE, fontweight="bold")
    ax.text(FINAL - 1.75, N_LAYERS + 0.6, "Portuguese", ha="right", va="bottom", fontsize=7.5, color=GRAY)
    ax.plot([FINAL - 4.05, FINAL - 1.8], [N_LAYERS + 1.25, N_LAYERS + 1.25], color=GRAY, lw=0.8)
    ax.text(FINAL - 4.55, N_LAYERS + 0.6, "next token:", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xlim(-0.6, len(TOKENS) + 1.4)
    ax.set_ylim(-9, N_LAYERS + 4)
    ax.set_yticks([0, 8, 20, 31])
    ax.set_yticklabels(["L0", "L8", "L20", "L31"], fontsize=6.8)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_bounds(0, 31)
    ax.tick_params(axis="y", length=2)


def _plane_inset(fig, ax):
    ins = ax.inset_axes([0.05, 0.70, 0.25, 0.29])
    ins.set_xlim(-0.15, 1.2)
    ins.set_ylim(-0.15, 1.2)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(False)
    ins.set_facecolor(WHITE)
    ins.add_patch(FancyBboxPatch((-0.12, -0.12), 1.3, 1.3, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=WHITE, edgecolor=GRAY_LIGHT, lw=0.8))
    ins.add_patch(FancyArrowPatch((0, 0), (1.05, 0), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=0.9))
    ins.add_patch(FancyArrowPatch((0, 0), (0, 1.05), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=0.9))
    ins.text(1.02, -0.05, "Brazil", fontsize=6.2, ha="right", va="top", color=INK2)
    ins.text(0.05, 1.05, "Mexico", fontsize=6.2, ha="left", va="top", color=INK2)
    ins.plot([0.85], [0.18], "o", color=GRAY, ms=4.5, zorder=3)
    ins.plot([0.18], [0.85], "o", color=BLUE, ms=4.5, zorder=3)
    ins.add_patch(FancyArrowPatch((0.78, 0.25), (0.25, 0.78), arrowstyle="-|>", mutation_scale=6, color=BLUE, lw=1.0, connectionstyle="arc3,rad=0.35"))
    ins.text(0.62, 0.66, "swap", fontsize=6.2, color=BLUE, ha="left", va="bottom")
    ins.set_title("lens coordinates at p", fontsize=6.6, color=INK2, pad=2)
    con = ConnectionPatch(xyA=(1.18, -0.14), coordsA=ins.transData, xyB=(BRIDGE - 0.12, 14.2), coordsB=ax.transData, color=GRAY_LIGHT, lw=0.8, zorder=1)
    fig.add_artist(con)


def _mechanism_panel(ax, rates):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    def box(x, y, w, h, text, fc, ec, tc=INK, fs=7.0, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=fc, edgecolor=ec, lw=1.0, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight=weight, transform=ax.transAxes, linespacing=1.25)
    box(0.02, 0.66, 0.50, 0.24, "content\nthe plane at p separates\nthe two entities", "#e9f1fb", BLUE)
    box(0.02, 0.34, 0.50, 0.24, "routing\na transport head\n(L23 h8) reads p", "#ece9f6", VIOLET)
    box(0.70, 0.50, 0.28, 0.24, "answer\nfollows", WHITE, INK, weight="bold")
    for y0 in (0.78, 0.46):
        ax.add_patch(FancyArrowPatch((0.53, y0), (0.69, 0.62), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.9, transform=ax.transAxes, connectionstyle="arc3,rad=0.0"))
    ax.text(0.27, 0.62, "and", ha="center", va="center", fontsize=6.8, color=INK2, transform=ax.transAxes, fontstyle="italic")
    c, r, cr, n = rates
    rows = [("content only", c, BLUE), ("routing only", r, VIOLET), ("content + routing", cr, INK)]
    ax.text(0.02, 0.24, f"flip rate, flippable items (n = {n})", fontsize=6.6, color=INK2, transform=ax.transAxes, va="bottom")
    for k, (label, v, col) in enumerate(rows):
        y = 0.17 - k * 0.075
        ax.text(0.02, y, label, fontsize=6.8, color=INK, transform=ax.transAxes, va="center")
        ax.add_patch(Rectangle((0.50, y - 0.022), 0.40 * v, 0.044, facecolor=col, edgecolor="none", transform=ax.transAxes))
        ax.add_patch(Rectangle((0.50, y - 0.022), 0.40, 0.044, facecolor="none", edgecolor=GRAY_LIGHT, lw=0.6, transform=ax.transAxes))
        ax.text(0.915, y, f"{v:.2f}", fontsize=6.8, color=INK2, transform=ax.transAxes, va="center")


def make() -> None:
    apply()
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    _setup_panel(ax)
    _plane_inset(fig, ax)
    _mechanism_panel(ax2, _recipe_rates())
    ax.text(-0.09, 1.0, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax2.text(-0.02, 1.0, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top")
    save(fig, "Fig1_overview")


if __name__ == "__main__":
    make()
