"""Fig 1: overview — the setup (a two-hop prompt, a lens edit at the bridge position) and
the two conditions that decide whether the edited entity reaches the answer."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import style as S
from .data import forced_head_arm, load_jsonl, rate, three_groups_9b

BLUE_SOFT = "#dbe8f9"
VIOLET_SOFT = "#e2dff4"
YELLOW_SOFT = "#fbeec9"
AQUA_SOFT = "#d5f0e5"
GRAY_SOFT = "#f1f0ec"


def _numbers():
    rows = load_jsonl("block75_donor_free_recipe/donor_free.jsonl")
    flip, resc, fail = three_groups_9b()
    arm = forced_head_arm(rows[0]["arms"])
    by = {r["index"]: r for r in rows}
    both = rate([by[i]["arms"][arm] for i in flip])
    content_only = rate([by[i]["arms"]["clamp@4_best"] for i in flip])
    routing_only = rate([r["arms"]["F[h8@23,1]_best"] for r in rows])
    r66 = {r["index"]: r for r in load_jsonl("block66_key_value/key_value.jsonl")}
    neither = rate([r66[i]["arms"]["K@19,23+clamp2d@4"] for i in fail])
    return both, content_only, routing_only, neither, len(flip), len(fail), len(rows)


def _arrow(ax, p0, p1, color=S.INK2, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8, color=color, lw=lw))


def _setup(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    S.note(ax, "a two-hop prompt: the bridge entity is never in the input or the output", x=0, y=1.0, size=7.4, color=S.INK)
    ax.text(0, 0.90, "read:  the lens reads “Brazil” at p, layers 8–20\n"
                     "rewrite:  swap the two coordinates; the other lens\n              now reads “Mexico”\n"
                     "use:  the answer follows only with content at p\n           and a transport head that reads p",
            fontsize=5.8, color=S.INK2, va="top", linespacing=1.35)
    xb, xf = 0.60, 0.84
    y0, y1 = 0.20, 0.72

    def ly(l):
        return y0 + (y1 - y0) * l / 32
    # prompt row
    yt = 0.055
    ax.text(xb - 0.05, yt + 0.03, "The language spoken in the country where the Amazon River", ha="right", va="center",
            fontsize=6.0, color=S.INK)
    for x, tok, fc in ((xb, "ends", BLUE_SOFT), (xf, "is", VIOLET_SOFT)):
        ax.add_patch(Rectangle((x - 0.04, yt), 0.08, 0.06, fc=fc, ec=S.LIGHTGRAY, lw=0.6))
        ax.text(x, yt + 0.03, tok, ha="center", va="center", fontsize=6.4, color=S.INK)
    ax.text(xb, yt - 0.02, "bridge position p\n(“Brazil” reads best here)", ha="center", va="top", fontsize=5.6, color=S.BLUE, linespacing=1.2)
    ax.text(xf, yt - 0.02, "final position\n(answer produced)", ha="center", va="top", fontsize=5.6, color=S.VIOLET, linespacing=1.2)
    # residual-stream columns
    for x in (xb, xf):
        ax.add_patch(Rectangle((x - 0.022, y0), 0.044, y1 - y0, fc=S.SURFACE, ec=S.LIGHTGRAY, lw=0.6))
    for l in (0, 8, 16, 24, 31):
        ax.text(xb - 0.03, ly(l), f"L{l}", ha="right", va="center", fontsize=5.3, color=S.MUTED)
    ax.add_patch(Rectangle((xb - 0.022, ly(8)), 0.044, ly(21) - ly(8), fc=BLUE_SOFT, ec="none"))
    ax.add_patch(Rectangle((xf - 0.022, ly(17)), 0.044, ly(25) - ly(17), fc=VIOLET_SOFT, ec="none"))
    ax.add_patch(Rectangle((xf - 0.022, ly(26)), 0.044, ly(30) - ly(26), fc=YELLOW_SOFT, ec="none"))
    # lens plane inset
    px, py, pw, ph = 0.05, 0.30, 0.24, 0.30
    ax.add_patch(Rectangle((px, py), pw, ph, fc=S.SURFACE, ec=S.LIGHTGRAY, lw=0.6))
    ax.plot([px + 0.035, px + pw - 0.02], [py + 0.045, py + 0.045], color=S.INK2, lw=0.6)
    ax.plot([px + 0.035, px + 0.035], [py + 0.045, py + ph - 0.03], color=S.INK2, lw=0.6)
    ax.text(px + pw - 0.02, py + 0.02, "“Brazil” coordinate", ha="right", va="top", fontsize=5.3, color=S.INK2)
    ax.text(px + 0.008, py + ph - 0.03, "“Mexico” coordinate", ha="left", va="top", fontsize=5.3, color=S.INK2, rotation=90)
    cx0, cy0 = px + pw - 0.06, py + 0.09
    cx1, cy1 = px + 0.085, py + ph - 0.07
    ax.plot(cx0, cy0, "o", color=S.INK2, ms=4)
    ax.plot(cx1, cy1, "o", color=S.BLUE, ms=4)
    _arrow(ax, (cx0, cy0), (cx1, cy1), color=S.BLUE, lw=1.0)
    ax.text(px + pw / 2, py + ph + 0.02, "2-D lens plane at p, layers 8–20:\nswap or clamp the two coordinates", ha="center",
            va="bottom", fontsize=5.8, color=S.BLUE, linespacing=1.2)
    _arrow(ax, (px + pw + 0.005, py + ph / 2), (xb - 0.026, ly(14)), color=S.BLUE, lw=0.8, style="-")
    # bridge-side annotation, transport arrow, final-side annotations
    ax.text(xb + 0.03, ly(11), "MLPs L18–20 complete\nthe entity state", ha="left", va="center", fontsize=5.6, color=S.INK2, linespacing=1.2)
    _arrow(ax, (xb + 0.024, ly(19.5)), (xf - 0.026, ly(21)), color=S.VIOLET, lw=1.1)
    ax.text((xb + xf) / 2, ly(24.5), "attention L17–24 (L23 h8)\ncarries the entity", ha="center", va="bottom", fontsize=5.6, color=S.VIOLET, linespacing=1.2)
    ax.text(xf + 0.03, ly(28), "MLPs L26–29\namplify", ha="left", va="center", fontsize=5.6, color=S.INK2, linespacing=1.2)
    _arrow(ax, (xf, y1 + 0.005), (xf, y1 + 0.05), color=S.INK2, lw=0.9)
    ax.text(xf, y1 + 0.06, "Portuguese → Spanish", ha="center", va="bottom", fontsize=6.6, color=S.INK, fontweight="semibold")


def _grid(ax, nums):
    both, content_only, routing_only, neither, nflip, nfail, nall = nums
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    S.note(ax, "a single-position edit at p is used only when both hold", x=0, y=1.0, size=7.4, color=S.INK)
    cells = [
        (0.33, 0.52, "content + routing\nclamp ×4 +\nhead forced onto p", both, AQUA_SOFT, S.AQUA),
        (0.33, 0.17, "content only\nclamp ×4", content_only, AQUA_SOFT, S.AQUA),
        (0.67, 0.52, "routing only\nhead forced,\nno clamp", routing_only, GRAY_SOFT, S.GRAY),
        (0.67, 0.17, "neither\nkeys + clamp,\nplane empty", neither, GRAY_SOFT, S.GRAY),
    ]
    for x, y, lab, v, fc, ec in cells:
        ax.add_patch(FancyBboxPatch((x, y), 0.30, 0.3, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 0.15, y + 0.205, f"{v:.2f}", ha="center", va="center", fontsize=12.5, fontweight="bold", color=S.INK)
        ax.text(x + 0.15, y + 0.07, lab, ha="center", va="center", fontsize=5.2, color=S.INK2, linespacing=1.2)
    ax.text(0.48, 0.88, "head reads p", ha="center", va="center", fontsize=6.0, color=S.INK2)
    ax.text(0.82, 0.88, "head looks away", ha="center", va="center", fontsize=6.0, color=S.INK2)
    ax.text(0.30, 0.67, "plane holds\nthe entities", ha="right", va="center", fontsize=6.0, color=S.INK2, linespacing=1.2)
    ax.text(0.30, 0.32, "plane empty\nat p", ha="right", va="center", fontsize=6.0, color=S.INK2, linespacing=1.2)
    ax.text(0.0, 0.02, f"flip rate, Qwen3.5-9B, band L8–20. left column: the {nflip} items the band clamp\n"
                       f"flips; top right: all {nall}; bottom right: the {nfail} items no plane edit rescues.",
            fontsize=5.3, color=S.MUTED, va="bottom", linespacing=1.25)


def main() -> None:
    S.apply()
    nums = _numbers()
    fig = plt.figure(figsize=(S.FULL, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    _setup(ax_a)
    _grid(ax_b, nums)
    S.panel_label(ax_a, "a", dx=-0.005, dy=1.02)
    S.panel_label(ax_b, "b", dx=-0.02, dy=1.02)
    S.headline(fig, "Reading a concept, rewriting it, and the model using the rewrite are three different things", y=1.03)
    S.save(fig, "Fig1_overview")
    print("  grid:", [round(v, 2) if isinstance(v, float) else v for v in nums])


if __name__ == "__main__":
    main()
