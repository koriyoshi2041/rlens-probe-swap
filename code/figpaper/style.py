"""Shared style for the write-up figures: palette, rcParams, save helper.

Palette pairs were checked with a colour-vision-deficiency validator on a white surface.
Pairs that may share one figure: BLUE/RED, BLUE/ORANGE, BLUE/VIOLET, VIOLET/GRAY,
VIOLET/YELLOW (labelled). RED and ORANGE never share a figure. GRAY is de-emphasis only.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "figures_paper"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"  # J-lens, idempotent clamp, primary series
BLUE_LIGHT = "#9dc3ee"  # lighter step of BLUE (same-hue ramp)
RED = "#e34948"  # published swap (involution); damaged outcomes
ORANGE = "#eb6834"  # R-lens
VIOLET = "#4a3aa7"  # attention / transport / routing
YELLOW = "#eda100"  # MLP
GRAY = "#898781"  # controls, never-flipped items
GRAY_DARK = "#4f4e4a"
GRAY_LIGHT = "#c9c8c1"
INK = "#1a1a1a"
INK2 = "#555555"
RULE = "#e3e1da"
WHITE = "#ffffff"


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK2,
            "axes.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.labelcolor": INK,
            "text.color": INK,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.borderaxespad": 0.2,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, facecolor=WHITE)
    plt.close(fig)
    print("wrote", name)


def ygrid(ax) -> None:
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def xgrid(ax) -> None:
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)


def value_label(ax, x, y, text, dy=0.012, color=INK2, fontsize=7, ha="center", va="bottom") -> None:
    ax.text(x, y + dy, text, ha=ha, va=va, fontsize=fontsize, color=color)


def ci_bars(ax, x, m, lo, hi, color=INK2, horizontal=False, lw=0.9) -> None:
    """Thin 95% CI whisker without caps."""
    if horizontal:
        ax.plot([lo, hi], [x, x], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=3)
