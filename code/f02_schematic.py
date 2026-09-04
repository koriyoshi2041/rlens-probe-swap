#!/usr/bin/env python3
"""F26: mechanism schematic (drawn from the verified claims; no data plotted)."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures" / "F26_mechanism_schematic.png"
C = {"J": "#1f77b4", "R": "#e6801c", "grey": "#8c9ea3", "green": "#2ca25f", "mlp": "#b3c6d6", "attn": "#f0b27a"}


def box(ax, x, y, w, h, text, color, fs=8.5, alpha=0.25, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=color, ec="k", lw=0.8, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, x0, y0, x1, y1, text=None, color="k", ls="-", fs=7.5, dy=0.075):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2, ls=ls))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # columns: bridge position (left), final position (right)
    ax.text(0.22, 0.97, "bridge position  (where the intermediate entity is readable)", ha="center", fontsize=9.5, weight="bold")
    ax.text(0.78, 0.97, "final position  (where the answer is produced)", ha="center", fontsize=9.5, weight="bold")
    # residual stream at bridge, L8–20
    box(ax, 0.06, 0.62, 0.32, 0.26, "", C["grey"], alpha=0.10)
    ax.text(0.22, 0.86, "residual stream, L8–20", ha="center", fontsize=8.5, style="italic")
    box(ax, 0.08, 0.70, 0.13, 0.12, "2-D lens plane\n(J or R): entity\ncoordinates c", C["J"], fs=8, alpha=0.35)
    box(ax, 0.23, 0.70, 0.13, 0.12, "complement:\nresidual PCA ∪\nlens top-256", C["grey"], fs=8, alpha=0.35)
    ax.text(0.22, 0.645, "content: low-rank, mostly outside the 2-D plane (block 46/71/72/77)", ha="center", fontsize=7.5)
    # MLP generation L18–20
    box(ax, 0.06, 0.40, 0.26, 0.13, "MLPs L18–20 at the bridge position\nregenerate the entity representation\n(≈4.5× on the L20 yardstick; block 77)", C["mlp"], fs=7.8, alpha=0.6)
    arrow(ax, 0.22, 0.62, 0.22, 0.53)
    # transport
    box(ax, 0.42, 0.40, 0.20, 0.13, "transport stage: L17–24 attention\nL23 h8/h9/h0 respond to the edit;\nstage necessary, single heads redundant", C["attn"], fs=7.5, alpha=0.6)
    arrow(ax, 0.32, 0.465, 0.42, 0.465, "keys → 'attend here'\n(entity-graded)", fs=7.2)
    # final position
    box(ax, 0.72, 0.40, 0.24, 0.13, "final position residual\n→ MLPs L26–29 amplify\n→ unembed → answer", C["green"], fs=7.8, alpha=0.35)
    arrow(ax, 0.62, 0.465, 0.72, 0.465, "values → entity content", fs=7.2)
    # interventions row
    ax.text(0.5, 0.30, "what an edit needs at the bridge position", ha="center", fontsize=9, weight="bold")
    box(ax, 0.05, 0.10, 0.26, 0.15, "2-D lens clamp (published method fixed)\nworks only where (i) the plane separates the\ntwo entities and (ii) the transport head reads\nthe position; the involution form cancels on\neven band widths (block 1–4)", C["J"], fs=7.5, alpha=0.25)
    box(ax, 0.37, 0.10, 0.26, 0.15, "routing-assisted clamp (no donor)\nforce L23 h8's attention onto the edited\nposition + 2-D clamp: 0.79 → 0.96 on\nflippable items, 9B/4B/out-of-sample\n(block 75/76/80)", C["attn"], fs=7.5, alpha=0.5)
    box(ax, 0.69, 0.10, 0.26, 0.15, "donor complement paste\nsupplies keys and content; rescues items\nthe plane never can; 89% of its effect\npasses through the L17–24 inputs at the\nbridge position (block 62/73)", C["grey"], fs=7.5, alpha=0.3)
    ax.text(0.5, 0.03, "arrows: information flow in the clean model; boxes in the bottom row: interventions and what they can and cannot do", ha="center", fontsize=7.5, style="italic")
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
