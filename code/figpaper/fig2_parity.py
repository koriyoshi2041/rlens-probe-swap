"""Fig 2: the published coordinate swap cancels itself on even band widths; the clamp does not."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from . import style as S
from .data import RES, load_jsonl, mean_ci, rate


def _ladder_fn(rel: str, stage_key: bool):
    rows = load_jsonl(rel)
    rows = [r for r in rows if r["lens"] == "J" and (not stage_key or r["stage"] == "ladder")]
    widths = sorted({int(r["width"]) for r in rows})

    def get(arm, w):
        return [r for r in rows if r["arm"] == arm and int(r["width"]) == w]
    return get, widths


def _odd_even(get, widths, arm):
    acc = {}
    for w in widths:
        for r in get(arm, w):
            acc.setdefault(int(r["index"]), {}).setdefault("odd" if w % 2 else "even", []).append(float(r["delta_margin"]))
    diff = {i: float(np.mean(v["odd"]) - np.mean(v["even"])) for i, v in acc.items() if "odd" in v and "even" in v}
    return mean_ci(diff)


def _ladder(ax, get, widths, title):
    for w in widths:
        if w % 2 == 0:
            ax.axvspan(w - 0.5, w + 0.5, color=S.GRID, alpha=0.55, lw=0, zorder=0)
    inv = [rate(get("involution", w)) for w in widths]
    cl = [rate(get("clamp", w)) for w in widths]
    ax.plot(widths, cl, "-o", color=S.BLUE, ms=3.4, lw=1.4, label="clamp (this work)", zorder=3)
    ax.plot(widths, inv, "-o", color=S.RED, ms=3.4, lw=1.4, label="published swap", zorder=3)
    ax.set_xticks(widths)
    ax.set_xlim(widths[0] - 0.6, widths[-1] + 0.6)
    ax.set_ylim(0, 0.46)
    S.title(ax, title)
    return inv, cl


def _schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0, 6.1, "why: coordinates re-read from the edited activation make the swap an involution",
            fontsize=7.0, fontweight="semibold", va="top", color=S.INK)
    layers = ["layer l", "l + 1", "l + 2", "l + 3"]
    xs = [0.3, 2.7, 5.1, 7.5]
    rows = (
        ("published tool: coordinates re-read from the live activation at every layer", 4.35, S.RED,
         ["(a, b) → (b, a)", "(b, a) → (a, b)", "(a, b) → (b, a)", "(b, a) → (a, b)"], "after an even number of layers the state is back where it started"),
        ("clamp: coordinates set to the clean run's swapped values at every layer", 1.95, S.BLUE,
         ["→ (b, a)", "→ (b, a)", "→ (b, a)", "→ (b, a)"], "the swap holds at every depth, so more layers means more effect"),
    )
    for name, y, col, seq, tag in rows:
        ax.text(0, y + 0.85, name, fontsize=6.3, color=col, fontweight="semibold", va="bottom")
        for x, lab, s in zip(xs, layers, seq):
            ax.add_patch(plt.Rectangle((x, y - 0.36), 2.1, 0.72, fc=S.SURFACE, ec=col, lw=0.8))
            ax.text(x + 1.05, y + 0.08, s, ha="center", va="center", fontsize=6.4, color=S.INK)
            ax.text(x + 1.05, y - 0.24, lab, ha="center", va="center", fontsize=5.4, color=S.MUTED)
            if x != xs[-1]:
                ax.add_patch(FancyArrowPatch((x + 2.13, y), (x + 2.38, y), arrowstyle="-|>", mutation_scale=7, color=col, lw=0.8))
        ax.text(0, y - 0.62, tag, fontsize=6.0, color=S.INK2, va="top")
    ax.text(0, 0.05, "M(α) = I + α(S − I) has eigenvalue 1 − 2α on the antisymmetric mode: α = 1 gives −1\n"
                     "(involution); α = 2, the paper's “double strength”, gives −3 and grows ×3 per layer",
            fontsize=6.0, color=S.INK2, va="bottom", linespacing=1.3)


def main() -> None:
    S.apply()
    get9, w9 = _ladder_fn("block11_mechanism/m3_parity.jsonl", stage_key=False)
    get4, w4 = _ladder_fn("block27_4b_band8_20/replication.jsonl", stage_key=True)
    get3, w3 = _ladder_fn("block81_qwen3_4b/replication.jsonl", stage_key=True)
    meta3 = json.load(open(RES / "block81_qwen3_4b" / "meta.json"))

    fig = plt.figure(figsize=(S.FULL, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.5, wspace=0.24)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    titles = ["Qwen3.5-9B, band from L8 (n = 59)", "Qwen3.5-4B, band from L8 (n = 52)",
              f"Qwen3-4B, dense attention, own\nJ-lens, band from L{meta3['band'][0]} (n = {meta3['n_eligible']})"]
    stats = {}
    for ax, get, w, t, tag in zip(axes, (get9, get4, get3), (w9, w4, w3), titles, ("9B", "4B", "Q3-4B")):
        _ladder(ax, get, w, t)
        stats[tag] = {arm: _odd_even(get, w, arm) for arm in ("involution", "clamp")}
    axes[0].set_ylabel("flip rate (top-1 becomes the target answer)")
    axes[1].set_xlabel("band width in layers (shaded columns: even widths)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].legend(loc="upper left", fontsize=6.2, handlelength=1.3)
    for ax, letter in zip(axes, "abc"):
        S.panel_label(ax, letter, dx=-0.04 if letter != "a" else -0.3, dy=1.12 if letter == "c" else 1.02)

    axd = fig.add_subplot(gs[1, :2])
    _schematic(axd)
    S.panel_label(axd, "d", dx=-0.005, dy=0.98)

    axe = fig.add_subplot(gs[1, 2])
    ys, labels = [], []
    y = 0.0
    for tag, name in (("9B", "Qwen3.5-9B"), ("4B", "Qwen3.5-4B"), ("Q3-4B", "Qwen3-4B")):
        for arm, col, lab in (("involution", S.RED, "published swap"), ("clamp", S.BLUE, "clamp")):
            m, lo, hi, n = stats[tag][arm]
            axe.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=3.8, elinewidth=0.9, capsize=0,
                         label=lab if tag == "9B" else None)
            ys.append(y)
            y -= 1
        labels.append(name)
        y -= 0.7
    axe.axvline(0, color=S.LIGHTGRAY, lw=0.7)
    axe.set_yticks([ys[i] - 0.5 for i in range(0, len(ys), 2)])
    axe.set_yticklabels(labels, fontsize=6.4)
    axe.set_xlim(-1.3, 7.2)
    axe.set_xlabel("odd − even, paired Δmargin (nat)")
    S.title(axe, "the parity effect belongs\nto the involution only")
    axe.tick_params(axis="y", length=0)
    axe.legend(loc="upper right", fontsize=6.0)
    S.panel_label(axe, "e", dx=-0.3, dy=1.12)
    S.headline(fig, "Applied across a band of layers, the published swap undoes itself on even widths;\npinning the coordinates to the clean run's values fixes it in three models", y=1.0)
    S.save(fig, "Fig2_parity")
    for tag, d in stats.items():
        for arm, (m, lo, hi, n) in d.items():
            print(f"  odd-even {tag} {arm}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] n={n}")


if __name__ == "__main__":
    main()
