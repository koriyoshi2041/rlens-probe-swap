"""Fig 5: R-lens vs J-lens — the original question. R helps only in early layers, only
after the protocol fix, and the advantage survives energy matching."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .data import item_values, load_jsonl, paired_ci, per_item, rate_ci, sel


def _sweep(ax):
    sw = load_jsonl("block03_sweep/sweep.jsonl")
    layers = sorted({r["layer"] for r in sw})
    ax.axvspan(layers[0] - 0.7, 7.5, color=S.GRID, alpha=0.5, lw=0)
    ax.text(5.5, 0.66, "R > J", ha="center", fontsize=6.4, color=S.INK2)
    ax.text(12.5, 0.66, "J > R", ha="center", fontsize=6.4, color=S.INK2)
    out = {}
    for alpha, col, lab in ((0.5, S.LIGHTGRAY, "single-layer swap, α = 0.5"), (1.0, S.ORANGE, "single-layer swap, α = 1")):
        ms, lo, hi = [], [], []
        for l in layers:
            d = paired_ci(per_item(sel(sw, lens="R", layer=l, alpha=alpha)), per_item(sel(sw, lens="J", layer=l, alpha=alpha)))
            ms.append(d["mean_diff"]); lo.append(d["lo95"]); hi.append(d["hi95"])
            out[(alpha, l)] = d
        ax.errorbar(layers, ms, yerr=[np.array(ms) - np.array(lo), np.array(hi) - np.array(ms)], fmt="-o", color=col, ms=3.4,
                    lw=1.3, elinewidth=0.8, capsize=0, label=lab)
    ax.axhline(0, color=S.LIGHTGRAY, lw=0.7)
    ax.set_ylim(-1.2, 0.8)
    ax.set_xlabel("layer of a single-layer swap")
    ax.set_ylabel("R − J, paired Δmargin (nat), 95% CI")
    S.title(ax, "R beats J only below L8")
    ax.legend(loc="lower left", fontsize=6.0)
    return out


def _bands(ax):
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    r38 = load_jsonl("block27_4b_band3_8/replication.jsonl")
    r820 = load_jsonl("block27_4b_band8_20/replication.jsonl")
    groups = [
        ("9B L3–8", {L: sel(c9, band="early_L3_8", lens=L, pair="entity", arm="full") for L in "JR"}),
        ("9B L8–20", {L: sel(c9, band="primary_L8_20", lens=L, pair="entity", arm="full") for L in "JR"}),
        ("4B L3–8", {L: sel(r38, stage="band", lens=L, arm="clamp", control="full") for L in "JR"}),
        ("4B L8–20", {L: sel(r820, stage="band", lens=L, arm="clamp", control="full") for L in "JR"}),
    ]
    out = []
    step = 3.5
    for g, (lab, d) in enumerate(groups):
        for k, (L, col) in enumerate((("J", S.BLUE), ("R", S.ORANGE))):
            m, lo, hi, n = rate_ci(d[L])
            S.ci_bar(ax, g * step + k * 0.9, m, lo, hi, col, width=0.8, label=f"{L}-lens clamp" if g == 0 else None)
        pd = paired_ci(per_item(d["R"]), per_item(d["J"]))
        out.append((lab, pd))
        sig = "*" if not (pd["lo95"] <= 0 <= pd["hi95"]) else ""
        ax.text(g * step + 0.45, 0.61, f"R − J\n{pd['mean_diff']:+.2f}{sig}\n[{pd['lo95']:+.2f}, {pd['hi95']:+.2f}]",
                ha="center", va="top", fontsize=5.0, color=S.INK2, linespacing=1.2)
    ax.set_xticks([g * step + 0.45 for g in range(len(groups))])
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.4)
    ax.set_xlim(-0.7, (len(groups) - 1) * step + 1.6)
    ax.set_ylim(0, 0.64)
    ax.set_ylabel("clamp flip rate (95% CI)")
    S.title(ax, "clamp flip rates: R's early-band edge\nin two models, none in the working band")
    ax.legend(loc="upper left", fontsize=6.0, bbox_to_anchor=(0.0, 0.8))
    S.ygrid(ax)
    return out


def _energy(ax):
    rows = [r for r in load_jsonl("block60_selfconsistent/selfconsistent.jsonl") if r["part"] == "a"]
    pts = {}
    for arm in ("J@1", "J@1.5", "J@2", "R@0.8", "R@1", "R@energy_of_J"):
        rs = sel(rows, arm=arm)
        pts[arm] = (float(np.median([r["energy"] for r in rs])), float(np.mean([r["top1_is_swap"] for r in rs])))
    for arms, col, lab in ((("J@1", "J@1.5", "J@2"), S.BLUE, "J-lens clamp, scale 1 → 1.5 → 2"), (("R@0.8", "R@1"), S.ORANGE, "R-lens clamp, scale 0.8 → 1")):
        xs = [pts[a][0] for a in arms]; ys = [pts[a][1] for a in arms]
        ax.plot(xs, ys, "-o", color=col, ms=4, lw=1.2, label=lab)
    e, f = pts["R@energy_of_J"]
    ax.plot(e, f, "D", color=S.ORANGE, ms=5, mec=S.INK, mew=0.6, label="R rescaled to J's energy")
    d = paired_ci(item_values(sel(rows, arm="R@energy_of_J"), "delta_margin"), item_values(sel(rows, arm="J@1"), "delta_margin"))
    S.note(ax, f"R at J's energy still flips {f:.2f};\nΔmargin vs J at scale 1: {d['mean_diff']:+.2f} [{d['lo95']:+.2f}, {d['hi95']:+.2f}]",
           x=0.03, y=0.97)
    ax.set_xlabel("median injected energy Σ‖δ‖², early band L3–8")
    ax.set_ylabel("flip rate")
    ax.set_ylim(0.08, 0.36)
    S.title(ax, "the early-band edge survives\nenergy matching")
    ax.legend(loc="lower right", fontsize=6.0)
    S.ygrid(ax)
    return pts, d


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.45)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    sweep = _sweep(ax_a)
    bands = _bands(ax_b)
    pts, d = _energy(ax_c)
    for ax, letter, dx in ((ax_a, "a", -0.3), (ax_b, "b", -0.2), (ax_c, "c", -0.3)):
        S.panel_label(ax, letter, dx=dx, dy=1.12)
    S.headline(fig, "The original question: R-lens does not beat J-lens in the working band; its small early-layer edge is real and only shows once the protocol is fixed", y=1.04)
    S.save(fig, "Fig5_R_vs_J")
    for (alpha, l), dd in sweep.items():
        if alpha == 1.0:
            print(f"  sweep L{l} alpha 1: {dd['mean_diff']:+.2f} [{dd['lo95']:+.2f}, {dd['hi95']:+.2f}]")
    for lab, pd in bands:
        print(f"  band {lab}: R-J {pd['mean_diff']:+.2f} [{pd['lo95']:+.2f}, {pd['hi95']:+.2f}] n={pd['n']}")
    for a, (e, f) in pts.items():
        print(f"  energy {a}: E={e:.1f} flip={f:.2f}")
    print(f"  R@energy_of_J − J@1: {d['mean_diff']:+.2f} [{d['lo95']:+.2f}, {d['hi95']:+.2f}]")


if __name__ == "__main__":
    main()
