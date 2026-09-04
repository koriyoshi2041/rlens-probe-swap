"""Fig 3: where the lens edit sits between cheap baselines and a full activation patch,
and how much of its effect is a first-order push on the answer logit."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .data import load_jsonl, per_item, rate_ci, sel

P = "primary_L8_20"


def _bars():
    b2 = load_jsonl("block02/block02_records.jsonl")
    lad = load_jsonl("block10_ladder/ladder.jsonl")
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    sub = load_jsonl("block10_ladder/subspace_patch.jsonl")
    pat = load_jsonl("block07_causal/patch.jsonl")
    return [
        ("random plane (Gram-matched Gaussian)", sel(b2, condition="swap_gauss", lens="J", band=P, alpha=1.0, positions="all"), S.GRAY),
        ("mismatched entity pair (another item's plane)", sel(b2, condition="swap_shuffled", lens="J", band=P, alpha=1.0, positions="all"), S.GRAY),
        ("logit lens (identity transport), clamp", sel(lad, lens="logit", arm="clamp", scale=1.0), S.GRAY),
        ("published swap (involution), J-lens", sel(b2, condition="swap_raw", lens="J", band=P, alpha=1.0, positions="all"), S.RED),
        ("J-lens clamp, answer push projected out", sel(c9, band=P, lens="J", pair="entity", arm="ortho_rescaled"), S.BLUE_LIGHT),
        ("J-lens clamp", sel(c9, band=P, lens="J", pair="entity", arm="full"), S.BLUE),
        ("J-lens clamp, ×2 over-drive", sel(lad, lens="J", arm="clamp", scale=2.0), S.BLUE),
        ("J-lens clamp, 14 minimal pairs", sel(sub, lens="J", arm="clamp_exchange"), S.BLUE),
        ("full-layer activation patch, same 14 pairs", sel(pat, arm="patch_full", lens="J"), S.INK2),
    ]


def _retention_points():
    """Share of the clamp Δmargin that survives projecting out the answer-contrast direction."""
    pts = []
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    for band, lab, centre in (("early_L3_8", "9B L3–8", 5.5 / 32), ("primary_L8_20", "9B L8–20", 14 / 32)):
        full = per_item(sel(c9, band=band, lens="J", pair="entity", arm="full"))
        orth = per_item(sel(c9, band=band, lens="J", pair="entity", arm="ortho_rescaled"))
        pts.append((centre, np.mean(list(orth.values())) / np.mean(list(full.values())), lab, "o"))
    for d, lab, centre in (("block27_4b_band3_8", "4B L3–8", 5.5 / 32), ("block27_4b_band8_20", "4B L8–20", 14 / 32), ("block27_4b", "4B L24–29", 26.5 / 32)):
        rep = load_jsonl(f"{d}/replication.jsonl")
        full = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="full"))
        orth = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="ortho_rescaled"))
        pts.append((centre, np.mean(list(orth.values())) / np.mean(list(full.values())), lab, "s"))
    rep = load_jsonl("block81_qwen3_4b/replication.jsonl")
    full = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="full"))
    orth = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="ortho_rescaled"))
    pts.append((15 / 36, np.mean(list(orth.values())) / np.mean(list(full.values())), "Qwen3-4B L10–20", "^"))
    return pts


OFFSETS = {"9B L3–8": (6, -9), "4B L3–8": (6, 4), "9B L8–20": (6, -9), "4B L8–20": (6, 4), "4B L24–29": (-6, 4), "Qwen3-4B L10–20": (6, 5)}


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    bars = _bars()
    ys = np.arange(len(bars))[::-1]
    vals = {}
    for y, (lab, rows, col) in zip(ys, bars):
        m, lo, hi, n = rate_ci(rows)
        vals[lab] = (m, lo, hi, n, y)
        ax.barh(y, m, height=0.62, color=col, zorder=2)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=S.INK2, elinewidth=0.7, capsize=0, zorder=3)
        ax.text(hi + 0.015, y, f"{m:.2f}", va="center", fontsize=6.6, color=S.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("flip rate, band L8–20, all prompt positions (95% cluster-bootstrap CI)")
    S.title(ax, "2-D lens edits, from cheap baselines to a full activation patch")
    y_clamp = vals["J-lens clamp"][4]
    y_orth = vals["J-lens clamp, answer push projected out"][4]
    m_clamp = vals["J-lens clamp"][0]
    m_orth = vals["J-lens clamp, answer push projected out"][0]
    ymid = (y_clamp + y_orth) / 2
    ax.annotate("", xy=(m_orth, ymid), xytext=(m_clamp, ymid),
                arrowprops={"arrowstyle": "<->", "color": S.INK2, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.62, ymid, "first-order push on the answer logit\n(positive control: clamping the answer\npair itself projects out to 0.02)",
            fontsize=5.9, color=S.INK2, ha="left", va="center", linespacing=1.25)
    S.panel_label(ax, "a", dx=-0.66)

    ax2 = fig.add_subplot(gs[0, 1])
    pts = _retention_points()
    for centre, ret, lab, mk in pts:
        col = S.INK2 if "Qwen3-4B" in lab else S.BLUE
        ax2.plot(centre, ret, marker=mk, color=col, ms=5.5, ls="none")
        ax2.annotate(lab, (centre, ret), textcoords="offset points", xytext=OFFSETS[lab], fontsize=6.0, color=S.INK2,
                     ha="right" if OFFSETS[lab][0] < 0 else "left")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0.05, 0.95)
    ax2.set_xlabel("band centre (fraction of model depth)")
    ax2.set_ylabel("share of the clamp effect retained\nafter projecting out the direct push")
    S.title(ax2, "mediated share by band: low only\nwhere the answer is already forming")
    S.ygrid(ax2)
    S.panel_label(ax2, "b", dx=-0.32, dy=1.12)
    S.headline(fig, "The published swap barely beats the baselines; about a quarter of the clamp's effect is a direct push on the answer logit", y=1.04)
    S.save(fig, "Fig3_ladder")
    for lab, (m, lo, hi, n, _) in vals.items():
        print(f"  {lab}: {m:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
    for centre, ret, lab, _ in pts:
        print(f"  retained {lab}: {ret:.2f}")


if __name__ == "__main__":
    main()
