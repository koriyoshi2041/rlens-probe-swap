"""Fig 9: the model regenerates the missing content from a single paste, and the MLPs at
L18-20 do the regenerating."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .data import load_jsonl

BAND = list(range(8, 21))


def _propagation(ax):
    r64 = load_jsonl("block64_propagation/propagation.jsonl")
    r72 = load_jsonl("block72_subspace_controls/subspace_controls.jsonl")
    series = [("full", r64, "full donor vector", S.INK2, "-"), ("J+PCA256", r72, "plane + residual PCA-256", S.VIOLET, "-"),
              ("J+R256", r64, "plane + lens top-256", S.BLUE, "-"), ("J+rand256", r64, "plane + 256 random dirs", S.GRAY, "--"),
              ("J_plane", r64, "plane only (2-D)", S.BLUE_LIGHT, ":")]
    out = {}
    for sub, src, lab, col, ls in series:
        med = [float(np.median([r["arms"][sub]["single_L8"]["realised"][i] for r in src])) for i in range(len(BAND))]
        out[sub] = med
        ax.plot(BAND, med, ls, color=col, marker="o", ms=3, lw=1.3)
        ax.text(BAND[-1] + 0.35, med[-1] + (0.02 if sub == "J+rand256" else (-0.03 if sub == "J_plane" else 0)), lab,
                va="center", ha="left", fontsize=6.0, color=S.INK2)
    ax.set_xlim(7.5, 26)
    ax.set_xlabel("layer (paste applied once, at L8, at the bridge position)")
    ax.set_ylabel("share of the donor difference realised\n⟨h_edit − h_clean, d_l⟩ / ‖d_l‖² (median)")
    S.title(ax, "after one paste the model completes the donor state itself,\nbut only from lens or principal-component content")
    ax.set_xticks(BAND)
    S.ygrid(ax)
    return out


def _growth(ax):
    r77 = load_jsonl("block77_growth_pca/growth_pca.jsonl")
    subs = [("J+R256", "plane +\nlens top-256", S.BLUE), ("J+PCA256", "plane +\nPCA-256", S.VIOLET), ("full", "full donor\nvector", S.INK2), ("J+rand256", "plane +\nrandom 256", S.GRAY)]
    out = {}
    for k, (sub, lab, col) in enumerate(subs):
        attn, mlp, tot = [], [], []
        for r in r77:
            g = r["growth"][sub]
            base = max(g["realised_L8_on_d20"], 1e-6)
            attn.append(sum(l["attn_on_d20"] for l in g["layers"]) / base)
            mlp.append(sum(l["mlp_on_d20"] for l in g["layers"]) / base)
            tot.append(g["layers"][-1]["realised_d20"] / base)
        a, m, t = float(np.median(attn)), float(np.median(mlp)), float(np.median(tot))
        out[sub] = (a, m, t)
        ax.bar(k, 1.0, 0.6, color=S.LIGHTGRAY, zorder=2, label="put in by the L8 paste" if k == 0 else None)
        ax.bar(k, a, 0.6, bottom=1.0, color=S.VIOLET, zorder=2, label="added by attention, L9–20" if k == 0 else None)
        ax.bar(k, m, 0.6, bottom=1.0 + a, color=S.YELLOW, zorder=2, label="added by MLPs, L9–20 (mostly L18–20)" if k == 0 else None)
        ax.text(k, 1.0 + a + m + 0.12, f"×{t:.1f}", ha="center", fontsize=6.4, color=S.INK)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([s[1] for s in subs], fontsize=6.2, linespacing=1.15)
    ax.set_ylabel("realised share at L20, relative to\nwhat the L8 paste put in (= 1)")
    S.title(ax, "the growth from L8 to L20\ncomes from the MLPs")
    ax.legend(loc="upper left", fontsize=5.6, ncol=1)
    ax.set_ylim(0, 7.0)
    S.ygrid(ax)
    return out


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.35)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    prop = _propagation(ax_a)
    grow = _growth(ax_b)
    S.panel_label(ax_a, "a", dx=-0.16, dy=1.12)
    S.panel_label(ax_b, "b", dx=-0.22, dy=1.12)
    S.headline(fig, "Content pasted into the lens or principal-component directions is completed by the MLPs at L18–20 before the transport stage reads it", y=1.04)
    S.save(fig, "Fig9_generation")
    for k, v in prop.items():
        print(f"  realised {k}: L8 {v[0]:.2f} -> L20 {v[-1]:.2f}")
    for k, v in grow.items():
        print(f"  growth {k}: attn +{v[0]:.2f}, mlp +{v[1]:.2f}, total x{v[2]:.2f}")


if __name__ == "__main__":
    main()
