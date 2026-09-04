"""Fig 8: the L17-24 attention stage is necessary for the edit and for the model's own
two-hop answer; single heads and single layers are redundant."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .data import load_jsonl, rate_ci, sel


def _restore(ax):
    rows = load_jsonl("block39_stage_necessity/stage_necessity.jsonl")
    arms = [("clamp_only", "clamp\nonly", S.BLUE), ("attn_9_16@final", "attn\nL9–16\n(control)", S.VIOLET_LIGHT),
            ("attn_17_24@final", "attn\nL17–24", S.VIOLET), ("attn_25_31@final", "attn\nL25–31", S.VIOLET_LIGHT),
            ("mlp_17_24@final", "MLP\nL17–24", S.YELLOW_LIGHT), ("mlp_26_29@final", "MLP\nL26–29", S.YELLOW)]
    out = []
    for k, (arm, lab, col) in enumerate(arms):
        m, lo, hi, n = rate_ci(sel(rows, arm=arm))
        out.append((lab.replace("\n", " "), m, lo, hi))
        S.ci_bar(ax, k, m, lo, hi, col)
        ax.text(k, hi + 0.015, f"{m:.2f}", ha="center", fontsize=6.2, color=S.INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.0, linespacing=1.15)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("flip rate, clamp L8–20 (95% CI)")
    ax.set_xlabel("group restored to its clean output at the final position")
    S.title(ax, "restoring the L17–24 attention\noutputs removes the edit")
    S.ygrid(ax)
    return out


def _clean_run(ax):
    rows = [r for r in load_jsonl("block60_selfconsistent/selfconsistent.jsonl") if r["part"] == "b"]
    arms = [("attn_9_16", "attn\nL9–16\n(control)"), ("attn_17_24", "attn\nL17–24"), ("mlp_26_29", "MLP\nL26–29")]
    out = {}
    for k, (kind, col, lab) in enumerate((("two_hop", S.INK2, "two-hop prompt"), ("single_hop", S.LIGHTGRAY, "single-hop control"))):
        vals = []
        for arm, _ in arms:
            rs = [r for r in rows if r["arm"] == arm and r["prompt_kind"] == kind and r["clean_correct"]]
            vals.append(float(np.mean([r["still_correct"] for r in rs])))
        out[kind] = vals
        xs = np.arange(len(arms)) + (k - 0.5) * 0.36
        ax.bar(xs, vals, 0.34, color=col, label=lab, zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.0, color=S.INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=6.0, linespacing=1.15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy after mean-ablation, no lens edit")
    ax.set_xlabel("group ablated at the final position")
    S.title(ax, "the model's own two-hop\nanswer uses this stage", pad=16)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.6, columnspacing=0.8)
    S.ygrid(ax)
    return out


def _heads(ax):
    specs = [("Qwen3.5-9B, mean-ablation", "block79_clean_head_ablation/head_ablation.jsonl", "o", S.BLUE, {"h8": "1 head", "h8h9h0": "3 heads", "all@23": "all L23", "all@19": "all L19", "rand3@23": "3 random"}),
             ("Qwen3.5-9B, zero-ablation", "block79_clean_head_ablation_zero/head_ablation.jsonl", "o", S.BLUE_LIGHT, {"h8": "1 head", "h8h9h0": "3 heads", "all@23": "all L23", "all@19": "all L19", "rand3@23": "3 random"}),
             ("Qwen3.5-4B, mean-ablation", "block79_4b_clean_head_ablation/head_ablation.jsonl", "s", S.BLUE, {"h8": "1 head", "h8h9h0": "3 heads", "all@23": "all L23", "all@19": "all L19", "rand3@23": "3 random"}),
             ("Qwen3-4B dense, zero-ablation", "block84_qwen3_4b_clean_head_ablation_zero/head_ablation.jsonl", "^", S.INK2, {"h10": "1 head", "all@23": "all L23", "all@19": "all L19", "rand3@23": "3 random"})]
    keys = ["1 head", "3 heads", "3 random", "all L23", "all L19", "stage"]
    cats = ["1\nhead", "3\nheads", "3 random\nheads", "all\nL23", "all\nL19", "whole\nstage\nL17–24"]
    xi = {c: i for i, c in enumerate(keys)}
    out = {}
    for name, rel, mk, col, amap in specs:
        rows = load_jsonl(rel)
        pts = []
        for arm, cat in amap.items():
            rs = [r for r in rows if r["arm"] == arm and r["prompt_kind"] == "two_hop" and r["clean_correct"]]
            if rs:
                pts.append((xi[cat], float(np.mean([r["still_correct"] for r in rs]))))
        out[name] = pts
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col, ms=5, ls="none", label=name, mec=S.SURFACE, mew=0.5)
    stage = [r for r in load_jsonl("block60_selfconsistent/selfconsistent.jsonl") if r["part"] == "b" and r["arm"] == "attn_17_24" and r["prompt_kind"] == "two_hop" and r["clean_correct"]]
    s9 = float(np.mean([r["still_correct"] for r in stage]))
    ax.plot(xi["stage"], s9, "o", color=S.BLUE, ms=5, mec=S.SURFACE, mew=0.5)
    mech = load_jsonl("block82_qwen3_4b_mechanism/mechanism.jsonl")
    win = [r for r in mech if r["stage"]["two_hop"]["clean_correct"]]
    s3 = float(np.mean([[w for w in r["stage"]["two_hop"]["windows"] if w["start"] == 17][0]["still_correct"] for r in win]))
    ax.plot(xi["stage"], s3, "^", color=S.INK2, ms=5, mec=S.SURFACE, mew=0.5)
    out["stage"] = {"9B": s9, "Qwen3-4B": s3}
    ax.axhline(1.0, color=S.LIGHTGRAY, lw=0.7)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=5.8, linespacing=1.15)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("two-hop accuracy after ablating\nthe unit at the final position")
    S.title(ax, "no single head or layer is\nneeded; the whole stage is")
    ax.legend(loc="lower left", fontsize=5.6)
    S.ygrid(ax)
    return out


def main() -> None:
    S.apply()
    fig = plt.figure(figsize=(S.FULL, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1.25], wspace=0.55)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    a = _restore(ax_a)
    b = _clean_run(ax_b)
    c = _heads(ax_c)
    for ax, letter, dx in ((ax_a, "a", -0.16), (ax_b, "b", -0.3), (ax_c, "c", -0.18)):
        S.panel_label(ax, letter, dx=dx, dy=1.14)
    S.headline(fig, "The edit reaches the answer through the L17–24 attention stage, which the model itself relies on for two-hop recall;\ninside that stage, heads and layers are redundant", y=1.06)
    S.save(fig, "Fig8_transport")
    for lab, m, lo, hi in a:
        print(f"  restore {lab}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"  clean-run: {b}")
    for k, v in c.items():
        print(f"  heads {k}: {v}")


if __name__ == "__main__":
    main()
