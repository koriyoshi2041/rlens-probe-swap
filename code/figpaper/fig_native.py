"""Fig. 16 - the model's own two-hop answers: ablating the transport stage hurts, ablating heads or single layers does not."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, sel
from style import GRAY, INK2, RULE, VIOLET, WHITE, apply, save, xgrid


def _acc(rows, kind):
    rs = [r for r in rows if r["prompt_kind"] == kind and r["clean_correct"]]
    return float(np.mean([r["still_correct"] for r in rs])) if rs else np.nan


def _b60(arm, kind):
    rows = sel(load_jsonl("block60_selfconsistent/selfconsistent.jsonl"), part="b", arm=arm)
    return _acc(rows, kind)


def _b82_window(kind, start=17):
    rows = load_jsonl("block82_qwen3_4b_mechanism/mechanism.jsonl")
    rs = [r for r in rows if r["stage"][kind]["clean_correct"]]
    vals = [w["still_correct"] for r in rs for w in r["stage"][kind]["windows"] if w["start"] == start]
    return float(np.mean(vals))


def make() -> None:
    apply()
    b79 = load_jsonl("block79_clean_head_ablation/head_ablation.jsonl")
    b79b = load_jsonl("block79_4b_clean_head_ablation/head_ablation.jsonl")
    b84 = load_jsonl("block84_qwen3_4b_clean_head_ablation_zero/head_ablation.jsonl")
    # rows: (label, {model: (two_hop, single_hop)})
    table = [
        ("attention L17–24 (stage)", {"9B": (_b60("attn_17_24", "two_hop"), _b60("attn_17_24", "single_hop")), "Q3": (_b82_window("two_hop"), _b82_window("single_hop"))}),
        ("attention L9–16 (control)", {"9B": (_b60("attn_9_16", "two_hop"), _b60("attn_9_16", "single_hop"))}),
        ("MLP L26–29", {"9B": (_b60("mlp_26_29", "two_hop"), _b60("mlp_26_29", "single_hop"))}),
        ("all heads, L23", {"9B": (_acc(sel(b79, arm="all@23"), "two_hop"), _acc(sel(b79, arm="all@23"), "single_hop")), "4B": (_acc(sel(b79b, arm="all@23"), "two_hop"), _acc(sel(b79b, arm="all@23"), "single_hop")), "Q3": (_acc(sel(b84, arm="all@23"), "two_hop"), _acc(sel(b84, arm="all@23"), "single_hop"))}),
        ("all heads, L19", {"9B": (_acc(sel(b79, arm="all@19"), "two_hop"), _acc(sel(b79, arm="all@19"), "single_hop")), "4B": (_acc(sel(b79b, arm="all@19"), "two_hop"), _acc(sel(b79b, arm="all@19"), "single_hop")), "Q3": (_acc(sel(b84, arm="all@19"), "two_hop"), _acc(sel(b84, arm="all@19"), "single_hop"))}),
        ("transport head (L23 h8 / h10)", {"9B": (_acc(sel(b79, arm="h8"), "two_hop"), _acc(sel(b79, arm="h8"), "single_hop")), "4B": (_acc(sel(b79b, arm="h8"), "two_hop"), _acc(sel(b79b, arm="h8"), "single_hop")), "Q3": (_acc(sel(b84, arm="h10"), "two_hop"), _acc(sel(b84, arm="h10"), "single_hop"))}),
    ]
    markers = {"9B": ("o", "Qwen3.5-9B"), "4B": ("s", "Qwen3.5-4B"), "Q3": ("^", "Qwen3-4B (dense)")}
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ys = np.arange(len(table))[::-1]
    for y, (label, per_model) in zip(ys, table):
        for j, (mk, (marker, _)) in enumerate(markers.items()):
            if mk not in per_model:
                continue
            two, one = per_model[mk]
            dy = (j - 1) * 0.22
            ax.plot([one], [y + dy], marker=marker, ms=5, color=GRAY, mfc=WHITE, mew=1.0, ls="none", zorder=3)
            ax.plot([two], [y + dy], marker=marker, ms=5, color=VIOLET, ls="none", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([t[0] for t in table])
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("accuracy after mean-ablation at the final position")
    xgrid(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h1 = [plt.Line2D([], [], marker=m, color=INK2, ls="none", ms=5, label=lab) for m, lab in markers.values()]
    h2 = [plt.Line2D([], [], marker="o", color=VIOLET, ls="none", ms=5, label="two-hop prompt"), plt.Line2D([], [], marker="o", color=GRAY, mfc=WHITE, mew=1.0, ls="none", ms=5, label="single-hop prompt")]
    leg1 = ax.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(0.36, 0.0), fontsize=7)
    save(fig, "Fig16_native_two_hop_ablation")


if __name__ == "__main__":
    make()
