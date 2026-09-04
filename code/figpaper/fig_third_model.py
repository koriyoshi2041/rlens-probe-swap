"""Fig. 18 - Qwen3-4B (dense attention): sliding-window ablation of attention outputs at the final position."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl
from style import GRAY, VIOLET, WHITE, apply, save, ygrid


def make() -> None:
    apply()
    rows = load_jsonl("block82_qwen3_4b_mechanism/mechanism.jsonl")
    starts = [w["start"] for w in rows[0]["stage"]["two_hop"]["windows"]]
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for kind, col, mfc, label in (("two_hop", VIOLET, VIOLET, "two-hop prompt"), ("single_hop", GRAY, WHITE, "single-hop prompt")):
        rs = [r for r in rows if r["stage"][kind]["clean_correct"]]
        acc = [float(np.mean([r["stage"][kind]["windows"][i]["still_correct"] for r in rs])) for i in range(len(starts))]
        ax.plot([s + 3.5 for s in starts], acc, "-o", color=col, mfc=mfc, mew=1.0, label=f"{label} (n = {len(rs)})")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([s + 3.5 for s in starts][::2])
    ax.set_xlabel("centre of the ablated 8-layer window")
    ax.set_ylabel("accuracy after ablation")
    ax.legend(loc="lower left")
    ygrid(ax)
    save(fig, "Fig18_third_model_stage")


if __name__ == "__main__":
    make()
