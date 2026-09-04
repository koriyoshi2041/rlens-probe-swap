"""Fig. 6 - when the answer changes, the model's own report of the bridge entity changes with it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import load_jsonl, sel
from style import BLUE, GRAY, RULE, apply, save


def make() -> None:
    apply()
    rows = load_jsonl("block26_entity_probe/entity_probe.jsonl")
    clean = {r["index"]: r for r in sel(rows, arm="clean", probe="q_noans")}
    arm = {r["index"]: r for r in sel(rows, arm="J", probe="q_noans")}
    idx = sorted(set(clean) & set(arm))
    da = np.array([arm[i]["answer_margin"] - clean[i]["answer_margin"] for i in idx])
    de = np.array([arm[i]["entity_margin"] - clean[i]["entity_margin"] for i in idx])
    fl = np.array([bool(arm[i]["top1_is_swap_answer"]) for i in idx])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.axhline(0, color=RULE, lw=0.8, zorder=1)
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    ax.scatter(da[~fl], de[~fl], s=16, color=GRAY, edgecolor="white", linewidth=0.5, label=f"answer unchanged (n = {int((~fl).sum())})", zorder=3)
    ax.scatter(da[fl], de[fl], s=16, color=BLUE, edgecolor="white", linewidth=0.5, label=f"answer flips (n = {int(fl.sum())})", zorder=4)
    ax.set_xlabel("Δ answer margin at the prompt (nat)")
    ax.set_ylabel("Δ entity-report margin at the probe (nat)")
    ax.legend(loc="lower right")
    save(fig, "Fig6_entity_probe")


if __name__ == "__main__":
    make()
