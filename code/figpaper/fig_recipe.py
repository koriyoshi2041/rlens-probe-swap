"""Fig. 14 - routing assist without a donor: forcing the transport head onto the edited position, plus the 2-D clamp."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data import RES, bootstrap_named, clusters, load_jsonl, newitems, paired_ci, zero_set_9b, zero_set_from_replication
from style import BLUE, GRAY, INK2, RULE, apply, save

ARM = "F[h8@23,1]+clamp@4_best"
BASE = "clamp@4_best"


def _rows_indexed(rel):
    return {int(r["index"]): r for r in load_jsonl(rel)}


def _entry(rows, ids, label):
    a = {i: rows[i]["arms"][ARM]["delta_margin"] for i in ids}
    b = {i: rows[i]["arms"][BASE]["delta_margin"] for i in ids}
    d = paired_ci(a, b, clusters(ids))
    fa = np.mean([rows[i]["arms"][ARM]["top1_is_swap"] for i in ids])
    fb = np.mean([rows[i]["arms"][BASE]["top1_is_swap"] for i in ids])
    return label, d["mean_diff"], d["lo95"], d["hi95"], len(ids), fb, fa


def make() -> None:
    apply()
    entries = []
    models = [
        ("Qwen3.5-9B", "block75_donor_free_recipe/donor_free.jsonl", zero_set_9b()),
        ("Qwen3.5-4B", "block80_4b_donor_free_recipe/donor_free.jsonl", zero_set_from_replication("block27_4b_band8_20/replication.jsonl")),
        ("Qwen3-4B (dense)", "block83_qwen3_4b_donor_free_recipe/donor_free.jsonl", zero_set_from_replication("block81_qwen3_4b/replication.jsonl")),
    ]
    for name, rel, zero in models:
        rows = _rows_indexed(rel)
        ids = sorted(rows)
        entries.append(_entry(rows, [i for i in ids if i not in zero], f"{name}, flippable items"))
        entries.append(_entry(rows, ids, f"{name}, all items"))
    oos = {r["name"]: r for r in load_jsonl("block76_oos_recipe/oos_recipe.jsonl")}
    _, fact = newitems()
    names = sorted(oos)
    diff = {n: oos[n]["arms"][ARM]["delta_margin"] - oos[n]["arms"][BASE]["delta_margin"] for n in names}
    m, lo, hi = bootstrap_named(diff, {n: fact[n] for n in names})
    fb = np.mean([oos[n]["arms"][BASE]["top1_is_swap"] for n in names])
    fa = np.mean([oos[n]["arms"][ARM]["top1_is_swap"] for n in names])
    entries.append(("Qwen3.5-9B, held-out non-geographic items", m, lo, hi, len(names), fb, fa))

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ys = np.arange(len(entries))[::-1]
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    for y, (label, m, lo, hi, n, fb, fa) in zip(ys, entries):
        col = BLUE if "flippable" in label else GRAY
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=2)
        ax.plot([m], [y], "o", color=col, ms=5, zorder=3)
        ax.text(4.35, y, f"{fb:.2f} → {fa:.2f}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{e[0]} (n = {e[4]})" for e in entries])
    ax.set_xlim(-0.6, 5.4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("forced head + clamp ×4 − clamp ×4, paired Δmargin (nat)")
    ax.text(4.35, ys[0] + 0.85, "flip rate", ha="left", va="bottom", fontsize=7, color=INK2)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "Fig14_donor_free_recipe")


if __name__ == "__main__":
    make()
