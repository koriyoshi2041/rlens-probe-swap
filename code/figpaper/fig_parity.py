"""Fig. 2 - flip rate vs band width: published swap (involution) alternates, clamp is monotone."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from data import RES, flip_rate, load_jsonl, sel
from style import BLUE, RED, apply, save, ygrid


def _ladder(rows, arm, widths, **extra):
    return [flip_rate(sel(rows, lens="J", arm=arm, width=w, **extra)) for w in widths]


def make() -> None:
    apply()
    m3 = load_jsonl("block11_mechanism/m3_parity.jsonl")
    rep4 = load_jsonl("block27_4b_band8_20/replication.jsonl")
    rep3 = load_jsonl("block81_qwen3_4b/replication.jsonl")
    meta3 = json.load(open(RES / "block81_qwen3_4b" / "meta.json"))
    panels = [
        ("Qwen3.5-9B", m3, {}, 8),
        ("Qwen3.5-4B", rep4, {"stage": "ladder"}, 8),
        ("Qwen3-4B (dense attention)", rep3, {"stage": "ladder"}, meta3["band"][0]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True)
    for ax, (title, rows, extra, start) in zip(axes, panels):
        widths = sorted({int(r["width"]) for r in sel(rows, **extra)})
        n = len({r["index"] for r in sel(rows, lens="J", arm="clamp", width=widths[0], **extra)})
        ax.plot(widths, _ladder(rows, "clamp", widths, **extra), "-o", color=BLUE, label="idempotent clamp", zorder=3)
        ax.plot(widths, _ladder(rows, "involution", widths, **extra), "-o", color=RED, label="published swap", zorder=3)
        ax.set_title(f"{title}, n = {n}")
        ax.set_xticks(widths)
        ax.set_xlabel(f"band width (layers, from L{start})")
        ax.set_ylim(0, 0.5)
        ygrid(ax)
    axes[0].set_ylabel("flip rate (top-1 = target answer)")
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "Fig2_parity_ladder")


if __name__ == "__main__":
    make()
