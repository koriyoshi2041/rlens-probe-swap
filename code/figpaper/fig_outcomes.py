"""Fig. 5 - what happens to the answer: hand-classified continuations under the clamp; token-level under the published swap."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from data import RES, load_jsonl, sel
from style import BLUE, BLUE_LIGHT, GRAY, GRAY_LIGHT, INK, INK2, RED, WHITE, apply, save


def _segments_clamp():
    mc = json.load(open(RES / "block23_semantic" / "manual_classification.json"))
    sup = sel(load_jsonl("block12_suppression/suppression.jsonl"), lens="J", arm="clamp")
    strict = sum(bool(r["top1_is_swap"]) for r in sup)
    semantic = len(mc["rewrite_semantic_only"])
    damaged = len(mc["damaged"])
    unchanged = len(sup) - strict - semantic - damaged
    return [("rewritten, exact token", strict, BLUE), ("rewritten, other token", semantic, BLUE_LIGHT), ("damaged", damaged, RED), ("unchanged", unchanged, GRAY_LIGHT)]


def _segments_involution():
    b2 = sel(load_jsonl("block02/block02_records.jsonl"), condition="swap_raw", lens="J", band="primary_L8_20", alpha=1.0, positions="all")
    by = {}
    for r in b2:
        by.setdefault(int(r["index"]), []).append(r)
    flips = sum(all(x["top1_is_swap"] for x in v) for v in by.values())
    kept = sum(all(x["top1_is_answer"] for x in v) for v in by.values())
    other = len(by) - flips - kept
    return [("rewritten, exact token", flips, BLUE), ("other token", other, RED), ("unchanged", kept, GRAY_LIGHT)]


def _draw_row(ax, y, segs):
    left = 0
    for label, n, col in segs:
        if n <= 0:
            continue
        ax.barh(y, n, left=left, height=0.55, color=col, edgecolor=WHITE, linewidth=1.5, zorder=2)
        txt_col = WHITE if col in (BLUE, RED) else INK
        if n >= 5:
            ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=7.5, color=txt_col)
        else:
            ax.text(left + n / 2, y + 0.36, str(n), ha="center", va="bottom", fontsize=7, color=INK2)
        left += n


def make() -> None:
    apply()
    fig, ax = plt.subplots(figsize=(5.2, 1.75))
    _draw_row(ax, 1, _segments_clamp())
    _draw_row(ax, 0, _segments_involution())
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["idempotent clamp\n(hand-read continuations)", "published swap\n(next token)"])
    ax.set_xlim(0, 59)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 59])
    ax.set_xlabel("items (n = 59), J-lens, band L8–20")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, BLUE_LIGHT, RED, GRAY_LIGHT)]
    ax.legend(handles, ["rewritten, exact token", "rewritten, other token", "damaged / other token", "unchanged"], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32), columnspacing=1.0, handlelength=1.0)
    save(fig, "Fig5_answer_outcomes")


if __name__ == "__main__":
    make()
