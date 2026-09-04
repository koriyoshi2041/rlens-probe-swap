"""Fig 7: a single-position edit needs content and routing; routing can be supplied
without any donor, in three models and out of sample."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .data import (band_flip_9b, forced_head_arm, load_jsonl, paired_ci, paired_ci_named, rate, three_groups_9b,
                   zero_set)

GROUP_COLS = (S.AQUA, S.BLUE, S.GRAY)


def _groups():
    flip, resc, fail = three_groups_9b()
    return [(f"flips\n({len(flip)})", flip), (f"×4\nrescues\n({len(resc)})", resc), (f"×4\nfails\n({len(fail)})", fail)]


def _routing(ax, groups):
    r63 = {r["index"]: r for r in load_jsonl("block63_why_plane_ignored/why_plane_ignored.jsonl")}
    rng = np.random.default_rng(0)
    meds = []
    for k, ((lab, ids), col) in enumerate(zip(groups, GROUP_COLS)):
        ys = [r63[i]["attention"]["clean"]["L23_h8_to_bridge"] for i in ids]
        ax.scatter(k + rng.uniform(-0.2, 0.2, len(ys)), ys, s=12, color=col, alpha=0.85, zorder=2, linewidths=0)
        m = float(np.median(ys))
        meds.append(m)
        ax.hlines(m, k - 0.32, k + 0.32, color=S.INK, lw=1.3, zorder=3)
        ax.text(k + 0.36, m, f"{m:.2f}", va="center", fontsize=6.0, color=S.INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.6, linespacing=1.15)
    ax.set_xlabel("band clamp outcome", fontsize=6.0)
    ax.set_ylabel("L23 h8 attention onto the bridge position, clean run")
    S.title(ax, "before any edit: does the\ntransport head read p?")
    ax.set_xlim(-0.6, 2.8)
    S.ygrid(ax)
    return meds


def _transplants(ax, groups):
    r66 = {r["index"]: r for r in load_jsonl("block66_key_value/key_value.jsonl")}
    r73 = {r["index"]: r for r in load_jsonl("block73_routing_all_layers/routing_all_layers.jsonl")}
    arms = [("clamp2d@4", "clamp\n×4", r66), ("K@19,23", "keys\nonly", r66), ("K@19,23+clamp2d@4", "keys +\nclamp", r66),
            ("KV@19,23", "keys +\nvalues", r66), ("transport<-clamp", "L17–24\n← clamped\nstate", r73),
            ("KV19,23+lin_in", "L17–24\n← pasted\nstate", r73), ("orth@0.25", "paste\noutside\nplane", r66)]
    x = np.arange(len(arms))
    w = 0.27
    vals = {}
    fail_ids = groups[2][1]
    att_clean = np.median([r66[i]["arms"]["clean"]["L23_h8_to_bridge"] for i in fail_ids])
    att_keys = np.median([r66[i]["arms"]["K@19,23"]["L23_h8_to_bridge"] for i in fail_ids])
    for k, ((lab, ids), col) in enumerate(zip(groups, GROUP_COLS)):
        v = [rate([src[i]["arms"][a] for i in ids if i in src]) for a, _, src in arms]
        vals[lab] = v
        ax.bar(x + (k - 1) * w, v, w * 0.95, color=col, label=lab.replace("\n", " "), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=5.6, linespacing=1.15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flip rate")
    ax.set_xlabel("what the transport layers receive at the bridge position\n(keys and values taken from the paste run)")
    ax.set_title(f"routing alone does nothing: keys move h8's attention\n{att_clean:.2f} → {att_keys:.2f} but not the answer; "
                 "routing + content\nflips nearly everything that has content", loc="left", pad=17, fontsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=5.7, columnspacing=1.0)
    S.ygrid(ax)
    return vals, att_clean, att_keys


def _recipe(ax):
    rows_out = []
    specs = [("9B h8", "block75_donor_free_recipe/donor_free.jsonl", None),
             ("4B h8", "block80_4b_donor_free_recipe/donor_free.jsonl", "block27_4b_band8_20/replication.jsonl"),
             ("Q3-4B h10", "block83_qwen3_4b_donor_free_recipe/donor_free.jsonl", "block81_qwen3_4b/replication.jsonl")]
    bf9 = band_flip_9b()
    y = 0.0
    ticks, labels = [], []
    for name, rel, zero_rel in specs:
        rows = load_jsonl(rel)
        zero = {i for i, f in bf9.items() if not f} if zero_rel is None else zero_set(zero_rel)
        arm = forced_head_arm(rows[0]["arms"])
        for sub, col, keep in (("flips", S.AQUA, lambda r: r["index"] not in zero), ("all", S.GRAY, lambda r: True)):
            rs = [r for r in rows if keep(r)]
            A = {r["index"]: r["arms"][arm]["delta_margin"] for r in rs}
            B = {r["index"]: r["arms"]["clamp@4_best"]["delta_margin"] for r in rs}
            d = paired_ci(A, B)
            fa = rate([r["arms"][arm] for r in rs]); fb = rate([r["arms"]["clamp@4_best"] for r in rs])
            ax.errorbar(d["mean_diff"], y, xerr=[[d["mean_diff"] - d["lo95"]], [d["hi95"] - d["mean_diff"]]], fmt="o", color=col, ms=3.8, elinewidth=0.9, capsize=0)
            ax.text(4.3, y, f"{fb:.2f} → {fa:.2f}", va="center", fontsize=5.8, color=S.INK2)
            ticks.append(y); labels.append(f"{name} · {sub} ({len(rs)})")
            rows_out.append((name, sub, d, fb, fa))
            y -= 1
        y -= 0.45
    oos = load_jsonl("block76_oos_recipe/oos_recipe.jsonl")
    for arm, lab in (("F[h8@23,1]+clamp@4_best", "force h8"), ("F[all@23,1]+clamp@4_best", "force all L23")):
        A = {r["name"]: r["arms"][arm]["delta_margin"] for r in oos}
        B = {r["name"]: r["arms"]["clamp@4_best"]["delta_margin"] for r in oos}
        d = paired_ci_named(A, B)
        fa = rate([r["arms"][arm] for r in oos]); fb = rate([r["arms"]["clamp@4_best"] for r in oos])
        ax.errorbar(d["mean_diff"], y, xerr=[[d["mean_diff"] - d["lo95"]], [d["hi95"] - d["mean_diff"]]], fmt="D", color=S.VIOLET, ms=3.8, elinewidth=0.9, capsize=0)
        ax.text(4.3, y, f"{fb:.2f} → {fa:.2f}", va="center", fontsize=5.8, color=S.INK2)
        ticks.append(y); labels.append(f"new · {lab} ({len(oos)})")
        rows_out.append(("OOS", lab, d, fb, fa))
        y -= 1
    ax.text(4.3, 0.85, "flip rate", va="bottom", fontsize=5.8, color=S.INK2, fontweight="semibold")
    ax.axvline(0, color=S.LIGHTGRAY, lw=0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.5, 6.0)
    ax.set_xticks([0, 2, 4])
    ax.set_ylim(y + 0.4, 1.3)
    ax.set_xlabel("forced head + clamp − clamp,\npaired Δmargin (nat), 95% CI\n“new” = out-of-sample items, 9B")
    S.title(ax, "no donor: force the head onto p,\nthen clamp (“flips”: band-clamp\nflippable items)")
    return rows_out


def main() -> None:
    S.apply()
    groups = _groups()
    fig = plt.figure(figsize=(S.FULL, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.7, 1.05], wspace=0.62)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))
    meds = _routing(ax_a, groups)
    vals, a0, a1 = _transplants(ax_b, groups)
    rec = _recipe(ax_c)
    for ax, letter, dx, dy in ((ax_a, "a", -0.45, 1.2), (ax_b, "b", -0.1, 1.2), (ax_c, "c", -0.55, 1.2)):
        S.panel_label(ax, letter, dx=dx, dy=dy)
    S.headline(fig, "Whether the model uses the edit depends on two separable conditions: content in the plane at the edited position,\nand a transport head that reads that position", y=1.1)
    S.save(fig, "Fig7_content_routing")
    print(f"  h8 medians by group: {np.round(meds, 2).tolist()}")
    for lab, v in vals.items():
        print(f"  arms {lab.replace(chr(10), ' ')}: {np.round(v, 2).tolist()}")
    print(f"  failing 27 keys-only attention {a0:.2f} -> {a1:.2f}")
    for name, sub, d, fb, fa in rec:
        print(f"  recipe {name} {sub}: {d['mean_diff']:+.2f} [{d['lo95']:+.2f}, {d['hi95']:+.2f}] flips {fb:.2f}->{fa:.2f} n={d['n']}")


if __name__ == "__main__":
    main()
