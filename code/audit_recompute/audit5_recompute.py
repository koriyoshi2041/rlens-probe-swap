#!/usr/bin/env python3
"""Recompute audit 5: log blocks 75-79 (+ block-79 4B paragraph, block-69 bootstrap supplement), rows B4, C15,
C18-C21 and section H of the numbers list.  Independent of the project's analysis scripts (rlens is NOT imported);
reads only raw jsonl/json under 正式研究同步/results/ and code/data/new_items_nongeo_v2.json.
Run from code/:  python3 audit_recompute/audit5_recompute.py [--seed N]
Prints the per-claim tables (markdown) and the mismatch summary.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[3] / "正式研究同步" / "results"
V2 = pathlib.Path(__file__).resolve().parents[1] / "data" / "new_items_nongeo_v2.json"
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
N_BOOT = 10000
CI_TOL = 0.15


# ----------------------------------------------------------------------------- helpers
def jl(rel):
    return [json.loads(l) for l in (RES / rel).read_text(encoding="utf-8").splitlines() if l.strip()]


def med(xs):
    xs = list(xs)
    return float(np.median(xs)) if xs else float("nan")


def rate(xs):
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def auc(score, label):
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def union_find_clusters(indices, pairs):
    parent = {i: i for i in indices}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in indices}


def fact_pairs(items):
    """items: index -> dict(intermediate, swap_to).  Same unordered entity pair => same fact cluster."""
    key_to = {}
    for idx, it in items.items():
        key = tuple(sorted((it["intermediate"].strip().lower(), it["swap_to"].strip().lower())))
        key_to.setdefault(key, []).append(int(idx))
    pairs = []
    for members in key_to.values():
        members = sorted(members)
        pairs += [[members[0], o] for o in members[1:]]
    return pairs


def boot_paired(a, b, clusters, seed=SEED, n_boot=N_BOOT):
    """Cluster bootstrap of mean(a-b) over shared items; returns dict(n, mean, lo, hi, win, median)."""
    shared = sorted(set(a) & set(b))
    diff = {i: a[i] - b[i] for i in shared}
    groups = {}
    for i in shared:
        groups.setdefault(clusters.get(i, i), []).append(diff[i])
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for t in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        draws[t] = np.mean([v for k in picked for v in groups[keys[k]]])
    return {"n": len(shared), "mean": float(np.mean(list(diff.values()))), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "win": float(np.mean([d > 0 for d in diff.values()])),
            "median": med(diff.values()), "n_clusters": len(keys)}


# ----------------------------------------------------------------------------- check registry
CHECKS = []
CURRENT = {"block": ""}


def section(name):
    CURRENT["block"] = name


def _dp(s):
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def _num(s):
    return float(s.replace("−", "-").replace("+", "").replace("%", "").strip())


def chk(cid, claim, logged, recomputed, kinds=None, note=""):
    """logged: list of strings as printed in the log; recomputed: list of floats; kinds: 'n' (printed precision) or 'ci' (±0.15)."""
    logged = list(logged)
    recomputed = list(recomputed)
    kinds = kinds or ["n"] * len(logged)
    assert len(logged) == len(recomputed) == len(kinds), (cid, logged, recomputed)
    ok = []
    for ls, rv, k in zip(logged, recomputed, kinds):
        lv = _num(ls)
        if ls.endswith("%"):
            lv, dp = lv / 100.0, _dp(ls) + 2
        else:
            dp = _dp(ls)
        tol = CI_TOL if k == "ci" else 0.5 * 10 ** (-dp) + 1e-9
        ok.append(abs(rv - lv) <= tol)
    CHECKS.append({"block": CURRENT["block"], "id": cid, "claim": claim, "logged": logged, "rec": recomputed,
                   "ok": ok, "note": note})


def fmt(v, dp=3):
    return f"{v:+.{dp}f}" if isinstance(v, float) else str(v)


# ----------------------------------------------------------------------------- data + groups
audit = {r["index"]: r for r in json.load(open(RES / "block01" / "q01_data_audit_rows.json"))}
sup = jl("block12_suppression/suppression.jsonl")
ZERO = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
e62 = jl("block62_energy_of_arms/energy_of_arms.jsonl")
RESC4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
b69 = jl("block69_position_choice/position_choice.jsonl")
ALL59 = sorted(r["index"] for r in b69)
FLIP24 = [i for i in ALL59 if i not in ZERO]
ZERO35 = [i for i in ALL59 if i in ZERO]
FAIL27 = [i for i in ZERO35 if i not in RESC4]
CL9B = union_find_clusters(ALL59, fact_pairs({i: audit[i] for i in ALL59}))

v2_items = json.load(open(V2))["items"]
V2_INDEX = {it["name"]: i for i, it in enumerate(v2_items)}
b76 = jl("block76_oos_recipe/oos_recipe.jsonl")
for r in b76:
    r["index"] = V2_INDEX[r["name"]]
IDX76 = sorted(r["index"] for r in b76)
CL_V2 = union_find_clusters(IDX76, fact_pairs({i: v2_items[i] for i in IDX76}))

b75 = jl("block75_donor_free_recipe/donor_free.jsonl")
b77 = jl("block77_growth_pca/growth_pca.jsonl")
b78 = jl("block78_early_band_routing/early_routing.jsonl")
b79 = jl("block79_clean_head_ablation/head_ablation.jsonl")
b79b = jl("block79_4b_clean_head_ablation/head_ablation.jsonl")
b66 = jl("block66_key_value/key_value.jsonl")
b73 = jl("block73_routing_all_layers/routing_all_layers.jsonl")


def arm_vals(rows, arm, field, idx=None):
    return {r["index"]: r["arms"][arm][field] for r in rows if arm in r["arms"] and (idx is None or r["index"] in idx)}


def flip(rows, arm, idx=None):
    return rate(arm_vals(rows, arm, "top1_is_swap", idx).values())


def ans(rows, arm, idx=None):
    return rate(arm_vals(rows, arm, "top1_is_answer", idx).values())


def dm(rows, arm, idx=None):
    return med(arm_vals(rows, arm, "delta_margin", idx).values())


def paired(rows, a, b, idx, clusters, seed=SEED):
    return boot_paired(arm_vals(rows, a, "delta_margin", idx), arm_vals(rows, b, "delta_margin", idx), clusters, seed=seed)


def ci_note(d):
    return f"mean {d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] win {d['win']:.2f} n={d['n']} clusters={d['n_clusters']}"


# ============================================================================= block 69 supplement (bootstrap)
section("S. 第 69 块补充（事实级聚类 bootstrap）+ C21 restatement")
chk("S.0", "groups: all / zero-flip / clamp2d@4-fails / flippable; p_h8≠p_best count", ["59", "35", "27", "24", "27"],
    [len(ALL59), len(ZERO35), len(FAIL27), len(FLIP24), sum(1 for r in b69 if r["p_h8"] != r["p_best"])])
DIFF27 = [r["index"] for r in b69 if r["p_h8"] != r["p_best"]]
d = paired(b69, "orth@0.25_h8", "orth@0.25_best", DIFF27, CL9B)
chk("S.1", "orth ×0.25 @p_h8 − @p_best, p_h8≠p_best 27: mean [CI]; flips", ["−0.40", "−1.28", "+0.46", "0.59", "0.41"],
    [d["mean"], d["lo"], d["hi"], flip(b69, "orth@0.25_h8", DIFF27), flip(b69, "orth@0.25_best", DIFF27)], ["n", "ci", "ci", "n", "n"], ci_note(d))
d = paired(b69, "clamp@4_h8", "clamp@4_best", DIFF27, CL9B)
chk("S.2", "clamp ×4 @p_h8 − @p_best, 27: mean [CI]; flips", ["+1.77", "+0.11", "+3.75", "0.33", "0.15"],
    [d["mean"], d["lo"], d["hi"], flip(b69, "clamp@4_h8", DIFF27), flip(b69, "clamp@4_best", DIFF27)], ["n", "ci", "ci", "n", "n"], ci_note(d))
d = paired(b69, "clamp@1_h8", "clamp@1_best", DIFF27, CL9B)
chk("S.3", "clamp ×1 @p_h8 − @p_best, 27", ["+1.09", "+0.12", "+2.27"], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
d = paired(b69, "clamp@4_all", "clamp@4_best", DIFF27, CL9B)
chk("S.4", "clamp ×4 @all − @p_best, 27: mean [CI]; flips", ["+6.27", "+3.96", "+8.77", "0.48", "0.15"],
    [d["mean"], d["lo"], d["hi"], flip(b69, "clamp@4_all", DIFF27), flip(b69, "clamp@4_best", DIFF27)], ["n", "ci", "ci", "n", "n"], ci_note(d))
d1 = paired(b69, "orth@0.25_h8", "orth@0.25_best", ALL59, CL9B)
d2 = paired(b69, "clamp@4_h8", "clamp@4_best", ALL59, CL9B)
d3 = paired(b69, "clamp@4_all", "clamp@4_best", ALL59, CL9B)
chk("S.5", "all 59: orth / clamp ×4 / all-positions, mean [CI]", ["−0.18", "−0.60", "+0.22", "+0.81", "+0.03", "+1.71", "+4.40", "+3.10", "+5.85"],
    [d1["mean"], d1["lo"], d1["hi"], d2["mean"], d2["lo"], d2["hi"], d3["mean"], d3["lo"], d3["hi"]], ["n", "ci", "ci"] * 3,
    "; ".join(ci_note(x) for x in (d1, d2, d3)))
d = paired(b66, "K@19,23+clamp2d@4", "clamp2d@4", FLIP24, CL9B)
chk("S.6", "block66 donor key + clamp − clamp, flippable 24: mean [CI], win", ["+2.33", "+1.31", "+3.58", "0.96"], [d["mean"], d["lo"], d["hi"], d["win"]], ["n", "ci", "ci", "n"], ci_note(d))
d = paired(b73, "PAT_h8@23+clamp", "clamp2d@4", FLIP24, CL9B)
chk("S.7", "block73 single-head pattern + clamp − clamp, flippable 24: mean [CI], win (24/24)", ["+1.35", "+0.68", "+2.24", "1.00"], [d["mean"], d["lo"], d["hi"], d["win"]], ["n", "ci", "ci", "n"], ci_note(d))
d = paired(b69, "K<-foil+clamp@4_best", "clamp@4_best", FLIP24, CL9B)
chk("S.8", "block69 foil key + clamp − clamp, flippable 24: mean [CI]", ["+0.84", "−0.14", "+1.96"], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
d = paired(b66, "K@19,23+clamp2d@4", "clamp2d@4", ALL59, CL9B)
chk("S.9", "block66 donor key + clamp − clamp, all 59", ["+1.46", "+0.91", "+2.10"], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
n_incons = sum(1 for r in b69 if r["p_h8"] != r["p_best"])
n_incons_fail = sum(1 for r in b69 if r["index"] in FAIL27 and r["p_h8"] != r["p_best"])
chk("S.10", "C21: p_h8≠p_best 46% (fail27 67%); clamp@4_all flip all 0.61 / fail27 0.30", ["46%", "67%", "0.61", "0.30"],
    [n_incons / 59, n_incons_fail / 27, flip(b69, "clamp@4_all", ALL59), flip(b69, "clamp@4_all", FAIL27)], note=f"{n_incons}/59, {n_incons_fail}/27")

# ============================================================================= block 75
section("75. 无捐赠者的编辑配方 (block75_donor_free_recipe/donor_free.jsonl, 59 items × 23 arms)")
chk("75.0", "n items; every item has all 23 arms", ["59", "23"], [len(b75), min(len(r["arms"]) for r in b75)])
F = FLIP24
rows_tab = [
    ("75.1", "clamp@4_best", "钳位 ×4 @p_best", "0.79", "+9.7"),
    ("75.2", "F[h8@23,1]+clamp@4_best", "强制 h8 one-hot + 钳位 ×4 @p_best", "0.96", "+11.2"),
    ("75.3", "F[h8@23,0.8]+clamp@4_best", "强制 h8 (0.8) + 钳位 @p_best", "0.88", "+10.8"),
    ("75.4", "F[h8h9h0@23,1]+clamp@4_best", "强制 h8+h9+h0 + 钳位 @p_best", "0.96", "+13.2"),
    ("75.5", "F[all@23,1]+clamp@4_best", "强制 L23 全部头 + 钳位 @p_best", "0.96", "+13.0"),
    ("75.6", "K<-donor+clamp@4_best", "捐赠者 key + 钳位 @p_best", "0.96", "+12.1"),
    ("75.7", "clamp@4_h8", "钳位 ×4 @p_h8", "0.83", "+10.9"),
    ("75.8", "F[h8@23,1]+clamp@4_h8", "强制 h8 + 钳位 @p_h8", "0.83", "+11.2"),
    ("75.9", "F[all@23,1]+clamp@4_h8", "强制 L23 全部头 + 钳位 @p_h8", "0.88", "+12.8"),
]
for cid, arm, label, lf, ld in rows_tab:
    chk(cid, f"flippable 24: {label} — flip, ΔM median", [lf, ld], [flip(b75, arm, F), dm(b75, arm, F)], note=arm)
d = paired(b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", F, CL9B)
chk("75.10", "flippable 24: F[h8,1]+clamp@4_best − clamp@4_best: mean [CI], win", ["+1.66", "+0.88", "+2.71", "0.96"], [d["mean"], d["lo"], d["hi"], d["win"]], ["n", "ci", "ci", "n"], ci_note(d))
d = paired(b75, "F[h8@23,1]+clamp@4_h8", "clamp@4_h8", F, CL9B)
chk("75.11", "flippable 24: F[h8,1]+clamp@4_h8 − clamp@4_h8: mean [CI]", ["+1.15", "+0.56", "+1.99"], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
d = paired(b75, "F[all@23,1]+clamp@4_h8", "clamp@4_h8", F, CL9B)
chk("75.12", "flippable 24: F[all@23,1]+clamp@4_h8 − clamp@4_h8: mean [CI]", ["+1.84", "+1.00", "+3.06"], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
d = paired(b75, "K<-donor+clamp@4_best", "clamp@4_best", F, CL9B)
dd = {i: abs(b75_r["arms"]["K<-donor+clamp@4_best"]["delta_margin"] - b66_r["arms"]["K@19,23+clamp2d@4"]["delta_margin"])
      for b75_r, b66_r in zip(sorted(b75, key=lambda r: r["index"]), sorted(b66, key=lambda r: r["index"])) for i in [b75_r["index"]]}
chk("75.13", "table row 捐赠者 key CI '+2.33 [+1.31, +3.58]' is the block-66 contrast (S.6); same contrast from block75's own arm", ["+2.33", "+1.31", "+3.58"],
    [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], f"block75 K<-donor+clamp@4_best − clamp@4_best: {ci_note(d)}; max |ΔM diff| block75 vs block66 per item = {max(dd.values()):.4f}")
chk("75.14", "flippable 24: 只强制 h8 不钳位 (F[h8@23,1]_best): flip, top1=answer, ΔM", ["0.00", "0.96", "−0.1"],
    [flip(b75, "F[h8@23,1]_best", F), ans(b75, "F[h8@23,1]_best", F), dm(b75, "F[h8@23,1]_best", F)],
    note=f"F[h8@23,1]_h8 flippable: flip {flip(b75, 'F[h8@23,1]_h8', F):.2f} ans {ans(b75, 'F[h8@23,1]_h8', F):.2f} ΔM {dm(b75, 'F[h8@23,1]_h8', F):+.2f}")
chk("75.15", "读法 (2): 强制单独 0.97 保留原答案 (all 59, F[h8@23,1]_best)", ["0.97"], [ans(b75, "F[h8@23,1]_best", ALL59)],
    note=f"all59 F[h8@23,1]_best flip {flip(b75, 'F[h8@23,1]_best', ALL59):.3f}; F[h8@23,1]_h8 ans {ans(b75, 'F[h8@23,1]_h8', ALL59):.3f}; F[all@23,1]_best ans {ans(b75, 'F[all@23,1]_best', ALL59):.3f}")
d = paired(b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", ALL59, CL9B)
d_alt = paired(b75, "F[h8@23,1]+clamp@4_h8", "clamp@4_h8", ALL59, CL9B)
chk("75.16", "all 59: F[h8,1]+clamp@4_best flip (clamp 0.46); paired +0.69 [+0.37, +1.09]", ["0.53", "0.46", "+0.69", "+0.37", "+1.09"],
    [flip(b75, "F[h8@23,1]+clamp@4_best", ALL59), flip(b75, "clamp@4_best", ALL59), d["mean"], d["lo"], d["hi"]], ["n", "n", "n", "ci", "ci"],
    f"@p_best all 59: {ci_note(d)} | the logged CI is the @p_h8 contrast F[h8@23,1]+clamp@4_h8 − clamp@4_h8, all 59: {ci_note(d_alt)}")
force_best = ["F[h8@23,1]+clamp@4_best", "F[h8@23,0.8]+clamp@4_best", "F[h8h9h0@23,1]+clamp@4_best", "F[all@23,1]+clamp@4_best", "F[all@19,23,1]+clamp@4_best"]
force_h8 = ["F[h8@23,1]+clamp@4_h8", "F[h8@23,0.8]+clamp@4_h8", "F[h8h9h0@23,1]+clamp@4_h8", "F[all@23,1]+clamp@4_h8", "F[all@19,23,1]+clamp@4_h8"]
zf = [flip(b75, a, ZERO35) for a in force_best]
chk("75.17", "zero-flip 35: clamp@4_best 0.23 → forcing arms @p_best 0.23–0.29 (min, max)", ["0.23", "0.23", "0.29"],
    [flip(b75, "clamp@4_best", ZERO35), min(zf), max(zf)], note="; ".join(f"{a} {v:.3f}" for a, v in zip(force_best, zf)) + f"; K<-donor+clamp@4_best {flip(b75, 'K<-donor+clamp@4_best', ZERO35):.3f}")
ff = {a: flip(b75, a, FAIL27) for a in force_best}
fh = {a: flip(b75, a, FAIL27) for a in force_h8}
chk("75.18", "fail 27: max flip over forcing arms @p_best 0.11 (h8h9h0 3/27); @p_h8 max 0.19", ["0.11", "3", "0.19"],
    [max(ff.values()), ff["F[h8h9h0@23,1]+clamp@4_best"] * 27, max(fh.values())],
    note="best: " + "; ".join(f"{a} {v * 27:.0f}/27" for a, v in ff.items()) + " | h8: " + "; ".join(f"{a} {v * 27:.0f}/27" for a, v in fh.items()) + f"; K<-donor+clamp@4_best {flip(b75, 'K<-donor+clamp@4_best', FAIL27) * 27:.0f}/27; clamp@4_h8 {flip(b75, 'clamp@4_h8', FAIL27) * 27:.0f}/27")
chk("75.19", "all 59: clamp@4_all + F h8 flip 0.61 (without 0.61); ΔM +9.8 vs +9.4", ["0.61", "0.61", "+9.8", "+9.4"],
    [flip(b75, "F[h8@23,1]_h8+clamp@4_all", ALL59), flip(b75, "clamp@4_all", ALL59), dm(b75, "F[h8@23,1]_h8+clamp@4_all", ALL59), dm(b75, "clamp@4_all", ALL59)])
chk("75.20", "读法 (1) restatement: 0.96, +1.7 vs +2.3 (rounded CI means)", ["0.96", "+1.7", "+2.3"],
    [flip(b75, "F[h8@23,1]+clamp@4_best", F), paired(b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", F, CL9B)["mean"], paired(b66, "K@19,23+clamp2d@4", "clamp2d@4", F, CL9B)["mean"]])

# ============================================================================= block 76
section("76. 样本外配方与预测量 (block76_oos_recipe/oos_recipe.jsonl, v2 non-geographic, 28 items × 17 arms)")
chk("76.0", "n=28; p_h8==p_best 39%; block-55 band clamp flip 0.29; fact clusters 21", ["28", "39%", "0.29", "21"],
    [len(b76), rate(r["p_h8"] == r["p_best"] for r in b76), rate(r["band_clamp_flip_block55"] for r in b76), len(set(CL_V2.values()))],
    note=f"{sum(r['p_h8'] == r['p_best'] for r in b76)}/28 same site; clamp@1_all flip {flip(b76, 'clamp@1_all'):.3f} = block55 field")
tab76 = [
    ("76.1", "clamp@1_all", "钳位 ×1 @全部位置", "0.29", "+2.8"),
    ("76.2", "clamp@4_all", "钳位 ×4 @全部位置", "0.71", "+9.8"),
    ("76.3", "clamp@4_best", "钳位 ×4 @p_best", "0.32", "+1.7"),
    ("76.4", "clamp@4_h8", "钳位 ×4 @p_h8", "0.36", "+3.8"),
    ("76.5", "F[h8@23,1]+clamp@4_best", "强制 h8 + 钳位 ×4 @p_best", "0.43", "+3.3"),
    ("76.6", "F[all@23,1]+clamp@4_best", "强制 L23 全部头 + 钳位 ×4 @p_best", "0.50", "+4.3"),
    ("76.7", "K<-donor+clamp@4_best", "捐赠者 key + 钳位 ×4 @p_best", "0.46", "+3.9"),
    ("76.8", "F[h8@23,1]+clamp@4_h8", "强制 h8 + 钳位 @p_h8", "0.43", "+5.1"),
    ("76.9", "F[all@23,1]+clamp@4_h8", "全部头 + 钳位 @p_h8", "0.57", "+6.0"),
    ("76.10", "orth@0.25_best", "正交补 ×0.25 @p_best", "0.39", "+6.7"),
    ("76.11", "orth@0.25_h8", "正交补 ×0.25 @p_h8", "0.64", "+6.0"),
    ("76.12", "full@0.25_best", "整向量 ×0.25 @p_best", "0.50", "+8.0"),
    ("76.13", "full@0.25_h8", "整向量 ×0.25 @p_h8", "0.79", "+7.4"),
]
for cid, arm, label, lf, ld in tab76:
    chk(cid, f"{label} — flip, ΔM median", [lf, ld], [flip(b76, arm), dm(b76, arm)], note=arm)
a4 = ans(b76, "clamp@4_all")
n_nei = sum(1 for r in b76 if not r["arms"]["clamp@4_all"]["top1_is_swap"] and not r["arms"]["clamp@4_all"]["top1_is_answer"])
chk("76.14", "clamp@4_all: top1 = original answer 0.07, neither 0.22 (读法: 22% 改坏)", ["0.07", "0.22"], [a4, 1 - flip(b76, "clamp@4_all") - a4],
    note=f"neither = {n_nei}/28 = {n_nei / 28:.4f}; 1 − 0.71 − 0.07 = 0.22 only from the rounded rates (20/28 swap, 2/28 answer)")
ci76 = [
    ("76.15", "clamp@4_all", "clamp@1_all", "钳位 ×4 @all − ×1 @all", ["+5.76", "+4.42", "+7.06", "0.93"]),
    ("76.16", "clamp@4_h8", "clamp@4_best", "钳位 ×4 @p_h8 − @p_best", ["+0.67", "−0.13", "+1.60", None]),
    ("76.17", "F[h8@23,1]+clamp@4_best", "clamp@4_best", "强制 h8 + 钳位 − 钳位 @p_best", ["+0.72", "+0.38", "+1.09", "0.82"]),
    ("76.18", "F[all@23,1]+clamp@4_best", "clamp@4_best", "强制 L23 全部头 + 钳位 − 钳位 @p_best", ["+1.38", "+0.94", "+1.82", "0.93"]),
    ("76.19", "K<-donor+clamp@4_best", "clamp@4_best", "捐赠者 key + 钳位 − 钳位 @p_best", ["+1.03", "+0.50", "+1.57", None]),
    ("76.20", "F[h8@23,1]+clamp@4_h8", "clamp@4_h8", "强制 h8 + 钳位 − 钳位 @p_h8", ["+0.65", "+0.42", "+0.91", None]),
    ("76.21", "F[all@23,1]+clamp@4_h8", "clamp@4_h8", "全部头 + 钳位 − 钳位 @p_h8", ["+1.15", "+0.80", "+1.50", None]),
    ("76.22", "orth@0.25_h8", "orth@0.25_best", "正交补 位置差 (h8 − best)", ["−0.22", "−1.17", "+0.81", None]),
    ("76.23", "full@0.25_h8", "full@0.25_best", "整向量 位置差 (h8 − best)", ["−0.06", "−0.96", "+0.94", None]),
]
for cid, a, b, label, lg in ci76:
    d = paired(b76, a, b, IDX76, CL_V2)
    if lg[3] is None:
        chk(cid, f"{label}: mean [CI]", lg[:3], [d["mean"], d["lo"], d["hi"]], ["n", "ci", "ci"], ci_note(d))
    else:
        chk(cid, f"{label}: mean [CI], win", lg, [d["mean"], d["lo"], d["hi"], d["win"]], ["n", "ci", "ci", "n"], ci_note(d))


def predictors(target, site):
    lab = [r["arms"][target]["top1_is_swap"] for r in b76]
    y = [r["arms"][target]["delta_margin"] for r in b76]
    h8 = [r["h8_mass"][site] for r in b76]
    gap = [r["gap"][site] for r in b76]
    cos = [r["cos"] for r in b76]
    prod = [g * h for g, h in zip(gap, h8)]
    return {"pos": sum(lab), "h8": auc(h8, lab), "rho_h8": spearman(h8, y), "gap": auc(gap, lab), "prod": auc(prod, lab), "cos": auc(cos, lab)}


p = predictors("clamp@4_best", "best")
chk("76.24", "predict clamp@4_best flip (9/28): h8 AUC, ρ(h8, ΔM), gap, gap×h8, cos", ["9", "0.88", "+0.74", "0.80", "0.90", "0.54"],
    [p["pos"], p["h8"], p["rho_h8"], p["gap"], p["prod"], p["cos"]])
p = predictors("clamp@4_h8", "h8")
chk("76.25", "predict clamp@4_h8 flip (10/28): h8, gap, gap×h8, cos", ["10", "0.92", "0.69", "0.90", "0.45"], [p["pos"], p["h8"], p["gap"], p["prod"], p["cos"]])
p_h = predictors("clamp@1_all", "h8")
p_b = predictors("clamp@1_all", "best")
chk("76.26", "predict clamp@1_all flip (8/28): gap 0.74, h8 0.61, cos 0.51 (features at the p_h8 site)", ["8", "0.74", "0.61", "0.51"],
    [p_h["pos"], p_h["gap"], p_h["h8"], p_h["cos"]], note=f"p_best-site features: gap {p_b['gap']:.2f}, h8 {p_b['h8']:.2f}, gap×h8 {p_b['prod']:.2f}; p_h8-site gap×h8 {p_h['prod']:.2f}")
chk("76.27", "读法 restatements: 0.32 → 0.50; 0.29 → 0.71; 贴入 0.64 对 0.39", ["0.32", "0.50", "0.29", "0.71", "0.64", "0.39"],
    [flip(b76, "clamp@4_best"), flip(b76, "F[all@23,1]+clamp@4_best"), flip(b76, "clamp@1_all"), flip(b76, "clamp@4_all"), flip(b76, "orth@0.25_h8"), flip(b76, "orth@0.25_best")])

# ============================================================================= block 77
section("77. 残差 PCA 必要性 + 生成定位 (block77_growth_pca/growth_pca.jsonl, 59 items)")
chk("77.0", "n items", ["59"], [len(b77)])
nec = [
    ("77.1", "full", "无（整向量）", "0.66", "0.51", "+8.8", "4.49"),
    ("77.2", "full-PCA64", "残差 PCA 前 64", "0.53", "0.43", "+4.6", "0.99"),
    ("77.3", "full-PCA256", "残差 PCA 前 256", "0.19", "0.17", "+2.3", "0.12"),
    ("77.4", "full-PCA1024", "残差 PCA 前 1024", "0.08", "0.09", "+1.1", "0.03"),
    ("77.5", "full-R256", "R 前 256", "0.31", "0.34", "+4.4", "0.61"),
    ("77.6", "full-(R256+PCA256)", "R 前 256 ∪ PCA 前 256", "0.05", "0.03", "+1.0", "0.02"),
    ("77.7", "full-rand256", "随机 256", "0.64", "0.51", "+7.6", "2.42"),
]
for cid, arm, label, lf, lz, ld, lk in nec:
    a = [r["necessity"][arm] for r in b77]
    z = [r["necessity"][arm] for r in b77 if r["index"] in ZERO]
    chk(cid, f"减去 {label}: flip / zero-flip flip / ΔM / KL", [lf, lz, ld, lk],
        [rate(q["top1_is_swap"] for q in a), rate(q["top1_is_swap"] for q in z), med(q["delta_margin"] for q in a), med(q["kl"] for q in a)], note=arm)
growth = [
    ("77.8", "J+R256", "J + R 前 256", ["0.07", "0.33", "4.7", "+0.03", "+0.22", "+0.067", "+0.043", "+0.054"]),
    ("77.9", "J+PCA256", "J + PCA 前 256", ["0.13", "0.59", "4.5", "+0.06", "+0.40", "+0.104", "+0.085", "+0.093"]),
    ("77.10", "full", "整向量", ["0.18", "0.75", "4.2", "+0.06", "+0.51", "+0.131", "+0.101", "+0.115"]),
    ("77.11", "J+rand256", "J + 随机 256", ["0.02", "0.05", "2.7", "+0.01", "+0.03", "+0.010", "+0.006", "+0.004"]),
]
for cid, sub, label, lg in growth:
    g = [r["growth"][sub] for r in b77]
    l8 = med(q["realised_L8_on_d20"] for q in g)
    l20 = med(q["layers"][-1]["realised_d20"] for q in g)
    assert all(q["layers"][-1]["layer"] == 20 for q in g)
    ratio_items = med(q["layers"][-1]["realised_d20"] / q["realised_L8_on_d20"] for q in g)
    ratio_meds = l20 / l8
    sa = med(sum(l["attn_on_d20"] for l in q["layers"]) for q in g)
    sm = med(sum(l["mlp_on_d20"] for l in q["layers"]) for q in g)
    m18, m19, m20 = (med(next(l for l in q["layers"] if l["layer"] == L)["mlp_on_d20"] for q in g) for L in (18, 19, 20))
    ok_ratio = ratio_items if abs(ratio_items - _num(lg[2])) <= abs(ratio_meds - _num(lg[2])) else ratio_meds
    ratio_printed = round(l20, 3) / round(l8, 2)
    chk(cid, f"{label}: L8 on d20, L20 on d20, 增长倍数, attn ΣL9–20, MLP ΣL9–20, MLP L18/L19/L20 (all medians over items)", lg,
        [l8, l20, ok_ratio, sa, sm, m18, m19, m20],
        note=f"{sub}: 增长倍数 as median of per-item ratios {ratio_items:.2f}, as ratio of unrounded medians {ratio_meds:.2f}, as ratio of the printed medians {round(l20, 3):.3f}/{round(l8, 2):.2f} = {ratio_printed:.2f}; attn Σ mean {float(np.mean([sum(l['attn_on_d20'] for l in q['layers']) for q in g])):.4f}; flip {rate(q['top1_is_swap'] for q in g):.2f}")
chk("77.12", "读法 restatements: 0.66→0.19, →0.31, →0.05, 随机 0.64; lens MLP 0.22 vs attn 0.03", ["0.66", "0.19", "0.31", "0.05", "0.64", "0.22", "0.03"],
    [rate(r["necessity"][a]["top1_is_swap"] for r in b77) for a in ("full", "full-PCA256", "full-R256", "full-(R256+PCA256)", "full-rand256")]
    + [med(sum(l["mlp_on_d20"] for l in r["growth"]["J+R256"]["layers"]) for r in b77), med(sum(l["attn_on_d20"] for l in r["growth"]["J+R256"]["layers"]) for r in b77)])

# ============================================================================= block 78
section("78. 早层 band 路由 J vs R (block78_early_band_routing/early_routing.jsonl, 59 items)")
chk("78.0", "n items", ["59"], [len(b78)])


def a78(arm, f, idx=None):
    return med(arm_vals(b78, arm, f, idx).values())


chk("78.1", "clean: h8→p_best, h8→p_h8, L23 全头→p_best (medians)", ["0.28", "0.42", "1.38"], [a78("clean", "h8_best"), a78("clean", "h8_h8pos"), a78("clean", "L23_sum_best")])
for cid, arm, lg in [("78.2", "J@1", ["0.14", "+0.75", "0.32", "0.42", "1.59"]), ("78.3", "R@1", ["0.25", "+1.38", "0.33", "0.44", "1.65"]),
                     ("78.4", "J@2", ["0.32", "+2.81", "0.32", "0.44", "1.51"]), ("78.5", "R@2", ["0.42", "+4.62", "0.31", "0.44", "1.52"])]:
    chk(cid, f"{arm}: flip, ΔM, h8→p_best, h8→p_h8, L23 全头→p_best", lg,
        [flip(b78, arm), a78(arm, "delta_margin"), a78(arm, "h8_best"), a78(arm, "h8_h8pos"), a78(arm, "L23_sum_best")])
chk("78.6", "KV@19,23 ← J@1 / R@1 (p_best): flip, ΔM", ["0.02", "0.02", "+0.12", "+0.12"],
    [flip(b78, "KV<-J@1_best"), flip(b78, "KV<-R@1_best"), a78("KV<-J@1_best", "delta_margin"), a78("KV<-R@1_best", "delta_margin")])
chk("78.7", "KV ← J@2 / R@2 (p_best): flip, ΔM", ["0.05", "0.07", "+0.25", "+0.31"],
    [flip(b78, "KV<-J@2_best"), flip(b78, "KV<-R@2_best"), a78("KV<-J@2_best", "delta_margin"), a78("KV<-R@2_best", "delta_margin")])
chk("78.8", "Z@19,23 ← J@2 / R@2 (final position): flip, ΔM", ["0.05", "0.07", "+0.87", "+1.06"],
    [flip(b78, "Z<-J@2"), flip(b78, "Z<-R@2"), a78("Z<-J@2", "delta_margin"), a78("Z<-R@2", "delta_margin")])
pr = {}
for sc in ("1", "2"):
    dmd = [r["arms"][f"R@{sc}"]["delta_margin"] - r["arms"][f"J@{sc}"]["delta_margin"] for r in b78]
    dh = [r["arms"][f"R@{sc}"]["h8_best"] - r["arms"][f"J@{sc}"]["h8_best"] for r in b78]
    dkv = [r["arms"][f"KV<-R@{sc}_best"]["delta_margin"] - r["arms"][f"KV<-J@{sc}_best"]["delta_margin"] for r in b78]
    pr[sc] = (med(dmd), rate(x > 0 for x in dmd), med(dh), med(dkv), float(np.mean(dmd)))
chk("78.9", "paired R − J: ΔM median @1 (R>J share), @2 (share); h8→p_best diff @1/@2; KV-transplant ΔM diff @1/@2",
    ["+0.38", "78%", "+0.44", "71%", "−0.001", "−0.002", "0.00", "0.00"],
    [pr["1"][0], pr["1"][1], pr["2"][0], pr["2"][1], pr["1"][2], pr["2"][2], pr["1"][3], pr["2"][3]], note=f"mean ΔM R−J @1 {pr['1'][4]:+.2f}, @2 {pr['2'][4]:+.2f}")
kv_all = [flip(b78, a) for a in ("KV<-J@1_best", "KV<-R@1_best", "KV<-J@2_best", "KV<-R@2_best")]
z_all = [flip(b78, a) for a in ("Z<-J@2", "Z<-R@2")]
chk("78.10", "读法/B4: 0.28 → 0.32 (h8→p_best clean → J@1); KV transplant flips 0.02–0.07; Z 0.05–0.07", ["0.28", "0.32", "0.02", "0.07", "0.05", "0.07"],
    [a78("clean", "h8_best"), a78("J@1", "h8_best"), min(kv_all), max(kv_all), min(z_all), max(z_all)],
    note=f"KV p_h8-site flips: {[round(flip(b78, a), 3) for a in ('KV<-J@1_h8', 'KV<-R@1_h8', 'KV<-J@2_h8', 'KV<-R@2_h8')]}; Z@1: {[round(flip(b78, a), 3) for a in ('Z<-J@1', 'Z<-R@1')]}")

# ============================================================================= block 79
section("79. clean 头消融 (block79_clean_head_ablation, 59 items; block79_4b_clean_head_ablation, 47 items)")


def abl(rows, arm, kind):
    s = [r for r in rows if r["arm"] == arm and r["prompt_kind"] == kind and r["clean_correct"]]
    return rate(r["still_correct"] for r in s), med(r["delta_logp_answer"] for r in s), len(s)


def clean_acc(rows, kind):
    s = [r for r in rows if r["arm"] == "h8" and r["prompt_kind"] == kind]
    return rate(r["clean_correct"] for r in s)


chk("79.0", "9B: n items; clean two-hop acc 1.00; clean single-hop acc 0.97", ["59", "1.00", "0.97"],
    [len({r["index"] for r in b79}), clean_acc(b79, "two_hop"), clean_acc(b79, "single_hop")],
    note=f"clean_correct identical across arms: {all(len({r['clean_correct'] for r in b79 if r['index'] == i and r['prompt_kind'] == k}) == 1 for i in {r['index'] for r in b79} for k in ('two_hop', 'single_hop'))}")
t79 = [("79.1", "h8", ["0.97", "−0.01", "1.00"]), ("79.2", "h9", ["1.00", "0.00", "1.00"]), ("79.3", "h0", ["1.00", "0.00", "1.00"]),
       ("79.4", "h8h9h0", ["0.95", "−0.02", "1.00"]), ("79.5", "rand3@23", ["1.00", "0.00", "1.00"]), ("79.6", "all@23", ["0.93", "−0.03", "1.00"]),
       ("79.7", "all@19", ["0.95", "−0.05", "1.00"])]
for cid, arm, lg in t79:
    two, one = abl(b79, arm, "two_hop"), abl(b79, arm, "single_hop")
    chk(cid, f"9B L23/L19 {arm}: two-hop acc, Δlog p(answer) median, single-hop acc", lg, [two[0], two[1], one[0]],
        note=f"n two-hop {two[2]}, single-hop {one[2]}; single-hop Δlogp {one[1]:+.3f}")
chk("79.8", "读法: all listed accuracies ≥ 0.93 (min over the 7 arms, two-hop)", ["0.93"], [min(abl(b79, a, "two_hop")[0] for a in ("h8", "h9", "h0", "h8h9h0", "rand3@23", "all@23", "all@19"))])
chk("79.9", "4B: n items; two-hop acc h8 / h8h9h0 / all@23 / all@19 / rand3", ["47", "0.98", "0.98", "0.94", "0.91", "1.00"],
    [len({r["index"] for r in b79b})] + [abl(b79b, a, "two_hop")[0] for a in ("h8", "h8h9h0", "all@23", "all@19", "rand3@23")],
    note=f"4B h9 {abl(b79b, 'h9', 'two_hop')[0]:.3f}, h0 {abl(b79b, 'h0', 'two_hop')[0]:.3f}; clean two-hop {clean_acc(b79b, 'two_hop'):.3f}, single-hop {clean_acc(b79b, 'single_hop'):.3f}")
chk("79.10", "4B: single-hop acc all 1.00 (min over 7 arms)", ["1.00"], [min(abl(b79b, a, "single_hop")[0] for a in ("h8", "h9", "h0", "h8h9h0", "rand3@23", "all@23", "all@19"))])
b60b = [r for r in jl("block60_selfconsistent/selfconsistent.jsonl") if r["part"] == "b"]
two61, one61 = abl(b60b, "attn_17_24", "two_hop"), abl(b60b, "attn_17_24", "single_hop")
chk("79.11", "(第 61 块) L17–24 注意力全部 (block60 part b, attn_17_24): two-hop acc, Δlog p median, single-hop acc", ["0.54", "−0.96", "0.96"],
    [two61[0], two61[1], one61[0]], note=f"n two-hop {two61[2]}, single-hop {one61[2]}")

# ============================================================================= numbers list rows
section("Rows B4 / C15 / C18 / C19 / C20 / C21 and section H (numbers contributed by blocks 75–79)")
chk("B4", "block78: 0.28 → 0.32; paired h8 diff −0.001; KV/Z transplant 0.02–0.07; R−J KV diff 0.00; ΔM R−J +0.38 (78%)",
    ["0.28", "0.32", "−0.001", "0.02", "0.07", "0.00", "+0.38", "78%"],
    [a78("clean", "h8_best"), a78("J@1", "h8_best"), pr["1"][2], min(kv_all + z_all), max(kv_all + z_all), pr["1"][3], pr["1"][0], pr["1"][1]])
chk("C15", "block79: 9B two-hop 0.97 / 0.95 / 0.93 / 0.95 (single 1.00); 4B 0.98 / 0.98 / 0.94 / 0.91",
    ["0.97", "0.95", "0.93", "0.95", "1.00", "0.98", "0.98", "0.94", "0.91"],
    [abl(b79, a, "two_hop")[0] for a in ("h8", "h8h9h0", "all@23", "all@19")] + [max(abl(b79, a, "single_hop")[0] for a in ("h8", "h8h9h0", "all@23", "all@19"))]
    + [abl(b79b, a, "two_hop")[0] for a in ("h8", "h8h9h0", "all@23", "all@19")])
chk("C18", "block77: 0.66 → 0.19; → 0.31; → 0.05; → 0.64; lens MLP 0.22 vs attn 0.03; growth ≈4.5 (lens/PCA/full), random 2.7",
    ["0.66", "0.19", "0.31", "0.05", "0.64", "0.22", "0.03", "4.5", "2.7"],
    [rate(r["necessity"][a]["top1_is_swap"] for r in b77) for a in ("full", "full-PCA256", "full-R256", "full-(R256+PCA256)", "full-rand256")]
    + [med(sum(l["mlp_on_d20"] for l in r["growth"]["J+R256"]["layers"]) for r in b77), med(sum(l["attn_on_d20"] for l in r["growth"]["J+R256"]["layers"]) for r in b77),
       float(np.mean([med(r["growth"][s]["layers"][-1]["realised_d20"] / r["growth"][s]["realised_L8_on_d20"] for r in b77) for s in ("J+R256", "J+PCA256", "full")])),
       med(r["growth"]["J+rand256"]["layers"][-1]["realised_d20"] / r["growth"]["J+rand256"]["realised_L8_on_d20"] for r in b77)],
    note="'≈4.5' compared to the mean of the three per-item-median ratios; unrounded ratio-of-medians 4.49 / 4.59 / 4.28, random 2.54 (per-item median 2.87); the logged 4.7 / 4.5 / 4.2 / 2.7 are ratios of the printed medians (see 77.8–77.11)")
pb = predictors("clamp@4_best", "best")
ph = predictors("clamp@4_h8", "h8")
pa = predictors("clamp@1_all", "h8")
chk("C19", "block76 AUC: h8 0.88–0.92; gap×h8 0.90; cos 0.45–0.54; all-position gap 0.74 vs h8 0.61; block79 h8 ablation 0.97",
    ["0.88", "0.92", "0.90", "0.45", "0.54", "0.74", "0.61", "0.97"],
    [pb["h8"], ph["h8"], min(pb["prod"], ph["prod"]), ph["cos"], pb["cos"], pa["gap"], pa["h8"], abl(b79, "h8", "two_hop")[0]], note=f"gap×h8: best-site {pb['prod']:.3f}, h8-site {ph['prod']:.3f}")
d75 = paired(b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", F, CL9B)
d75a = paired(b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", ALL59, CL9B)
d66 = paired(b66, "K@19,23+clamp2d@4", "clamp2d@4", F, CL9B)
d73 = paired(b73, "PAT_h8@23+clamp", "clamp2d@4", F, CL9B)
d76a = paired(b76, "F[h8@23,1]+clamp@4_best", "clamp@4_best", IDX76, CL_V2)
d76b = paired(b76, "F[all@23,1]+clamp@4_best", "clamp@4_best", IDX76, CL_V2)
d76c = paired(b76, "K<-donor+clamp@4_best", "clamp@4_best", IDX76, CL_V2)
chk("C20a", "block75: 0.79 → 0.96, +1.66 [+0.88, +2.71] win 0.96; donor key 0.96, +2.33 [+1.31, +3.58]; pattern 0.88, +1.35 [+0.68, +2.24]",
    ["0.79", "0.96", "+1.66", "+0.88", "+2.71", "0.96", "0.96", "+2.33", "+1.31", "+3.58", "0.88", "+1.35", "+0.68", "+2.24"],
    [flip(b75, "clamp@4_best", F), flip(b75, "F[h8@23,1]+clamp@4_best", F), d75["mean"], d75["lo"], d75["hi"], d75["win"],
     flip(b66, "K@19,23+clamp2d@4", F), d66["mean"], d66["lo"], d66["hi"], flip(b73, "PAT_h8@23+clamp", F), d73["mean"], d73["lo"], d73["hi"]],
    ["n", "n", "n", "ci", "ci", "n", "n", "n", "ci", "ci", "n", "n", "ci", "ci"])
chk("C20b", "block75: force-only 0 flip / 0.97 answer; all 59 +0.69 [+0.37, +1.09]; fail 27 max 3/27; block76 +0.72 [+0.38, +1.09], +1.38 [+0.94, +1.82] (0.32 → 0.50), +1.03 [+0.50, +1.57]",
    ["0", "0.97", "+0.69", "+0.37", "+1.09", "3", "+0.72", "+0.38", "+1.09", "+1.38", "+0.94", "+1.82", "0.32", "0.50", "+1.03", "+0.50", "+1.57"],
    [flip(b75, "F[h8@23,1]_best", ALL59), ans(b75, "F[h8@23,1]_best", ALL59), d75a["mean"], d75a["lo"], d75a["hi"], max(ff.values()) * 27,
     d76a["mean"], d76a["lo"], d76a["hi"], d76b["mean"], d76b["lo"], d76b["hi"], flip(b76, "clamp@4_best"), flip(b76, "F[all@23,1]+clamp@4_best"), d76c["mean"], d76c["lo"], d76c["hi"]],
    ["n", "n", "n", "ci", "ci", "n", "n", "ci", "ci", "n", "ci", "ci", "n", "n", "n", "ci", "ci"],
    note=f"'0' flip compared at 2 dp; the all-59 '+0.69 [+0.37, +1.09]' is the @p_h8 contrast ({ci_note(d_alt)}), the @p_best contrast is {ci_note(d75a)}")
dS2 = paired(b69, "clamp@4_h8", "clamp@4_best", DIFF27, CL9B)
dS1 = paired(b69, "orth@0.25_h8", "orth@0.25_best", DIFF27, CL9B)
dS4 = paired(b69, "clamp@4_all", "clamp@4_best", DIFF27, CL9B)
dS8 = paired(b69, "K<-foil+clamp@4_best", "clamp@4_best", FLIP24, CL9B)
d76all = paired(b76, "clamp@4_all", "clamp@1_all", IDX76, CL_V2)
chk("C21", "46% (fail27 67%); +1.8 [+0.1, +3.8] (0.33 vs 0.15); 0.59 vs 0.41, −0.4 [−1.3, +0.5]; +6.3 [+4.0, +8.8] (0.61, 0.30); foil +0.8 [−0.1, +2.0]; donor +2.3 [+1.3, +3.6]; OOS 0.29 → 0.71 (+5.8 [+4.4, +7.1]; 22%); 0.64 vs 0.39",
    ["46%", "67%", "+1.8", "+0.1", "+3.8", "0.33", "0.15", "0.59", "0.41", "−0.4", "−1.3", "+0.5", "+6.3", "+4.0", "+8.8", "0.61", "0.30", "+0.8", "−0.1", "+2.0", "+2.3", "+1.3", "+3.6", "0.29", "0.71", "+5.8", "+4.4", "+7.1", "22%", "0.64", "0.39"],
    [n_incons / 59, n_incons_fail / 27, dS2["mean"], dS2["lo"], dS2["hi"], flip(b69, "clamp@4_h8", DIFF27), flip(b69, "clamp@4_best", DIFF27),
     flip(b69, "orth@0.25_h8", DIFF27), flip(b69, "orth@0.25_best", DIFF27), dS1["mean"], dS1["lo"], dS1["hi"], dS4["mean"], dS4["lo"], dS4["hi"],
     flip(b69, "clamp@4_all", ALL59), flip(b69, "clamp@4_all", FAIL27), dS8["mean"], dS8["lo"], dS8["hi"], d66["mean"], d66["lo"], d66["hi"],
     flip(b76, "clamp@1_all"), flip(b76, "clamp@4_all"), d76all["mean"], d76all["lo"], d76all["hi"], 1 - flip(b76, "clamp@4_all") - ans(b76, "clamp@4_all"),
     flip(b76, "orth@0.25_h8"), flip(b76, "orth@0.25_best")],
    ["n", "n", "n", "ci", "ci", "n", "n", "n", "n", "n", "ci", "ci", "n", "ci", "ci", "n", "n", "n", "ci", "ci", "n", "ci", "ci", "n", "n", "n", "ci", "ci", "n", "n", "n"])
auc69 = auc([r["h8_mass"]["best"] for r in b69], [r["arms"]["clamp@4_best"]["top1_is_swap"] for r in b69])
chk("H1", "H-1: 0.79 → 0.96, +1.66 [+0.88, +2.71]; OOS +0.72 / +1.38", ["0.79", "0.96", "+1.66", "+0.88", "+2.71", "+0.72", "+1.38"],
    [flip(b75, "clamp@4_best", F), flip(b75, "F[h8@23,1]+clamp@4_best", F), d75["mean"], d75["lo"], d75["hi"], d76a["mean"], d76b["mean"]], ["n", "n", "n", "ci", "ci", "n", "n"])
chk("H2", "H-2: h8 attention paired diff −0.001; K/V transplant diff 0.00", ["−0.001", "0.00"], [pr["1"][2], pr["1"][3]])
chk("H3", "H-3: 0.66 → 0.19, 0.31, 0.05, 0.64; ≈4.5×, 2.7×", ["0.66", "0.19", "0.31", "0.05", "0.64", "4.5", "2.7"],
    [rate(r["necessity"][a]["top1_is_swap"] for r in b77) for a in ("full", "full-PCA256", "full-R256", "full-(R256+PCA256)", "full-rand256")]
    + [float(np.mean([med(r["growth"][s]["layers"][-1]["realised_d20"] / r["growth"][s]["realised_L8_on_d20"] for r in b77) for s in ("J+R256", "J+PCA256", "full")])),
       med(r["growth"]["J+rand256"]["layers"][-1]["realised_d20"] / r["growth"]["J+rand256"]["realised_L8_on_d20"] for r in b77)])
chk("H4", "H-4: in-sample AUC 0.84 (block69 h8→p_best predicting clamp@4_best flip); OOS 0.88–0.92; 0.90; 46%; 0.61; 0.29 → 0.71; +5.8 [+4.4, +7.1]; 22%",
    ["0.84", "0.88", "0.92", "0.90", "46%", "0.61", "0.29", "0.71", "+5.8", "+4.4", "+7.1", "22%"],
    [auc69, pb["h8"], ph["h8"], min(pb["prod"], ph["prod"]), n_incons / 59, flip(b69, "clamp@4_all", ALL59), flip(b76, "clamp@1_all"), flip(b76, "clamp@4_all"),
     d76all["mean"], d76all["lo"], d76all["hi"], 1 - flip(b76, "clamp@4_all") - ans(b76, "clamp@4_all")], ["n"] * 8 + ["n", "ci", "ci", "n"])
chk("H6", "H-6: 59 items (9B); v2 non-geographic 28", ["59", "28"], [len(b75), len(b76)])

# ============================================================================= seed sensitivity of the CIs
CI_CONTRASTS = [
    ("S.1", b69, "orth@0.25_h8", "orth@0.25_best", DIFF27, CL9B), ("S.2", b69, "clamp@4_h8", "clamp@4_best", DIFF27, CL9B),
    ("S.3", b69, "clamp@1_h8", "clamp@1_best", DIFF27, CL9B), ("S.4", b69, "clamp@4_all", "clamp@4_best", DIFF27, CL9B),
    ("S.5a", b69, "orth@0.25_h8", "orth@0.25_best", ALL59, CL9B), ("S.5b", b69, "clamp@4_h8", "clamp@4_best", ALL59, CL9B),
    ("S.5c", b69, "clamp@4_all", "clamp@4_best", ALL59, CL9B), ("S.6", b66, "K@19,23+clamp2d@4", "clamp2d@4", FLIP24, CL9B),
    ("S.7", b73, "PAT_h8@23+clamp", "clamp2d@4", FLIP24, CL9B), ("S.8", b69, "K<-foil+clamp@4_best", "clamp@4_best", FLIP24, CL9B),
    ("S.9", b66, "K@19,23+clamp2d@4", "clamp2d@4", ALL59, CL9B), ("75.10", b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", FLIP24, CL9B),
    ("75.11", b75, "F[h8@23,1]+clamp@4_h8", "clamp@4_h8", FLIP24, CL9B), ("75.12", b75, "F[all@23,1]+clamp@4_h8", "clamp@4_h8", FLIP24, CL9B),
    ("75.16", b75, "F[h8@23,1]+clamp@4_best", "clamp@4_best", ALL59, CL9B),
] + [(cid, b76, a, b, IDX76, CL_V2) for cid, a, b, _, _ in ci76]


def seed_spread(seeds=(0, 1, 2)):
    worst = 0.0
    for cid, rows, a, b, idx, cl in CI_CONTRASTS:
        res = [paired(rows, a, b, idx, cl, seed=s) for s in seeds]
        worst = max(worst, max(abs(r["lo"] - res[0]["lo"]) for r in res), max(abs(r["hi"] - res[0]["hi"]) for r in res))
    return worst


# ============================================================================= report
def main():
    n_checked = sum(len(c["logged"]) for c in CHECKS)
    n_bad = sum(sum(1 for o in c["ok"] if not o) for c in CHECKS)
    print(f"seed={SEED} n_boot={N_BOOT} | numbers checked: {n_checked} | mismatching numbers: {n_bad} | claims with a mismatch: {sum(1 for c in CHECKS if not all(c['ok']))}")
    print("\n## Mismatches")
    for c in CHECKS:
        if not all(c["ok"]):
            bad = [(l, r) for l, r, o in zip(c["logged"], c["rec"], c["ok"]) if not o]
            print(f"- [{c['block'].split('.')[0]}] {c['id']} {c['claim']}: " + "; ".join(f"logged {l} vs recomputed {r:.4f}" for l, r in bad) + (f"  ({c['note']})" if c["note"] else ""))
    blk = None
    for c in CHECKS:
        if c["block"] != blk:
            blk = c["block"]
            print(f"\n## {blk}\n\n| # | Claim | Logged | Recomputed | n | Verdict |\n|---|---|---|---|---|---|")
        rec = " / ".join(f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in c["rec"])
        verdict = "MATCH" if all(c["ok"]) else "MISMATCH"
        note = f" — {c['note']}" if c["note"] else ""
        print(f"| {c['id']} | {c['claim']}{note} | {' / '.join(c['logged'])} | {rec} | {len(c['logged'])} | {verdict} |")
    print(f"\nCI seed spread (max |Δ bound| across seeds 0,1,2 over {len(CI_CONTRASTS)} contrasts): {seed_spread():.3f}")


if __name__ == "__main__":
    main()
