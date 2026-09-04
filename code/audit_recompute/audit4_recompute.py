#!/usr/bin/env python3
"""Recompute audit 4: log blocks 69-74 (+ block-74 4B paragraph), rows C18-C21, and the recomputed numbers of the
"审计 3（推导）结果与处置" paragraph.  Independent of the project's analysis scripts; reads only raw jsonl/json.
Run from code/:  python3 audit_recompute/audit4_recompute.py
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
from scipy import stats

RES = pathlib.Path(__file__).resolve().parents[3] / "正式研究同步" / "results"
BAND = list(range(8, 21))


def jl(rel):
    return [json.loads(l) for l in (RES / rel).read_text(encoding="utf-8").splitlines() if l.strip()]


def med(xs):
    xs = list(xs)
    return float(np.median(xs)) if xs else float("nan")


def rate(xs):
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def auc(score, label):
    """Mann-Whitney AUC, ties count 0.5."""
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def groups_9b():
    sup = jl("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = jl("block62_energy_of_arms/energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    fail27 = zero - resc4
    allidx = {r["index"] for r in e62}
    flip24 = allidx - zero
    return zero, resc4, fail27, flip24, e62


def groups_4b():
    rep = jl("block27_4b_band8_20/replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    e65 = jl("block65_4b_energy_of_arms/energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e65 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    return zero, resc4


ZERO, RESC4, FAIL27, FLIP24, E62 = groups_9b()
print(f"[groups 9B] zero-flip {len(ZERO)}, clamp2d@4 rescued {len(RESC4)}, fail27 {len(FAIL27)}, flippable {len(FLIP24)}")


# ----------------------------------------------------------------------------------------------------------- block 69
def block69():
    rows = jl("block69_position_choice/position_choice.jsonl")
    G = {"all59": rows, "flip24": [r for r in rows if r["index"] in FLIP24], "zero35": [r for r in rows if r["index"] in ZERO],
         "fail27": [r for r in rows if r["index"] in FAIL27]}
    print("\n===== block 69 =====")
    for g, rs in G.items():
        print(f"  [{g}] n={len(rs)} p_h8==p_best {rate(r['p_best'] == r['p_h8'] for r in rs):.3f} ({sum(r['p_best'] == r['p_h8'] for r in rs)}) | h8 mass@best {med(r['h8_mass']['best'] for r in rs):.3f} @h8pos {med(r['h8_mass']['h8pos'] for r in rs):.3f} | hits best {med(r['hits_best'] for r in rs):.1f} h8pos {med(r['hits_h8'] for r in rs):.1f}")
    f27 = G["fail27"]
    toks = [r["tok_h8"] for r in f27]
    print("  fail27 tok_h8:", Counter(toks).most_common())
    print("  fail27 function words (' the',' a'):", sum(t in (" the", " a") for t in toks), "| p_h8 != p_best:", sum(r["p_h8"] != r["p_best"] for r in f27))
    print("  clean h8_best median all59:", round(med(r["arms"]["clean"]["h8_best"] for r in rows), 3), " flip24:", round(med(r["arms"]["clean"]["h8_best"] for r in G["flip24"]), 3))
    arms = ["clamp@1_best", "clamp@4_best", "clamp@4_h8", "clamp@4_both", "clamp@1_all", "clamp@4_all", "orth@0.25_best", "orth@0.25_h8", "full@0.25_h8",
            "foil_orth@0.25_best", "K<-donor", "K<-foil", "K<-donor+clamp@4_best", "K<-foil+clamp@4_best", "clamp@1_h8", "foil_full@0.25_best", "K<-foil+clamp@1_best"]
    print(f"  {'arm':24s} " + " | ".join(f"{g:>26s}" for g in G))
    for a in arms:
        cells = []
        for g, rs in G.items():
            q = [r["arms"][a] for r in rs]
            cells.append(f"f{rate(x['top1_is_swap'] for x in q):.3f}({sum(x['top1_is_swap'] for x in q):2d}) ans{rate(x['top1_is_answer'] for x in q):.2f} h8b{med(x['h8_best'] for x in q):.3f} dM{med(x['delta_margin'] for x in q):+.2f}")
        print(f"  {a:24s} " + " | ".join(cells))
    # consistency with block 62 / 66
    e62 = {(r["index"], r["arm"]): r["delta_margin"] for r in E62}
    for a69, a62 in (("clamp@4_best", "clamp2d@4"), ("clamp@1_best", "clamp2d@1"), ("orth@0.25_best", "orth@0.25")):
        d = [abs(r["arms"][a69]["delta_margin"] - e62[(r["index"], a62)]) for r in rows]
        print(f"  consistency {a69} vs block62 {a62}: max|diff| {max(d):.4f}")
    k66 = {r["index"]: r["arms"] for r in jl("block66_key_value/key_value.jsonl")}
    for a69, a66 in (("K<-donor+clamp@4_best", "K@19,23+clamp2d@4"), ("K<-donor", "K@19,23")):
        d = [abs(r["arms"][a69]["delta_margin"] - k66[r["index"]][a66]["delta_margin"]) for r in rows]
        print(f"  consistency {a69} vs block66 {a66}: max|diff| {max(d):.4f}")
    # audit-3 M-4 numbers on flippable 24
    fl = G["flip24"]
    for a in ("K<-foil+clamp@4_best", "K<-donor+clamp@4_best", "clamp@4_best"):
        q = [r["arms"][a] for r in fl]
        print(f"  flip24 {a:24s}: flips {sum(x['top1_is_swap'] for x in q)}/24, median dM {med(x['delta_margin'] for x in q):.2f}, h8_best {med(x['h8_best'] for x in q):.3f}")
    d_foil = [r["arms"]["K<-foil+clamp@4_best"]["delta_margin"] - r["arms"]["clamp@4_best"]["delta_margin"] for r in fl]
    d_don = [r["arms"]["K<-donor+clamp@4_best"]["delta_margin"] - r["arms"]["K<-foil+clamp@4_best"]["delta_margin"] for r in fl]
    print(f"  foil+clamp > clamp on {sum(x > 0 for x in d_foil)}/24 (median {med(d_foil):+.2f}); donor > foil on {sum(x > 0 for x in d_don)}/24 (median {med(d_don):+.2f})")
    # foil orth: top1 is the foil's answer?
    def is_foil_answer(r, arm):
        t = r["arms"][arm]["top1_str"].strip().lower()
        f = r["foil_answer"].strip().lower()
        return bool(t) and (f.startswith(t) or t.startswith(f[: max(1, len(t))]))
    for a in ("foil_orth@0.25_best", "K<-foil"):
        print(f"  {a}: top1 = foil answer in {rate(is_foil_answer(r, a) for r in rows):.2f}; top1 neither answer nor swap: {rate((not r['arms'][a]['top1_is_swap']) and (not r['arms'][a]['top1_is_answer']) for r in rows):.2f}")


# ----------------------------------------------------------------------------------------------------------- block 71
def block71():
    rows = jl("block71_necessity_ladder/necessity_ladder.jsonl")
    print("\n===== block 71 =====  n rows", len(rows), "items", len({r["index"] for r in rows}), "arms", sorted({r["arm"] for r in rows}))
    for arm in ["full", "full-Jtop2", "full-Jtop16", "full-Jtop64", "full-Jtop256", "full-Jtop1024", "full-Rtop256", "full-rand16", "full-rand64", "full-rand256", "full-rand1024"]:
        s = [r for r in rows if r["arm"] == arm]
        z = [r for r in s if r["index"] in ZERO]
        print(f"  {arm:14s} n={len(s)} flip {rate(r['top1_is_swap'] for r in s):.3f} ({sum(r['top1_is_swap'] for r in s)}) zero {rate(r['top1_is_swap'] for r in z):.3f} ({sum(r['top1_is_swap'] for r in z)}/{len(z)}) dM {med(r['delta_margin'] for r in s):+.2f} KL {med(r['kl'] for r in s):.3f} E {med(r['energy'] for r in s):.1f}")
    e62 = {(r["index"], r["arm"]): r for r in E62}
    for a71, a62 in (("full", "full@0.25"), ("full-Jtop2", "orth@0.25")):
        s = [r for r in rows if r["arm"] == a71]
        b = [e62[(r["index"], a62)] for r in s]
        print(f"  block62 {a62}: flip {rate(r['top1_is_swap'] for r in b):.3f} KL {med(r['kl'] for r in b):.3f} E {med(r['energy'] for r in b):.1f}; max|dM diff| {max(abs(x['delta_margin'] - y['delta_margin']) for x, y in zip(s, b)):.4f} max|KL diff| {max(abs(x['kl'] - y['kl']) for x, y in zip(s, b)):.4f}")


# ----------------------------------------------------------------------------------------------------------- block 70
def block70():
    rows = jl("block70_4b_key_value/key_value.jsonl")
    chk = json.load(open(RES / "block70_4b_key_value/self_check.json"))
    zero4, resc4 = groups_4b()
    print("\n===== block 70 (4B) =====  n", len(rows), "| self_check", chk)
    G = {"all47": rows, "flip": [r for r in rows if r["index"] not in zero4], "fail": [r for r in rows if r["index"] in zero4 and r["index"] not in resc4],
         "zero∩47": [r for r in rows if r["index"] in zero4]}
    print("  group sizes:", {g: len(rs) for g, rs in G.items()}, "| 4B zero set size", len(zero4), "resc4", len(resc4))
    arms = ["clean", "clamp2d@4", "orth@0.25", "K@19,23", "V@19,23", "KV@19,23", "PAT@19,23", "K@19,23+clamp2d@4", "PAT@19,23+clamp2d@4", "KV@19,23+clamp2d@4"]
    for a in arms:
        cells = []
        for g in ("flip", "fail"):
            q = [r["arms"][a] for r in G[g]]
            cells.append(f"{g}: f{rate(x['top1_is_swap'] for x in q):.3f}({sum(x['top1_is_swap'] for x in q):2d}/{len(q)}) dM{med(x['delta_margin'] for x in q):+.2f} h8 {med(x['L23_h8_to_bridge'] for x in q):.3f}")
        print(f"  {a:22s} " + " | ".join(cells))


# ----------------------------------------------------------------------------------------------------------- block 72
def block72():
    rows = jl("block72_subspace_controls/subspace_controls.jsonl")
    print("\n===== block 72 =====  n", len(rows), "n_entdiff", sorted({r["n_entdiff"] for r in rows}))
    SHOW = (9, 10, 12, 15, 18, 20)
    for arm in ("J+R256", "J+J256", "J+WU256", "J+PCA256", "J+entdiff", "J+rand58", "J+rand256"):
        m = [r["arms"][arm]["multi"] for r in rows]
        mz = [r["arms"][arm]["multi"] for r in rows if r["index"] in ZERO]
        s = [r["arms"][arm]["single_L8"] for r in rows]
        sz = [r["arms"][arm]["single_L8"] for r in rows if r["index"] in ZERO]
        ratios_all = [med(x["injected"][BAND.index(l)] / max(x["static"][BAND.index(l)], 1e-9) for x in m) for l in range(9, 21)]
        ratios_show = [ratios_all[l - 9] for l in SHOW]
        r8 = med(x["realised"][0] for x in s)
        r20 = med(x["realised"][12] for x in s)
        amp_item = med(x["realised"][12] / x["realised"][0] for x in s if x["realised"][0] > 1e-9)
        print(f"  {arm:10s} multi flip {rate(x['top1_is_swap'] for x in m):.3f}({sum(x['top1_is_swap'] for x in m)}) zero {rate(x['top1_is_swap'] for x in mz):.3f}({sum(x['top1_is_swap'] for x in mz)}) | inj {med(sum(x['injected']) for x in m):.1f} static {med(sum(x['static']) for x in m):.1f} | ratio L9-20 all [{min(ratios_all):.3f},{max(ratios_all):.3f}] shown(9,10,12,15,18,20) [{min(ratios_show):.3f},{max(ratios_show):.3f}] L8 {med(x['injected'][0] / max(x['static'][0], 1e-9) for x in m):.2f} | single flip {rate(x['top1_is_swap'] for x in s):.3f}({sum(x['top1_is_swap'] for x in s)}) zero {rate(x['top1_is_swap'] for x in sz):.3f}({sum(x['top1_is_swap'] for x in sz)}) | realised L8 {r8:.3f} -> L20 {r20:.3f} amp(med ratio) {r20 / r8:.3f} amp(item-median) {amp_item:.3f}")
        print("      realised per layer (median):", " ".join(f"{med(x['realised'][k] for x in s):.2f}" for k in range(13)))


# ----------------------------------------------------------------------------------------------------------- block 73
def block73():
    rows = jl("block73_routing_all_layers/routing_all_layers.jsonl")
    chk = [json.loads(l) for l in open(RES / "block73_routing_all_layers/self_check.json")] if False else None
    txt = (RES / "block73_routing_all_layers/self_check.json").read_text()
    try:
        chk = json.loads(txt)
        chk_ok = [chk["identity_ok"]]
    except json.JSONDecodeError:
        chk_ok = [json.loads(l)["identity_ok"] for l in txt.splitlines() if l.strip()]
    print("\n===== block 73 =====  n", len(rows), "| self_check identity_ok:", chk_ok)
    G = {"all59": rows, "flip24": [r for r in rows if r["index"] in FLIP24], "fail27": [r for r in rows if r["index"] in FAIL27]}
    arms = ["clamp2d@4", "orth@0.25", "orth-restore", "K19,23", "KV19,23", "lin_in", "K19,23+lin_in", "KV19,23+lin_in", "transport<-clamp",
            "K19,23+clamp", "lin_in+clamp", "K19,23+lin_in+clamp", "KV19,23+lin_in+clamp", "PAT_h8@23+clamp", "KV19,23+clamp"]
    for a in arms:
        q = {g: [r["arms"][a] for r in rs] for g, rs in G.items()}
        print(f"  {a:22s} all {rate(x['top1_is_swap'] for x in q['all59']):.3f}({sum(x['top1_is_swap'] for x in q['all59']):2d}) | flip24 {rate(x['top1_is_swap'] for x in q['flip24']):.3f}({sum(x['top1_is_swap'] for x in q['flip24']):2d}) dM {med(x['delta_margin'] for x in q['flip24']):+.2f} | fail27 {rate(x['top1_is_swap'] for x in q['fail27']):.3f}({sum(x['top1_is_swap'] for x in q['fail27']):2d}) dM {med(x['delta_margin'] for x in q['fail27']):+.2f}")
    sel = [r for r in rows if r["arms"]["orth@0.25"]["delta_margin"] > 1]
    sh = [r["arms"]["KV19,23+lin_in"]["delta_margin"] / r["arms"]["orth@0.25"]["delta_margin"] for r in sel]
    rest = [r["arms"]["orth-restore"]["delta_margin"] / r["arms"]["orth@0.25"]["delta_margin"] for r in sel]
    print(f"  share via transport inputs (KV+lin_in / orth): median {med(sh):.3f}; residual after restore (orth-restore / orth): median {med(rest):.3f}; 1-residual {1 - med(rest):.3f}; n={len(sel)} (orth dM>1)")
    sh0 = [r["arms"]["orth-restore"]["delta_margin"] / r["arms"]["orth@0.25"]["delta_margin"] for r in rows if r["arms"]["orth@0.25"]["delta_margin"] > 0]
    print(f"  (variant: orth dM>0, n={len(sh0)}: residual median {med(sh0):.3f})")
    fl = G["flip24"]
    d = [r["arms"]["PAT_h8@23+clamp"]["delta_margin"] - r["arms"]["clamp2d@4"]["delta_margin"] for r in fl]
    print(f"  PAT_h8@23+clamp − clamp on flip24: >0 in {sum(x > 0 for x in d)}/{len(d)}, median {med(d):+.3f}")
    d = [r["arms"]["K19,23+clamp"]["delta_margin"] - r["arms"]["clamp2d@4"]["delta_margin"] for r in fl]
    print(f"  K19,23+clamp − clamp on flip24 (block73 copy): >0 in {sum(x > 0 for x in d)}/{len(d)}, median {med(d):+.3f}")
    # consistency with block 66
    k66 = {r["index"]: r["arms"] for r in jl("block66_key_value/key_value.jsonl")}
    for a73, a66 in (("K19,23", "K@19,23"), ("KV19,23", "KV@19,23"), ("K19,23+clamp", "K@19,23+clamp2d@4"), ("clamp2d@4", "clamp2d@4"), ("orth@0.25", "orth@0.25")):
        dd = [abs(r["arms"][a73]["delta_margin"] - k66[r["index"]][a66]["delta_margin"]) for r in rows]
        print(f"  consistency {a73} vs block66 {a66}: max|diff| {max(dd):.4f}")


# ----------------------------------------------------------------------------------------------------------- block 74
def block74(tag):
    rows = jl(f"block74{tag}_random_token_planes/random_token_planes.jsonl")
    print(f"\n===== block 74{tag or ' 9B'} =====  n {len(rows)} | layers {[q['layer'] for q in rows[0]['layers']]} | n_other {len(rows[0]['layers'][0]['share_other'])} n_rand {len(rows[0]['layers'][0]['share_rand'])}")
    own = [np.mean([q["share_own"] for q in r["layers"]]) for r in rows]
    oth = [np.mean([np.median(q["share_other"]) for q in r["layers"]]) for r in rows]
    rnd = [np.mean([np.median(q["share_rand"]) for q in r["layers"]]) for r in rows]
    print(f"  grad share: own {med(own):.4f} other {med(oth):.4f} rand {med(rnd):.4f} | own/other {med(a / b for a, b in zip(own, oth)):.2f}x own/rand {med(a / b for a, b in zip(own, rnd)):.2f}x | ratio-of-medians {med(own) / med(oth):.2f} / {med(own) / med(rnd):.2f}")
    down = [np.mean([q["dshare_own"] for q in r["layers"]]) for r in rows]
    doth = [np.mean([np.median(q["dshare_other"]) for q in r["layers"]]) for r in rows]
    drnd = [np.mean([np.median(q["dshare_rand"]) for q in r["layers"]]) for r in rows]
    print(f"  donor-diff share: own {med(down):.4f} other {med(doth):.4f} rand {med(drnd):.4f} | own/other {med(a / b for a, b in zip(down, doth)):.2f}x own/rand {med(a / b for a, b in zip(down, drnd)):.2f}x | ratio-of-medians {med(down) / med(doth):.2f} / {med(down) / med(drnd):.2f}")
    pct = [np.mean([np.mean([o < q["share_own"] for o in q["share_other"]]) for q in r["layers"]]) for r in rows]
    print(f"  percentile vs other-item null: median {med(pct):.3f}; >=0.95: {sum(x >= 0.95 for x in pct)}/{len(pct)}")
    layers = sorted(int(l) for l in rows[0]["heads"]["clean"])
    nh = {l: len(rows[0]["heads"]["clean"][str(l)]) for l in layers}
    best = []
    for l in layers:
        for h in range(nh[l]):
            rise = med(r["heads"]["orth@0.25"][str(l)][h] - r["heads"]["clean"][str(l)][h] for r in rows)
            clean = med(r["heads"]["clean"][str(l)][h] for r in rows)
            best.append((rise, l, h, clean))
    best.sort(reverse=True)
    print(f"  head search layers {layers} heads/layer {sorted(set(nh.values()))}: top-6: " + "; ".join(f"L{l} h{h} +{rise:.3f} (clean {c:.3f})" for rise, l, h, c in best[:6]))
    # consistency of 9B own share with block 63 share_J
    if not tag:
        b63 = {r["index"]: r for r in jl("block63_why_plane_ignored/why_plane_ignored.jsonl")}
        dd = [abs(q["share_own"] - b63[r["index"]]["layers"][k]["share_J"]) for r in rows for k, q in enumerate(r["layers"])]
        print(f"  consistency share_own vs block63 share_J: max|diff| {max(dd):.2e}")
        h66 = {r["index"]: r["arms"]["clean"]["L23_h8_to_bridge"] for r in jl("block66_key_value/key_value.jsonl")}
        dd = [abs(r["heads"]["clean"]["23"][8] - h66[r["index"]]) for r in rows]
        print(f"  consistency heads clean L23 h8 vs block66 clean L23_h8_to_bridge: max|diff| {max(dd):.2e}")


# ------------------------------------------------------------------------------------------ audit-3 paragraph numbers
def audit3_para():
    print("\n===== audit-3 paragraph (recomputed numbers) =====")
    k66 = jl("block66_key_value/key_value.jsonl")
    fl = [r for r in k66 if r["index"] in FLIP24]
    fails = [r for r in fl if not r["arms"]["clamp2d@4"]["top1_is_swap"]]
    flips = [r for r in fl if r["arms"]["clamp2d@4"]["top1_is_swap"]]
    print(f"  flip24: clamp2d@4 fails on {len(fails)}; their clean h8: {sorted(round(r['arms']['clean']['L23_h8_to_bridge'], 3) for r in fails)}; flippers' median clean h8 {med(r['arms']['clean']['L23_h8_to_bridge'] for r in flips):.3f}; within-group AUC {auc([r['arms']['clean']['L23_h8_to_bridge'] for r in fl], [r['arms']['clamp2d@4']['top1_is_swap'] for r in fl]):.3f}")
    print(f"  donor-K+clamp rescues {sum(r['arms']['K@19,23+clamp2d@4']['top1_is_swap'] for r in fails)}/{len(fails)} of them (K alone: {sum(r['arms']['K@19,23']['top1_is_swap'] for r in fails)}; names {[r['name'] for r in fails]})")
    d = [r["arms"]["K@19,23+clamp2d@4"]["delta_margin"] - r["arms"]["clamp2d@4"]["delta_margin"] for r in fl]
    w = stats.wilcoxon(d)
    w_ex = stats.wilcoxon(d, method="exact") if hasattr(stats, "wilcoxon") else None
    print(f"  K+clamp − clamp on flip24: >0 {sum(x > 0 for x in d)}/24, median {med(d):+.3f}, Wilcoxon two-sided p {w.pvalue:.2e} (exact {w_ex.pvalue:.2e}); flips K+clamp {sum(r['arms']['K@19,23+clamp2d@4']['top1_is_swap'] for r in fl)}/24 vs clamp {sum(r['arms']['clamp2d@4']['top1_is_swap'] for r in fl)}/24")
    f27 = [r for r in k66 if r["index"] in FAIL27]
    print(f"  fail27: K+clamp median dM {med(r['arms']['K@19,23+clamp2d@4']['delta_margin'] for r in f27):+.3f}, KV median dM {med(r['arms']['KV@19,23']['delta_margin'] for r in f27):+.3f}, orth@0.25 {med(r['arms']['orth@0.25']['delta_margin'] for r in f27):+.3f}; K+clamp flips {sum(r['arms']['K@19,23+clamp2d@4']['top1_is_swap'] for r in f27)}/27")
    # K / pattern alone
    for a in ("K@19,23", "PAT@19,23"):
        print(f"  {a}: flip all {rate(r['arms'][a]['top1_is_swap'] for r in k66):.3f} flip24 {rate(r['arms'][a]['top1_is_swap'] for r in fl):.3f}; items dM<0: all {sum(r['arms'][a]['delta_margin'] < 0 for r in k66)} (<-0.25: {sum(r['arms'][a]['delta_margin'] < -0.25 for r in k66)}, <=-0.5: {sum(r['arms'][a]['delta_margin'] <= -0.5 for r in k66)}), flip24 {sum(r['arms'][a]['delta_margin'] < 0 for r in fl)}")
    # family / tertile AUCs
    b63 = jl("block63_why_plane_ignored/why_plane_ignored.jsonl")
    cat = {r["index"]: r["category"] for r in json.load(open(RES / "block01/q01_data_audit_rows.json"))}
    lab = {r["index"]: bool(r["top1_is_swap"]) for r in E62 if r["arm"] == "clamp2d@4"}
    h8 = {r["index"]: r["attention"]["clean"]["L23_h8_to_bridge"] for r in b63}
    gap = {r["index"]: float(np.mean([q["gap"] for q in r["layers"]])) for r in b63}
    idx = sorted(h8)
    print(f"  overall AUC h8 {auc([h8[i] for i in idx], [lab[i] for i in idx]):.3f} gap {auc([gap[i] for i in idx], [lab[i] for i in idx]):.3f} (n={len(idx)}, pos={sum(lab[i] for i in idx)})")
    fam = Counter(cat[i] for i in idx)
    for f in ("multihop", "city-capital", "person-firstname", "language-capital"):
        ii = [i for i in idx if cat[i] == f]
        print(f"  family {f:16s} n={len(ii)} pos={sum(lab[i] for i in ii)}: AUC h8 {auc([h8[i] for i in ii], [lab[i] for i in ii]):.3f} gap {auc([gap[i] for i in ii], [lab[i] for i in ii]):.3f}")
    order = sorted(idx, key=lambda i: gap[i])
    for name, parts in (("array_split(20,20,19)", np.array_split(np.array(order), 3)), ("split(19,20,20)", [order[:19], order[19:39], order[39:]])):
        out = []
        for part in parts:
            part = list(part)
            out.append(f"n={len(part)} pos={sum(lab[i] for i in part)} AUC {auc([h8[i] for i in part], [lab[i] for i in part]):.3f}")
        print(f"  gap tertiles [{name}]: " + " | ".join(out))
    q1, q2 = np.percentile([gap[i] for i in idx], [100 / 3, 200 / 3])
    parts = [[i for i in idx if gap[i] <= q1], [i for i in idx if q1 < gap[i] <= q2], [i for i in idx if gap[i] > q2]]
    print("  gap tertiles [percentile cut]: " + " | ".join(f"n={len(p)} pos={sum(lab[i] for i in p)} AUC {auc([h8[i] for i in p], [lab[i] for i in p]):.3f}" for p in parts))
    # single-hop delta log p (block 60 part b)
    b60 = [r for r in jl("block60_selfconsistent/selfconsistent.jsonl") if r["part"] == "b"]
    for kind in ("single_hop", "two_hop"):
        for arm in ("attn_17_24", "mlp_26_29", "attn_9_16"):
            s = [r for r in b60 if r["prompt_kind"] == kind and r["arm"] == arm]
            sc = [r for r in s if r["clean_correct"]]
            print(f"  block60b {kind:10s} {arm:10s}: n={len(s)} clean acc {rate(r['clean_correct'] for r in s):.3f} still acc {rate(r['still_correct'] for r in s):.3f} (among clean-correct {rate(r['still_correct'] for r in sc):.3f}) | dlogp mean {np.mean([r['delta_logp_answer'] for r in s]):+.3f} median {med(r['delta_logp_answer'] for r in s):+.3f} | clean-correct only: mean {np.mean([r['delta_logp_answer'] for r in sc]):+.3f} median {med(r['delta_logp_answer'] for r in sc):+.3f}")


if __name__ == "__main__":
    block69()
    block71()
    block70()
    block72()
    block73()
    block74("")
    block74("_4b")
    audit3_para()
