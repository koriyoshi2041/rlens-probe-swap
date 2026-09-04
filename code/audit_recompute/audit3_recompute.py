#!/usr/bin/env python3
"""Independent recomputation of every number in log blocks 62-68 (read-only on results).
Run:  python3 audit_recompute/audit3_recompute.py   (from the code directory)
Prints one line per checked quantity: [tag] recomputed value(s).  Comparison with the log is
done in README_recompute_audit_3.md.  Nothing here imports the project's analysis scripts."""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[3] / "正式研究同步" / "results"
BAND = list(range(8, 21))
SHOW = (9, 10, 12, 15, 18, 20)


def load(rel: str):
    return [json.loads(l) for l in (RES / rel).read_text(encoding="utf-8").splitlines() if l.strip()]


def med(xs):
    xs = [float(x) for x in xs]
    return float(np.median(xs)) if xs else float("nan")


def rate(xs):
    return float(np.mean([bool(x) for x in xs])) if xs else float("nan")


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def rank(a):
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a))
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def band_mean(row, key):
    return float(np.mean([q[key] for q in row["layers"]]))


# ----------------------------------------------------------------------------- group definitions
def groups_9b():
    sup = load("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load("block62_energy_of_arms/energy_of_arms.jsonl")
    resc = {arm: {r["index"] for r in e62 if r["arm"] == arm and r["top1_is_swap"]} for arm in ("clamp2d@4", "orth@0.1", "orth@0.25")}
    return zero, resc, e62


def groups_4b():
    rep = load("block27_4b_band8_20/replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    e65 = load("block65_4b_energy_of_arms/energy_of_arms.jsonl")
    resc = {arm: {r["index"] for r in e65 if r["arm"] == arm and r["top1_is_swap"]} for arm in ("clamp2d@4", "orth@0.1", "orth@0.25")}
    return zero, resc, e65


ARM_ORDER = ["clamp2d@1", "clamp2d@1.5", "clamp2d@2", "clamp2d@3", "clamp2d@4", "orth@0.1", "orth@0.25", "orth@0.5", "orth@1",
             "full@0.25", "full@0.5", "full@1", "rand16", "rand64", "rand256", "Rtop256"]


# ----------------------------------------------------------------------------- block 62 / 65
def energy_block(tag, rows, zero, dim):
    idx = sorted({r["index"] for r in rows})
    z = [i for i in idx if i in zero]
    print(f"\n[{tag}] n_items={len(idx)}  zero-flip={len(z)}")
    first = {r["index"]: r for r in rows if r["arm"] == "clamp2d@4"}
    de = [r["donor_diff_energy"] for r in first.values()]
    share = [r["donor_diff_inplane_energy"] / r["donor_diff_energy"] for r in first.values()]
    print(f"[{tag}.donor] static donor-diff energy median {med(de):.0f}; in-plane share median {med(share):.4f}; chance 2/{dim}={2 / dim:.5f}; enrichment {med(share) / (2 / dim):.1f}x")
    print(f"[{tag}.table] arm | median energy on zero set | rescued/n_zero | median energy rescued | not rescued | median KL not rescued | median KL zero")
    for arm in ARM_ORDER:
        s = [r for r in rows if r["arm"] == arm and r["index"] in zero]
        rs = [r for r in s if r["top1_is_swap"]]
        nr = [r for r in s if not r["top1_is_swap"]]
        print(f"   {arm:12s} E={med([r['energy'] for r in s]):7.1f}  resc {len(rs):2d}/{len(s)}  E_resc {med([r['energy'] for r in rs]):7.1f}  E_not {med([r['energy'] for r in nr]):7.1f}  KL_not {med([r['kl'] for r in nr]):.3f}  KL_zero {med([r['kl'] for r in s]):.3f}")
    c4 = {r["index"]: r for r in rows if r["arm"] == "clamp2d@4"}
    o1 = {r["index"]: r for r in rows if r["arm"] == "orth@0.1"}
    ratio = np.array([c4[i]["energy"] / o1[i]["energy"] for i in z])
    both = sum(c4[i]["top1_is_swap"] and o1[i]["top1_is_swap"] for i in z)
    oc = sum(c4[i]["top1_is_swap"] and not o1[i]["top1_is_swap"] for i in z)
    oo = sum(o1[i]["top1_is_swap"] and not c4[i]["top1_is_swap"] for i in z)
    print(f"[{tag}.pair] clamp2d@4 injects more than orth@0.1 on {(ratio > 1).sum()}/{len(z)}; median ratio {np.median(ratio):.2f}; both {both}, clamp-only {oc}, orth-only {oo}, neither {len(z) - both - oc - oo}")
    all_c4 = [r for r in rows if r["arm"] == "clamp2d@4"]
    print(f"[{tag}.c4all] clamp2d@4 flips on all items: {sum(r['top1_is_swap'] for r in all_c4)}/{len(all_c4)}")


# ----------------------------------------------------------------------------- block 63 / 67
def grad_block(tag, rows, zero, resc, real, chance256, head_key, sum_key, arms, orth_resc_arm):
    groups = {
        "flippable": [r for r in rows if r["index"] not in zero],
        "zero&c4resc": [r for r in rows if r["index"] in zero and r["index"] in resc["clamp2d@4"]],
        "zero&c4fail": [r for r in rows if r["index"] in zero and r["index"] not in resc["clamp2d@4"]],
        f"zero&{orth_resc_arm}resc": [r for r in rows if r["index"] in zero and r["index"] in resc[orth_resc_arm]],
        "zero&neither(c4,orth0.1)": [r for r in rows if r["index"] in zero and r["index"] not in resc["clamp2d@4"] and r["index"] not in resc["orth@0.1"]],
    }
    print(f"\n[{tag}.table] group n | share_J share_R rand2 J/rand2 | R256 rand256 R256/rand256 | gap | cos_g_clamp cos_g_d cos_g_dorth   (chance256={chance256:.4f})")
    for g, rs in groups.items():
        bm = lambda k: [band_mean(r, k) for r in rs]
        sj, sr, s2, s256, sr256 = bm("share_J"), bm("share_R"), bm("share_rand2"), bm("share_R256"), bm("share_rand256")
        print(f"   {g:26s} {len(rs):2d} | {med(sj):.4f} {med(sr):.4f} {med(s2):.5f} {med([a / b for a, b in zip(sj, s2)]):6.1f}x | {med(s256):.3f} {med(sr256):.4f} {med([a / b for a, b in zip(s256, sr256)]):.2f} | gap {med(bm('gap')):.2f} | {med(bm('cos_g_clamp')):.3f} {med(bm('cos_g_d')):.3f} {med(bm('cos_g_dorth')):.3f}")
    print(f"[{tag}.pred] group | clamp2d@4 pred/real | orth@0.1 pred/real | full@0.25 pred/real  (real = block energy file delta_margin, group median)")
    for g, rs in groups.items():
        line = f"   {g:26s}"
        for arm in ("clamp2d@4", "orth@0.1", "full@0.25"):
            line += f" | {med([r['pred'][arm] for r in rs]):6.2f} / {med([real[(r['index'], arm)] for r in rs]):5.2f}"
        print(line)
    print(f"[{tag}.att] {head_key} group medians per arm")
    for g, rs in groups.items():
        print(f"   {g:26s} " + "  ".join(f"{arm}={med([r['attention'][arm][head_key] for r in rs]):.3f}" for arm in arms))
    print(f"[{tag}.attsum] {sum_key} group medians per arm")
    for g, rs in groups.items():
        print(f"   {g:26s} " + "  ".join(f"{arm}={med([r['attention'][arm][sum_key] for r in rs]):.3f}" for arm in arms))
    zr = [r for r in rows if r["index"] in zero]
    for arm in ("clamp2d@4", "orth@0.1", "orth@0.25"):
        d = lambda r: r["attention"][arm][sum_key] - r["attention"]["clean"][sum_key]
        dr = [d(r) for r in zr if r["attention"][arm]["top1_is_swap"]]
        dn = [d(r) for r in zr if not r["attention"][arm]["top1_is_swap"]]
        print(f"[{tag}.dAtt] Δ{sum_key} under {arm}: rescued {med(dr):+.3f} (n={len(dr)}) vs not {med(dn):+.3f} (n={len(dn)})")
    for arm in ("clamp2d@4", "orth@0.1", "orth@0.25", "full@0.25"):
        diffs = [abs(r["attention"][arm]["delta_margin"] - real[(r["index"], arm)]) for r in rows]
        print(f"[{tag}.consist] Δmargin {arm} vs energy-file: max|diff| {max(diffs):.4f}")
    # pre-intervention AUCs
    lab4 = [r["index"] in resc["clamp2d@4"] for r in rows]
    labb = [r["index"] not in zero for r in rows]
    h = [r["attention"]["clean"][head_key] for r in rows]
    gap = [band_mean(r, "gap") for r in rows]
    fo = [r["pred"]["clamp2d@4"] for r in rows]
    sj = [band_mean(r, "share_J") for r in rows]
    print(f"[{tag}.auc] single-position clamp2d@4 flip (n={len(rows)}, pos={sum(lab4)}): h8 {auc(h, lab4):.3f} | first-order {auc(fo, lab4):.3f} | gap {auc(gap, lab4):.3f} | share_J {auc(sj, lab4):.3f}")
    zi = [k for k, r in enumerate(rows) if r["index"] in zero]
    print(f"[{tag}.aucZ] within zero set (n={len(zi)}, pos={sum(lab4[k] for k in zi)}): h8 {auc([h[k] for k in zi], [lab4[k] for k in zi]):.3f} | first-order {auc([fo[k] for k in zi], [lab4[k] for k in zi]):.3f} | gap {auc([gap[k] for k in zi], [lab4[k] for k in zi]):.3f} | share_J {auc([sj[k] for k in zi], [lab4[k] for k in zi]):.3f}")
    print(f"[{tag}.aucB] band-clamp flip (pos={sum(labb)}): h8 {auc(h, labb):.3f} | first-order {auc(fo, labb):.3f} | gap {auc(gap, labb):.3f} | share_J {auc(sj, labb):.3f}")
    print(f"[{tag}.corr] gap vs clean h8: pearson {np.corrcoef(gap, h)[0, 1]:+.3f}, spearman {np.corrcoef(rank(gap), rank(h))[0, 1]:+.3f}")
    return groups


def head_search_4b(rows):
    layers = sorted({int(k[1:].split("_")[0]) for k in rows[0]["attention"]["clean"] if k.endswith("_heads_to_bridge")})
    out = []
    for l in layers:
        nh = len(rows[0]["attention"]["clean"][f"L{l}_heads_to_bridge"])
        for hh in range(nh):
            rise = med([r["attention"]["orth@0.25"][f"L{l}_heads_to_bridge"][hh] - r["attention"]["clean"][f"L{l}_heads_to_bridge"][hh] for r in rows])
            clean = med([r["attention"]["clean"][f"L{l}_heads_to_bridge"][hh] for r in rows])
            out.append((rise, l, hh, clean))
    out.sort(reverse=True)
    print("[b67.heads] top-6 (layer, head) by median rise of attention onto bridge under orth@0.25 (all 47): " + "; ".join(f"L{l}h{h} rise {r:+.3f} clean {c:.3f}" for r, l, h, c in out[:6]))
    for l in layers:
        best = max([o for o in out if o[1] == l])
        print(f"[b67.heads] L{l} max rise: h{best[2]} {best[0]:+.3f} (clean {best[3]:.3f})")
    l23 = sorted([o for o in out if o[1] == 23], reverse=True)[:4]
    print("[b67.heads] L23 sorted: " + "; ".join(f"h{h} clean {c:.2f} rise {r:+.2f}" for r, l, h, c in l23))
    # consistency: L23_h8_to_bridge field == heads list [8]
    d = max(abs(r["attention"][a]["L23_h8_to_bridge"] - r["attention"][a]["L23_heads_to_bridge"][8]) for r in rows for a in r["attention"])
    print(f"[b67.heads] L23_h8_to_bridge field vs heads[8]: max|diff| {d:.2e}")


# ----------------------------------------------------------------------------- block 64 / 68
def prop_block(tag, rows, zero):
    subs = ("J_plane", "J+R256", "J+rand256", "full")
    print(f"\n[{tag}.multi] sub | flip all | flip zero | injected/static ratio medians at L{SHOW} | median sum injected / median sum static | range L9-L20")
    for sub in subs:
        rs = [r["arms"][sub]["multi"] for r in rows]
        rz = [r["arms"][sub]["multi"] for r in rows if r["index"] in zero]
        ratios = [med([m["injected"][BAND.index(l)] / m["static"][BAND.index(l)] for m in rs]) for l in SHOW]
        allr = [med([m["injected"][BAND.index(l)] / m["static"][BAND.index(l)] for m in rs]) for l in range(9, 21)]
        print(f"   {sub:10s} {rate([m['top1_is_swap'] for m in rs]):.3f} {rate([m['top1_is_swap'] for m in rz]):.3f} | " + " ".join(f"{x:.3f}" for x in ratios) + f" | {med([sum(m['injected']) for m in rs]):.1f} / {med([sum(m['static']) for m in rs]):.1f} | L9-20 [{min(allr):.3f},{max(allr):.3f}]")
    print(f"[{tag}.single] sub | flip all | flip zero | realised medians at L8,L{SHOW} | realised_sub L20 | moved L20 / moved L8 (median of ratio) | realised L20 flip/zero")
    for sub in subs:
        rs = [r["arms"][sub]["single_L8"] for r in rows]
        rz = [r["arms"][sub]["single_L8"] for r in rows if r["index"] in zero]
        rf = [r["arms"][sub]["single_L8"] for r in rows if r["index"] not in zero]
        f = [med([m["realised"][BAND.index(l)] for m in rs]) for l in (8,) + SHOW]
        fs = med([m["realised_sub"][12] for m in rs])
        amp = med([m["moved_energy"][12] / m["moved_energy"][0] for m in rs])
        print(f"   {sub:10s} {rate([m['top1_is_swap'] for m in rs]):.3f} {rate([m['top1_is_swap'] for m in rz]):.3f} | " + " ".join(f"{x:.3f}" for x in f) + f" | sub20 {fs:.3f} | amp {amp:.2f} | L20 flip {med([m['realised'][12] for m in rf]):.3f} zero {med([m['realised'][12] for m in rz]):.3f}")


# ----------------------------------------------------------------------------- block 66
def kv_block(zero, resc, e62):
    rows = load("block66_key_value/key_value.jsonl")
    chk = json.load(open(RES / "block66_key_value" / "self_check.json"))
    print(f"\n[b66.selfcheck] {chk}")
    groups = {"flippable24": [r for r in rows if r["index"] not in zero], "zero35": [r for r in rows if r["index"] in zero],
              "fail27": [r for r in rows if r["index"] in zero and r["index"] not in resc["clamp2d@4"]], "all59": rows}
    arms = list(rows[0]["arms"].keys())
    print("[b66.table] arm | per group: flip rate / median Δmargin / median L23 h8 / median L23 sum / n_flip")
    for arm in arms:
        line = f"   {arm:22s}"
        for g, rs in groups.items():
            a = [r["arms"][arm] for r in rs]
            line += f" | {g} {rate([q['top1_is_swap'] for q in a]):.3f} / {med([q['delta_margin'] for q in a]):+.2f} / {med([q['L23_h8_to_bridge'] for q in a]):.3f} / {med([q['L23_sum_to_bridge'] for q in a]):.2f} / {sum(q['top1_is_swap'] for q in a)}"
        print(line)
    rs = groups["fail27"]
    print("[b66.resc27] arm: rescued/27 ; toward original (not swap & ΔM<-0.5) ; (not swap & ΔM<0)")
    for arm in arms:
        a = [r["arms"][arm] for r in rs]
        print(f"   {arm:22s} {sum(q['top1_is_swap'] for q in a):2d}/27 ; {sum((not q['top1_is_swap']) and q['delta_margin'] < -0.5 for q in a):2d} ; {sum((not q['top1_is_swap']) and q['delta_margin'] < 0 for q in a):2d}")
    for g in ("flippable24", "zero35", "all59"):
        a = groups[g]
        print(f"[b66.toward] {g}: K-only ΔM<-0.5 & not swap {sum((not r['arms']['K@19,23']['top1_is_swap']) and r['arms']['K@19,23']['delta_margin'] < -0.5 for r in a)}; PAT-only {sum((not r['arms']['PAT@19,23']['top1_is_swap']) and r['arms']['PAT@19,23']['delta_margin'] < -0.5 for r in a)}")
    real = {(r["index"], r["arm"]): r["delta_margin"] for r in e62}
    for arm in ("clamp2d@4", "orth@0.25"):
        diffs = [abs(r["arms"][arm]["delta_margin"] - real[(r["index"], arm)]) for r in rows]
        print(f"[b66.consist] Δmargin {arm} block66 vs block62: max|diff| {max(diffs):.4f}")


def main():
    zero9, resc9, e62 = groups_9b()
    energy_block("b62", e62, zero9, 4096)
    rows63 = load("block63_why_plane_ignored/why_plane_ignored.jsonl")
    real9 = {(r["index"], r["arm"]): r["delta_margin"] for r in e62}
    grad_block("b63", rows63, zero9, resc9, real9, 256 / 4096, "L23_h8_to_bridge", "L23_sum_to_bridge", ("clean", "clamp2d@4", "orth@0.1", "orth@0.25", "full@0.25"), "orth@0.1")
    prop_block("b64", load("block64_propagation/propagation.jsonl"), zero9)
    zero4, resc4, e65 = groups_4b()
    energy_block("b65", e65, zero4, 2560)
    kv_block(zero9, resc9, e62)
    rows67 = load("block67_4b_why_plane_ignored/why_plane_ignored.jsonl")
    real4 = {(r["index"], r["arm"]): r["delta_margin"] for r in e65}
    head_search_4b(rows67)
    grad_block("b67", rows67, zero4, resc4, real4, 256 / 2560, "L23_h8_to_bridge", "L23_sum_to_bridge", ("clean", "clamp2d@4", "orth@0.1", "orth@0.25", "full@0.25"), "orth@0.25")
    prop_block("b68", load("block68_4b_propagation/propagation.jsonl"), zero4)


if __name__ == "__main__":
    main()
