#!/usr/bin/env python3
"""Recompute audit 6: log block 80 (4B donor-free recipe), the two block-79 zero-ablation paragraphs, the block-76
AUC-CI supplement, the timestamp-errata table (vs local rsync-preserved mtimes), the numbers those parts contributed
to rows C15 / C19 / C20 and H-1 / H-8 of the numbers list, and a log<->pack consistency sweep for blocks 75-80.

Independent recomputation in python3 + numpy.  Only the paired-difference CIs use the project's
rlens.analysis helpers (cluster_map, fact_pairs, paired_difference, load_records) as the task prescribes; an
independent vectorised cluster bootstrap is run alongside them (seeds 0/1/2) to report Monte-Carlo drift.
Run from code/:  python3 audit_recompute/audit6_recompute.py [--seed N] [--nboot N]
Reads only raw files under 正式研究同步/results/, code/data/new_items_nongeo_v2.json, and the two markdown files
(for the consistency sweep).  Writes nothing.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
from rlens.analysis import cluster_map, fact_pairs, load_records, paired_difference  # noqa: E402

ROOT = CODE.parent
RES = ROOT.parent / "正式研究同步" / "results"
V2 = CODE / "data" / "new_items_nongeo_v2.json"
LOG = ROOT / "00_研究日志_运行中.md"
PACK = ROOT / "01_数字与图表清单_供写作.md"

SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
N_BOOT = int(sys.argv[sys.argv.index("--nboot") + 1]) if "--nboot" in sys.argv else 10000
N_BOOT_AUC = 5000
TOL_PAIRED_CI = 0.15
TOL_AUC_CI = 0.05

CHECKS: list[dict] = []
MISMATCH: list[dict] = []
N_NUMBERS = 0


# ----------------------------------------------------------------------------- comparison helpers
def _parse(s: str):
    """Return (value, n_decimals) for a logged number string; percentages become fractions with 2 extra dp."""
    t = s.replace("−", "-").replace("**", "").replace("+", "").strip()
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    nd = len(t.split(".")[1]) if "." in t else 0
    v = float(t)
    return (v / 100.0, nd + 2) if pct else (v, nd)


def _rounded_candidates(x: float, nd: int) -> set:
    q = Decimal(1).scaleb(-nd)
    half_up = float(Decimal(f"{x:.12f}").quantize(q, rounding=ROUND_HALF_UP))
    return {half_up, float(f"{x:.{nd}f}")}


def agrees(logged: str, rec, tol=None) -> bool:
    if isinstance(rec, str):
        return logged.strip() == rec.strip()
    if rec is None or (isinstance(rec, float) and np.isnan(rec)):
        return False
    v, nd = _parse(logged)
    if tol is not None:
        return abs(float(rec) - v) <= tol + 1e-12
    return any(abs(c - v) < 1e-9 for c in _rounded_candidates(float(rec), nd))


def fmt(x) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.4f}"


def chk(cid: str, claim: str, logged: list, rec: list, tols=None, note: str = "", unverifiable=()) -> None:
    """One table row; `tols` is None or a list parallel to `logged` (None = exact at logged precision).
    Positions listed in `unverifiable` have no local artefact: reported as UNVERIFIABLE, not as a mismatch."""
    global N_NUMBERS
    tols = tols or [None] * len(logged)
    assert len(logged) == len(rec) == len(tols), cid
    verdicts = [agrees(l, r, t) or k in unverifiable for k, (l, r, t) in enumerate(zip(logged, rec, tols))]
    ok = all(verdicts)
    N_NUMBERS += len(logged)
    row = {"id": cid, "claim": claim, "logged": logged, "rec": rec, "ok": ok, "verdicts": verdicts, "note": note,
           "unverifiable": [logged[k] for k in unverifiable]}
    CHECKS.append(row)
    if not ok:
        MISMATCH.append(row)


def section(title: str) -> None:
    CHECKS.append({"section": title})


def med(xs) -> float:
    xs = list(xs)
    return float(np.median(xs)) if xs else float("nan")


def rate(xs) -> float:
    xs = [bool(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


# ----------------------------------------------------------------------------- independent bootstrap
def own_cluster_ci(values: dict, clusters: dict, n_boot: int, seed: int):
    """Vectorised cluster bootstrap of the mean (independent of rlens.analysis.bootstrap_mean)."""
    groups: dict = {}
    for i, v in values.items():
        groups.setdefault(clusters.get(i, i), []).append(v)
    keys = list(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), (n_boot, len(keys)))
    means = sums[picks].sum(1) / cnts[picks].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(keys)


def auc(score, label) -> float:
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return float(cmp.mean())


def auc_cluster_ci(score: dict, label: dict, clusters: dict, n_boot: int, seed: int):
    """Cluster bootstrap of the AUC: resample clusters with replacement, pool items, percentile 2.5/97.5.
    Draws with a single class are skipped (nan)."""
    groups: dict = {}
    for i in score:
        groups.setdefault(clusters[i], []).append(i)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        items = [i for k in picked for i in groups[keys[k]]]
        out[b] = auc([score[i] for i in items], [label[i] for i in items])
    valid = out[~np.isnan(out)]
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)), int(np.isnan(out).sum())


# ============================================================================= data
audit9b = {r["index"]: r for r in json.load(open(RES / "block01" / "q01_data_audit_rows.json"))}
b80 = load_records(RES / "block80_4b_donor_free_recipe" / "donor_free.jsonl")
rep4b = load_records(RES / "block27_4b_band8_20" / "replication.jsonl")
ZERO4B = {r["index"] for r in rep4b if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp"
          and r["control"] == "full" and not r["top1_is_swap"]}
rep4b_fb = load_records(RES / "block27_4b" / "replication.jsonl")
ZERO4B_FB = {r["index"] for r in rep4b_fb if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp"
             and r["control"] == "full" and not r["top1_is_swap"]}
b75 = load_records(RES / "block75_donor_free_recipe" / "donor_free.jsonl")
sup = load_records(RES / "block12_suppression" / "suppression.jsonl")
ZERO9B = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
b76 = load_records(RES / "block76_oos_recipe" / "oos_recipe.jsonl")
v2_items = json.load(open(V2))["items"]
V2_INDEX = {it["name"]: i for i, it in enumerate(v2_items)}
for r in b76:
    r["index"] = V2_INDEX[r["name"]]


def arm_vals(rows, arm, field="delta_margin") -> dict:
    return {r["index"]: float(r["arms"][arm][field]) for r in rows if arm in r["arms"]}


def paired(rows, a, b, clusters):
    A, B = arm_vals(rows, a), arm_vals(rows, b)
    d = paired_difference(A, B, clusters, n_boot=N_BOOT, seed=SEED)
    shared = sorted(set(A) & set(B))
    diff = {i: A[i] - B[i] for i in shared}
    own = {s: own_cluster_ci(diff, clusters, N_BOOT, s) for s in (0, 1, 2)}
    return d, own


# ============================================================================= (a) block 80
section("80. 第 80 块：4B 无捐赠者配方 (block80_4b_donor_free_recipe/donor_free.jsonl)")
ALL47 = sorted(r["index"] for r in b80)
FLIP16 = [r for r in b80 if r["index"] not in ZERO4B]
ZERO31 = [r for r in b80 if r["index"] in ZERO4B]
G80 = {"flip": FLIP16, "zero": ZERO31, "all": b80}
chk("80.0", "47 条（可翻转 16、零翻转 31）; p_h8 == p_best 62%",
    ["47", "16", "31", "62%"],
    [len(b80), len(FLIP16), len(ZERO31), rate(r["p_h8"] == r["p_best"] for r in b80)],
    note=f"zero set from block27_4b_band8_20 (width 13, 52 items, {len(ZERO4B)} not flipped); the fallback "
         f"block27_4b (width 6) would give {len(ALL47) - len(set(ALL47) & ZERO4B_FB)}/{len(set(ALL47) & ZERO4B_FB)}")

CL_FLIP = cluster_map([r["index"] for r in FLIP16], fact_pairs({r["index"]: audit9b[r["index"]] for r in FLIP16}))
CL_ALL = cluster_map(ALL47, fact_pairs({i: audit9b[i] for i in ALL47}))
N_CL = {"flip": len(set(CL_FLIP.values())), "all": len(set(CL_ALL.values()))}

ROWS80 = [
    ("钳位 ×4", "clamp@4_best", ["0.88", "+9.8", "0.29", "0.49"], None, None),
    ("强制 h8 + 钳位 ×4", "F[h8@23,1]+clamp@4_best", ["0.94", "+13.7", "0.35", "0.55"],
     ["+2.28", "+1.24", "+3.92"], ["+1.15", "+0.67", "+1.80"]),
    ("强制 h8+h9+h0 + 钳位", "F[h8h9h0@23,1]+clamp@4_best", ["1.00", "+14.8", "0.39", "0.60"], None, None),
    ("强制 L23 全部头 + 钳位", "F[all@23,1]+clamp@4_best", ["0.94", "+14.3", "0.39", "0.57"],
     ["+2.59", "+1.46", "+4.39"], ["+1.53", "+0.92", "+2.34"]),
    ("捐赠者 key + 钳位", "K<-donor+clamp@4_best", ["1.00", "+14.3", "0.39", "0.60"],
     ["+2.36", "+1.24", "+4.19"], ["+1.36", "+0.78", "+2.15"]),
    ("钳位 ×4 @全部位置", "clamp@4_all", ["0.94", "+14.0", "0.32", "0.53"],
     ["+3.08", "+1.04", "+5.84"], ["+2.17", "+0.83", "+3.70"]),
]
DRIFT: list[float] = []
PAIRED80: dict = {}
for k, (label, arm, cells, ci_flip, ci_all) in enumerate(ROWS80, 1):
    a_flip = [r["arms"][arm] for r in FLIP16]
    a_zero = [r["arms"][arm] for r in ZERO31]
    a_all = [r["arms"][arm] for r in b80]
    chk(f"80.{k}a", f"{label}: 可翻转 16 翻转 / ΔM 中位; 零翻转 31 翻转; 全体 47 翻转", cells,
        [rate(q["top1_is_swap"] for q in a_flip), med(q["delta_margin"] for q in a_flip),
         rate(q["top1_is_swap"] for q in a_zero), rate(q["top1_is_swap"] for q in a_all)])
    if ci_flip:
        for g, ci, cl in (("flip", ci_flip, CL_FLIP), ("all", ci_all, CL_ALL)):
            d, own = paired(G80[g], arm, "clamp@4_best", cl)
            PAIRED80[(arm, g)] = d
            DRIFT.extend(abs(own[s][0] - d["lo95"]) for s in own)
            DRIFT.extend(abs(own[s][1] - d["hi95"]) for s in own)
            chk(f"80.{k}{'b' if g == 'flip' else 'c'}",
                f"{label} − 钳位 ×4 @p_best, {'可翻转 16' if g == 'flip' else '全体 47'}: mean [lo, hi]",
                ci, [d["mean_diff"], d["lo95"], d["hi95"]], tols=[None, TOL_PAIRED_CI, TOL_PAIRED_CI],
                note=f"n={d['n']} clusters={N_CL[g]} win {d['win_rate']:.2f}; own bootstrap seeds 0/1/2: "
                     + " ; ".join(f"[{own[s][0]:+.2f}, {own[s][1]:+.2f}]" for s in own))

fo = "F[h8@23,1]_best"
chk("80.7", "只强制 h8: 可翻转翻转 0.00（原答案 1.00）; 零翻转 0.00; 全体 0.00",
    ["0.00", "1.00", "0.00", "0.00"],
    [rate(r["arms"][fo]["top1_is_swap"] for r in FLIP16), rate(r["arms"][fo]["top1_is_answer"] for r in FLIP16),
     rate(r["arms"][fo]["top1_is_swap"] for r in ZERO31), rate(r["arms"][fo]["top1_is_swap"] for r in b80)],
    note=f"top1_is_answer all 47 = {rate(r['arms'][fo]['top1_is_answer'] for r in b80):.3f}; "
         f"F[h8@23,1]_h8 flips {rate(r['arms']['F[h8@23,1]_h8']['top1_is_swap'] for r in b80):.2f}")

ALL59 = sorted(r["index"] for r in b75)
ZERO35 = [r for r in b75 if r["index"] in ZERO9B]
z9b_all = rate(r["arms"]["clamp@4_all"]["top1_is_swap"] for r in ZERO35)
chk("80.8", "读法: 强制 h8 版 +2.3 对捐赠者 key 版 +2.4（可翻转配对均值）; 4B 零翻转组全位置钳位 0.32 vs 9B 0.46",
    ["+2.3", "+2.4", "0.32", "0.46"],
    [PAIRED80[("F[h8@23,1]+clamp@4_best", "flip")]["mean_diff"], PAIRED80[("K<-donor+clamp@4_best", "flip")]["mean_diff"],
     rate(r["arms"]["clamp@4_all"]["top1_is_swap"] for r in ZERO31), z9b_all],
    note=f"9B zero-flip group = {len(ZERO35)} items (block12_suppression, J clamp not flipped) under block75 clamp@4_all")

# ============================================================================= (b) block 79 zero ablations
section("79z. 第 79 块补充：零消融 9B / 4B (block79_*clean_head_ablation_zero/head_ablation.jsonl)")


def ablation_stats(rows):
    out = {}
    for kind in ("two_hop", "single_hop"):
        for arm in sorted({r["arm"] for r in rows}):
            sel = [r for r in rows if r["prompt_kind"] == kind and r["arm"] == arm and r["clean_correct"]]
            out[(kind, arm)] = (rate(r["still_correct"] for r in sel), med(r["delta_logp_answer"] for r in sel), len(sel))
    return out


for tag, rel, logged in (
    ("9B", "block79_clean_head_ablation_zero", {"h8": ("0.93", "−0.03"), "h8h9h0": ("0.95", None), "all@23": ("0.90", "−0.10"),
                                                 "all@19": ("0.92", "−0.11"), "rand3@23": ("1.00", None), "single": ("0.96", "1.00")}),
    ("4B", "block79_4b_clean_head_ablation_zero", {"h8": ("0.98", None), "h8h9h0": ("0.94", None), "all@23": ("0.94", "−0.10"),
                                                    "all@19": ("0.94", "−0.20"), "rand3@23": ("0.98", None), "single": ("0.98", "1.00")}),
):
    rows = load_records(RES / rel / "head_ablation.jsonl")
    st = ablation_stats(rows)
    n2 = st[("two_hop", "h8")][2]
    n1 = st[("single_hop", "h8")][2]
    for arm in ("h8", "h8h9h0", "all@23", "all@19", "rand3@23"):
        acc, dlp, _ = st[("two_hop", arm)]
        lg = [logged[arm][0]] + ([logged[arm][1]] if logged[arm][1] else [])
        rc = [acc] + ([dlp] if logged[arm][1] else [])
        chk(f"79z.{tag}.{arm}", f"{tag} zero-ablation {arm}: two-hop accuracy" + (" (Δlog p median)" if logged[arm][1] else ""),
            lg, rc, note=f"n clean-correct two-hop = {n2}" + ("" if logged[arm][1] else f"; Δlog p median {dlp:+.3f}"))
    single = [st[("single_hop", a)][0] for a in ("h8", "h9", "h0", "h8h9h0", "rand3@23", "all@23", "all@19")]
    chk(f"79z.{tag}.single", f"{tag} zero-ablation single-hop accuracy range over 7 arms (min–max)",
        list(logged["single"]), [min(single), max(single)],
        note=f"n clean-correct single-hop = {n1}; per arm h8/h9/h0/h8h9h0/rand3/all23/all19 = "
             + "/".join(f"{x:.2f}" for x in single) + f"; two-hop h9/h0 = {st[('two_hop', 'h9')][0]:.2f}/{st[('two_hop', 'h0')][0]:.2f}")

# supplementary: block-79 mean ablations (the C15 numbers next to the zero-ablation ones)
section("79m. 第 79 块（均值消融, 供 C15 交叉核对；审计 5 已覆盖）")
for tag, rel, logged in (
    ("9B", "block79_clean_head_ablation", {"h8": "0.97", "h8h9h0": "0.95", "all@23": "0.93", "all@19": "0.95"}),
    ("4B", "block79_4b_clean_head_ablation", {"h8": "0.98", "h8h9h0": "0.98", "all@23": "0.94", "all@19": "0.91"}),
):
    st = ablation_stats(load_records(RES / rel / "head_ablation.jsonl"))
    chk(f"79m.{tag}", f"{tag} mean-ablation two-hop accuracy h8 / h8h9h0 / all@23 / all@19",
        list(logged.values()), [st[("two_hop", a)][0] for a in logged],
        note="single-hop min over arms = %.3f" % min(st[("single_hop", a)][0] for a in logged))

# ============================================================================= (c) block 76 AUC CIs
section("76s. 第 76 块补充：样本外 AUC 的事实级聚类 bootstrap 95% CI (block76_oos_recipe/oos_recipe.jsonl)")
IDX76 = sorted(r["index"] for r in b76)
CL76 = cluster_map(IDX76, fact_pairs({i: v2_items[i] for i in IDX76}))
BY76 = {r["index"]: r for r in b76}


def feats(site: str) -> dict:
    return {"h8": {i: BY76[i]["h8_mass"][site] for i in IDX76},
            "gap": {i: BY76[i]["gap"][site] for i in IDX76},
            "gap×h8": {i: BY76[i]["gap"][site] * BY76[i]["h8_mass"][site] for i in IDX76},
            "cos": {i: BY76[i]["cos"] for i in IDX76}}


AUC_LOG = {
    ("clamp@4_best", "best"): {"n_pos": "9", "h8": ("0.88", "0.75", "0.98"), "gap": ("0.80", "0.65", "0.95"),
                               "gap×h8": ("0.90", "0.78", "1.00"), "cos": ("0.54", "0.33", "0.72")},
    ("clamp@4_h8", "h8"): {"n_pos": "10", "h8": ("0.92", "0.79", "1.00"), "cos": ("0.45", "0.23", "0.69")},
    ("clamp@1_all", "h8"): {"n_pos": "8", "gap": ("0.74", "0.42", "1.00"), "h8": ("0.61", "0.37", "0.81")},
}
AUC_DRIFT: list[float] = []
AUC_RES: dict = {}
for (target, site), lg in AUC_LOG.items():
    label = {i: bool(BY76[i]["arms"][target]["top1_is_swap"]) for i in IDX76}
    chk(f"76s.{target}.npos", f"positives for target {target}", [lg["n_pos"]], [sum(label.values())],
        note=f"n={len(IDX76)}, clusters={len(set(CL76.values()))}")
    F = feats(site)
    for name, (a0, lo0, hi0) in ((k, v) for k, v in lg.items() if k != "n_pos"):
        point = auc([F[name][i] for i in IDX76], [label[i] for i in IDX76])
        cis = {s: auc_cluster_ci(F[name], label, CL76, N_BOOT_AUC, s) for s in (SEED, 1, 2)}
        lo, hi, nskip = cis[SEED]
        AUC_RES[(target, name)] = (point, lo, hi)
        AUC_DRIFT.extend(abs(cis[s][0] - lo) for s in cis)
        AUC_DRIFT.extend(abs(cis[s][1] - hi) for s in cis)
        chk(f"76s.{target}.{name}", f"AUC({name} @{site} site → {target}) [lo, hi]", [a0, lo0, hi0], [point, lo, hi],
            tols=[None, TOL_AUC_CI, TOL_AUC_CI],
            note=f"{N_BOOT_AUC} draws seed {SEED}, single-class draws skipped={nskip}; seeds 1/2: "
                 + " ; ".join(f"[{cis[s][0]:.2f}, {cis[s][1]:.2f}]" for s in (1, 2)))
# alternate site for the all-positions target (not logged; informational)
label_all = {i: bool(BY76[i]["arms"]["clamp@1_all"]["top1_is_swap"]) for i in IDX76}
Fb = feats("best")
ALT_ALL = {k: auc([Fb[k][i] for i in IDX76], [label_all[i] for i in IDX76]) for k in ("gap", "h8")}
chk("76s.reading", "读法: h8 AUC CIs for single-site targets exclude 0.5; cos CIs include 0.5 (count of 2 / 2)",
    ["2", "2"],
    [sum(AUC_RES[(t, "h8")][1] > 0.5 for t in ("clamp@4_best", "clamp@4_h8")),
     sum(AUC_RES[(t, "cos")][1] <= 0.5 <= AUC_RES[(t, "cos")][2] for t in ("clamp@4_best", "clamp@4_h8"))],
    note=f"clamp@1_all with p_best-site features (alternate, unlogged): gap {ALT_ALL['gap']:.2f}, h8 {ALT_ALL['h8']:.2f}")

# ============================================================================= (d) timestamps
section("T. 时间戳勘误表 vs 本机 rsync 保留的 mtime（block 目录内最新文件）")
ERRATA = [
    ("62", "block62_energy_of_arms", "10:00"), ("63", "block63_why_plane_ignored", "14:58"),
    ("64", "block64_propagation", "15:01"), ("65", "block65_4b_energy_of_arms", "15:04"),
    ("66", "block66_key_value", "15:09"), ("67", "block67_4b_why_plane_ignored", "15:11"),
    ("68", "block68_4b_propagation", "15:13"), ("69", "block69_position_choice", "16:15"),
    ("70", "block70_4b_key_value", "16:24"), ("71", "block71_necessity_ladder", "16:20"),
    ("72", "block72_subspace_controls", "16:29"), ("73", "block73_routing_all_layers", "16:45"),
    ("74 9B", "block74_random_token_planes", "16:46"), ("74 4B", "block74_4b_random_token_planes", "16:48"),
    ("75", "block75_donor_free_recipe", "18:25"), ("76", "block76_oos_recipe", "18:28"),
    ("77", "block77_growth_pca", "18:32"), ("78", "block78_early_band_routing", "18:37"),
    ("79 9B", "block79_clean_head_ablation", "18:39"), ("79 4B", "block79_4b_clean_head_ablation", "18:42"),
    ("80", "block80_4b_donor_free_recipe", "18:48"),
    ("79 零消融 9B", "block79_clean_head_ablation_zero", "23:00"), ("79 零消融 4B", "block79_4b_clean_head_ablation_zero", "23:03"),
]
# header times as they stood before the errata (read from the log at 23:2x on 9/3; those now overwritten kept as 原标)
ORIG_EST = {"63": "09-03 18:10", "64": "09-03 18:10", "66": "09-03 19:20", "67": "09-03 19:40", "68": "09-03 20:00",
            "69": "09-03 21:10", "70": "09-03 22:00", "71": "09-03 21:40", "72": "09-03 22:20", "73": "09-03 23:10",
            "74 9B": "09-03 23:25", "74 4B": "09-03 23:35", "75": "09-04 01:20", "76": "09-04 01:35", "77": "09-04 01:55",
            "78": "09-04 02:20", "79 9B": "09-04 02:40", "79 4B": "09-04 02:55", "80": "09-04 03:15",
            "79 零消融 9B": "09-04 04:40", "79 零消融 4B": "09-04 05:05"}
MTIMES: dict = {}
OFFSETS: dict = {}
for blk, d, logged in ERRATA:
    files = [os.path.join(r, f) for r, _, fs in os.walk(RES / d) for f in fs]
    newest = max(files, key=os.path.getmtime)
    t = dt.datetime.fromtimestamp(os.path.getmtime(newest))
    MTIMES[blk] = t
    floor = t.strftime("%H:%M")
    rounded = (t + dt.timedelta(seconds=30)).strftime("%H:%M")
    rec = floor if logged == floor else (rounded if logged == rounded else floor)
    note = f"{os.path.basename(newest)} mtime {t.strftime('%Y-%m-%d %H:%M:%S')}"
    if blk in ORIG_EST:
        est = dt.datetime.strptime("2026-" + ORIG_EST[blk], "%Y-%m-%d %H:%M")
        OFFSETS[blk] = (est - t).total_seconds() / 3600
        note += f"; header estimate {ORIG_EST[blk]} was {OFFSETS[blk]:+.1f} h late"
    if floor != rounded and logged == floor:
        note += " (floor of HH:MM:SS; rounds to " + rounded + ")"
    chk(f"T.{blk}", f"block {blk} 真实完成时间", [logged], [rec], note=note)
chk("T.81", "81–82 Qwen3-4B 拟合 23:18 启动", ["23:18"], ["(no block81 dir in local results)"], unverifiable=(0,),
    note="no local file; also H-8 '10.3 分钟完成' has no local artefact")
lo_off, hi_off = min(OFFSETS.values()), max(OFFSETS.values())
chk("T.offset", "勘误正文: 估算偏晚了 5–7 小时 (range of header-estimate − mtime over blocks 63–80 + 零消融)",
    ["5", "7"], [lo_off, hi_off], tols=[0.5, 0.5],
    note="offsets (h): " + ", ".join(f"{k} {v:+.1f}" for k, v in OFFSETS.items()))
tz = dt.datetime.now().astimezone().strftime("%Z %z")
chk("T.tz", "本机时区 = 北京时间 (UTC+8)", ["+0800"], [tz.split()[-1]], note=f"local zone {tz}")

# ============================================================================= (e) numbers-list rows
section("P. 清单 C15 / C19 / C20 / H-1 / H-8 中来自这些部分的数字")
CL59 = cluster_map(ALL59, fact_pairs({i: audit9b[i] for i in ALL59}))
FLIP24 = [r for r in b75 if r["index"] not in ZERO9B]
CL24 = cluster_map([r["index"] for r in FLIP24], fact_pairs({r["index"]: audit9b[r["index"]] for r in FLIP24}))
d75, _ = paired(FLIP24, "F[h8@23,1]+clamp@4_best", "clamp@4_best", CL24)
d76h8, _ = paired(b76, "F[h8@23,1]+clamp@4_best", "clamp@4_best", CL76)
d76all, _ = paired(b76, "F[all@23,1]+clamp@4_best", "clamp@4_best", CL76)
fl80 = [rate(r["arms"][a]["top1_is_swap"] for r in FLIP16) for a in
        ("F[h8@23,1]+clamp@4_best", "F[h8h9h0@23,1]+clamp@4_best", "F[all@23,1]+clamp@4_best", "K<-donor+clamp@4_best")]
chk("P.H1", "H-1: block75 可翻转 0.79 → 0.96, +1.66 [+0.88, +2.71]; block76 +0.72 / +1.38; block80 0.88 → 0.94–1.00, +2.28 [+1.24, +3.92]",
    ["0.79", "0.96", "+1.66", "+0.88", "+2.71", "+0.72", "+1.38", "0.88", "0.94", "1.00", "+2.28", "+1.24", "+3.92"],
    [rate(r["arms"]["clamp@4_best"]["top1_is_swap"] for r in FLIP24), rate(r["arms"]["F[h8@23,1]+clamp@4_best"]["top1_is_swap"] for r in FLIP24),
     d75["mean_diff"], d75["lo95"], d75["hi95"], d76h8["mean_diff"], d76all["mean_diff"],
     rate(r["arms"]["clamp@4_best"]["top1_is_swap"] for r in FLIP16), min(fl80), max(fl80),
     PAIRED80[("F[h8@23,1]+clamp@4_best", "flip")]["mean_diff"], PAIRED80[("F[h8@23,1]+clamp@4_best", "flip")]["lo95"],
     PAIRED80[("F[h8@23,1]+clamp@4_best", "flip")]["hi95"]],
    tols=[None, None, None, TOL_PAIRED_CI, TOL_PAIRED_CI, None, None, None, None, None, None, TOL_PAIRED_CI, TOL_PAIRED_CI],
    note=f"block76 CIs: h8 [{d76h8['lo95']:+.2f}, {d76h8['hi95']:+.2f}], all [{d76all['lo95']:+.2f}, {d76all['hi95']:+.2f}]")
chk("P.C15", "C15 零消融: 9B h8 0.93, L23 全部头 0.90, L19 全部头 0.92; 4B 0.98 / 0.94 / 0.94",
    ["0.93", "0.90", "0.92", "0.98", "0.94", "0.94"],
    [ablation_stats(load_records(RES / "block79_clean_head_ablation_zero" / "head_ablation.jsonl"))[("two_hop", a)][0] for a in ("h8", "all@23", "all@19")]
    + [ablation_stats(load_records(RES / "block79_4b_clean_head_ablation_zero" / "head_ablation.jsonl"))[("two_hop", a)][0] for a in ("h8", "all@23", "all@19")])
chk("P.C19", "C19 样本外: h8 AUC 0.88 [0.75, 0.98]–0.92 [0.79, 1.00]; gap×h8 0.90 [0.78, 1.00]; cos 0.45–0.54; 全位置 0.74 对 0.61; 消融后 0.97",
    ["0.88", "0.75", "0.98", "0.92", "0.79", "1.00", "0.90", "0.78", "1.00", "0.45", "0.54", "0.74", "0.61", "0.97"],
    [*AUC_RES[("clamp@4_best", "h8")], *AUC_RES[("clamp@4_h8", "h8")], *AUC_RES[("clamp@4_best", "gap×h8")],
     AUC_RES[("clamp@4_h8", "cos")][0], AUC_RES[("clamp@4_best", "cos")][0], AUC_RES[("clamp@1_all", "gap")][0],
     AUC_RES[("clamp@1_all", "h8")][0],
     ablation_stats(load_records(RES / "block79_clean_head_ablation" / "head_ablation.jsonl"))[("two_hop", "h8")][0]],
    tols=[None, TOL_AUC_CI, TOL_AUC_CI, None, TOL_AUC_CI, TOL_AUC_CI, None, TOL_AUC_CI, TOL_AUC_CI, None, None, None, None, None])
chk("P.C20", "C20: '4B 见 block80' carries no number; block75/76 numbers are in the sweep (S) and audit 5", [], [])
t7580 = max(MTIMES[b] for b in ("75", "76", "77", "78", "79 9B", "79 4B", "80"))
chk("P.H8", "H-8 / G: 第 75–80 块 18:48 完成; 零消融 23:03 完成; 拟合 23:18 启动",
    ["18:48", "23:03", "23:18"],
    [t7580.strftime("%H:%M"), MTIMES["79 零消融 4B"].strftime("%H:%M"), "(no local artefact)"], unverifiable=(2,),
    note="zero-ablation 9B 23:00 < 4B 23:03; '23:18' and '10.3 分钟' have no local artefact")

# ============================================================================= (f) log <-> pack consistency sweep
section("S. 一致性扫描：第 75–80 块的数字在日志与清单中是否一致")
log_lines = LOG.read_text(encoding="utf-8").splitlines()
pack_lines = PACK.read_text(encoding="utf-8").splitlines()


def slice_between(lines, start_pat, end_pat):
    s = next(i for i, l in enumerate(lines) if re.match(start_pat, l))
    e = next((i for i, l in enumerate(lines[s + 1:], s + 1) if re.match(end_pat, l)), len(lines))
    return "\n".join(lines[s:e])


def norm(t: str) -> str:
    return t.replace("−", "-").replace("**", "").replace("＋", "+")


LOG7580 = norm(slice_between(log_lines, r"^## 第 75 块", r"^## 第 81"))
LOG_ERR = norm(slice_between(log_lines, r"^### 时间戳勘误", r"^## 第 (8[3-9]|9\d) 块"))  # to end of file
LOGTXT = LOG7580 + "\n" + LOG_ERR


def pack_row(prefix_pat: str) -> str:
    return norm(next(l for l in pack_lines if re.match(prefix_pat, l)))


PACK_ROWS = {
    "B4": pack_row(r"^\| B4 \|"), "C15": pack_row(r"^\| C15 \|"), "C18": pack_row(r"^\| C18 \|"),
    "C19": pack_row(r"^\| C19 \|"), "C20": pack_row(r"^\| C20 \|"), "C21": pack_row(r"^\| C21 \|"),
    "E": norm(slice_between(pack_lines, r"^## E\.", r"^## F\.")), "G": pack_row(r"^- Toggl"),
    "H1": pack_row(r"^1\. \*\*主线"), "H2": pack_row(r"^2\. \*\*R vs J"), "H3": pack_row(r"^3\. \*\*C18"),
    "H4": pack_row(r"^4\. \*\*C19"), "H8": pack_row(r"^8\. \*\*机器"),
}
# (pack row, pack string, log string or None if identical literal expected, log block)
SWEEP = [
    ("B4", "0.28", None, "78"), ("B4", "0.32", None, "78"), ("B4", "-0.001", None, "78"), ("B4", "0.02–0.07", None, "78"),
    ("B4", "R - J 差 0.00", "KV 移植差 0.00", "78"), ("B4", "+0.38", None, "78"), ("B4", "78%", None, "78"),
    ("C15", "0.97 / 0.95 / 0.93 / 0.95", "| L23 h8 | 0.97 |", "79"), ("C15", "0.97 / 0.95 / 0.93 / 0.95", "| L23 h8+h9+h0 | 0.95 |", "79"),
    ("C15", "0.97 / 0.95 / 0.93 / 0.95", "| L23 全部 16 头 | 0.93 |", "79"), ("C15", "0.97 / 0.95 / 0.93 / 0.95", "| L19 全部头 | 0.95 |", "79"),
    ("C15", "单跳 1.00", "| L23 h8 | 0.97 | -0.01 | 1.00 |", "79"),
    ("C15", "0.98 / 0.98 / 0.94 / 0.91", "0.98、h8+h9+h0 0.98、L23 全部头 0.94、L19 全部头 0.91", "79 4B"),
    ("C15", "h8 0.93", "两跳 0.93", "79z 9B"), ("C15", "L23 全部头 0.90", None, "79z 9B"), ("C15", "L19 全部头 0.92", None, "79z 9B"),
    ("C15", "0.98 / 0.94 / 0.94", "0.98、h8+h9+h0 0.94、L23 全部头 0.94", "79z 4B"),
    ("C18", "0.66 → 0.19", "0.66 → 0.19", "77"), ("C18", "→ 0.31", "→ 0.31", "77"), ("C18", "→ 0.05", "→ 0.05", "77"),
    ("C18", "→ 0.64", "0.64", "77"), ("C18", "lens 0.22 对注意力 0.03", None, "77"),
    ("C18", "4.3–4.6 倍", "约 4.3–4.6 倍", "77"), ("C18", "随机 2.5–2.9 倍", "2.5 倍（逐条目比值中位数 2.9）", "77"),
    ("C19", "0.88 [0.75, 0.98]", None, "76s"), ("C19", "0.92 [0.79, 1.00]", None, "76s"), ("C19", "0.90 [0.78, 1.00]", None, "76s"),
    ("C19", "余弦 0.45–0.54", "0.54 [0.33, 0.72]", "76s"), ("C19", "0.74 对 0.61", "0.74 [0.42, 1.00]，h8 0.61", "76s"),
    ("C19", "消融后 0.97", "L23 h8 | 0.97", "79"),
    ("C20", "0.79 → 0.96", "0.79", "75"), ("C20", "+1.66 [+0.88, +2.71]", None, "75"), ("C20", "胜率 0.96", None, "75"),
    ("C20", "+2.33 [+1.31, +3.58]", None, "75"), ("C20", "原答案保留 0.97", "0.97 保留原答案", "75"),
    ("C20", "+1.01 [+0.59, +1.49]", None, "75"), ("C20", "+0.69 [+0.37, +1.09]", None, "75"), ("C20", "3/27", None, "75"),
    ("C20", "+0.72 [+0.38, +1.09]", None, "76"), ("C20", "+1.38 [+0.94, +1.82]", None, "76"), ("C20", "0.32 → 0.50", None, "76"),
    ("C20", "+1.03 [+0.50, +1.57]", None, "76"),
    ("C21", "0.29 → 0.71", "0.29 | +2.8", "76"), ("C21", "+5.8 [+4.4, +7.1]", "+5.76 [+4.42, +7.06]", "76"),
    ("C21", "21% 改坏", "6/28 = 0.21", "76"), ("C21", "0.64 对 0.39", None, "76"),
    ("E", "≈4.5 倍", "4.5 / 4.6 / 4.3", "77"), ("E", "PCA 0.93", "PCA 0.93", "72/77"),
    ("G", "偏晚 5–7 小时", "偏晚了 5–7 小时", "errata"), ("G", "18:48", None, "errata"), ("G", "23:03", None, "errata"),
    ("G", "23:18", None, "errata"),
    ("H1", "0.79 → 0.96", "0.79", "75"), ("H1", "+1.66 [+0.88, +2.71]", None, "75"), ("H1", "+0.72 / +1.38", "+0.72 [+0.38, +1.09]", "76"),
    ("H1", "0.88 → 0.94–1.00", "0.88 / +9.8", "80"), ("H1", "+2.28 [+1.24, +3.92]", None, "80"),
    ("H2", "-0.001", None, "78"), ("H2", "K/V 移植差 0.00", "KV 移植差 0.00", "78"),
    ("H3", "0.66 → 0.19", None, "77"), ("H3", "→ 0.31", None, "77"), ("H3", "→ 0.05", None, "77"), ("H3", "→ 0.64", "0.64", "77"),
    ("H3", "4.3–4.6 倍", None, "77"), ("H3", "随机 2.5–2.9 倍", "2.5 倍（逐条目比值中位数 2.9）", "77"),
    ("H4", "样本外 0.88–0.92", "0.88 [0.75, 0.98]", "76s"), ("H4", "0.90", None, "76s"), ("H4", "0.29 → 0.71", "0.29 | +2.8", "76"),
    ("H4", "+5.8 [+4.4, +7.1]", "+5.76 [+4.42, +7.06]", "76"), ("H4", "21% 改坏", "6/28 = 0.21", "76"),
    ("H8", "18:48", None, "errata"), ("H8", "23:03", None, "errata"), ("H8", "23:18", None, "errata"),
    ("H8", "10.3 分钟", "(absent from log)", "81"),
]
SWEEP_OUT: list = []
for row, ps, ls, blk in SWEEP:
    in_pack = norm(ps) in PACK_ROWS[row]
    want = norm(ls if ls else ps)
    in_log = want in LOGTXT if not want.startswith("(") else None
    SWEEP_OUT.append((row, ps, ls or "(same literal)", blk, in_pack, in_log))
N_NUMBERS += len(SWEEP)
SWEEP_BAD = [s for s in SWEEP_OUT if not s[4] or s[5] is False]

# ============================================================================= report
print(f"# audit6 recompute — seed {SEED}, paired bootstrap {N_BOOT} draws, AUC bootstrap {N_BOOT_AUC} draws")
print(f"numbers checked: {N_NUMBERS} (per-claim tables {N_NUMBERS - len(SWEEP)} + sweep {len(SWEEP)}); "
      f"table rows with a mismatch: {len(MISMATCH)}; sweep pairs not found in both texts: {len(SWEEP_BAD)}")
print(f"paired-CI max |own-bootstrap bound − rlens bound| over seeds 0/1/2: {max(DRIFT):.3f}; "
      f"AUC-CI max drift across seeds: {max(AUC_DRIFT):.3f}")
print()
for c in CHECKS:
    if "section" in c:
        print(f"\n## {c['section']}\n\n| # | Claim | Logged | Recomputed | n | Verdict |\n|---|---|---|---|---|---|")
        continue
    v = "MATCH" if c["ok"] else "MISMATCH (" + ", ".join(l for l, ok in zip(c["logged"], c["verdicts"]) if not ok) + ")"
    if c["unverifiable"]:
        v += " ; UNVERIFIABLE (" + ", ".join(c["unverifiable"]) + ")"
    print(f"| {c['id']} | {c['claim']}" + (f" — {c['note']}" if c["note"] else "") +
          f" | {' / '.join(c['logged'])} | {' / '.join(fmt(x) for x in c['rec'])} | {len(c['logged'])} | {v} |")
print("\n## S. sweep (pack string → log string; block)\n\n| row | pack | log | block | in pack | in log |\n|---|---|---|---|---|---|")
for row, ps, ls, blk, ip, il in SWEEP_OUT:
    print(f"| {row} | {ps} | {ls} | {blk} | {ip} | {'n/a' if il is None else il} |")
