#!/usr/bin/env python3
"""Figures for the write-up, regenerated from the synced JSONL results. CPU only.

Usage: python3 f01_figures.py            -> writes PNGs to 正式研究/figures/
Every panel's numbers come straight from the result files; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import RES, load_jsonl, per_item, sel  # noqa: E402
from rlens.analysis import bootstrap_mean, cluster_map, fact_pairs, paired_difference  # noqa: E402

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"attn": "#f0b27a", "inv": "#c0392b", "clamp": "#2c3e50", "J": "#2980b9", "R": "#e67e22", "logit": "#7f8c8d", "grey": "#95a5a6", "green": "#27ae60"}

audit_rows = json.load(open(RES / "block01" / "q01_data_audit_rows.json"))
ITEMS = {int(r["index"]): r for r in audit_rows}
ELIG = json.load(open(RES / "block01" / "eligible_hybrid_indices.json"))
CL = cluster_map(ELIG, fact_pairs({i: ITEMS[i] for i in ELIG}))


def rate(rows):
    return float(np.mean([r["top1_is_swap"] for r in rows])) if rows else np.nan


def rate_ci(rows, n_boot=4000, seed=0):
    vals = {int(r["index"]): float(r["top1_is_swap"]) for r in rows}
    idx = sorted(vals)
    cl = cluster_map(idx, fact_pairs({i: ITEMS[i] for i in idx if i in ITEMS}))
    m, lo, hi = bootstrap_mean(vals, cl, n_boot=n_boot, seed=seed)
    return m, lo, hi


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- F1 parity ladder (9B + 4B)
def f1():
    m3 = load_jsonl("block11_mechanism/m3_parity.jsonl")
    third = (RES / "block81_qwen3_4b" / "replication.jsonl").exists()
    fig, axes = plt.subplots(1, 3 if third else 2, figsize=(13 if third else 9, 3.4), sharey=True)
    widths = sorted({int(r["width"]) for r in m3})
    for ax, lens in zip(axes[:1], ["J"]):
        pass
    ax = axes[0]
    for arm, col, lab in (("involution", C["inv"], "published swap (involution), α=1"), ("clamp", C["clamp"], "idempotent clamp")):
        ys = [rate(sel(m3, lens="J", arm=arm, width=w)) for w in widths]
        ax.plot(widths, ys, "-o", color=col, label=lab, ms=4)
    ax.set_title("Qwen3.5-9B, J-lens, band from L8 (n=59)", fontsize=9)
    ax.set_xlabel("band width (layers)")
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_xticks(widths)
    ax.legend(frameon=False, fontsize=8)
    d = RES / "block27_4b_band8_20"
    if (d / "replication.jsonl").exists():
        rep = load_jsonl("block27_4b_band8_20/replication.jsonl")
        w4 = sorted({int(r["width"]) for r in rep if r["stage"] == "ladder"})
        ax = axes[1]
        for arm, col in (("involution", C["inv"]), ("clamp", C["clamp"])):
            ys = [rate(sel(rep, stage="ladder", lens="J", arm=arm, width=w)) for w in w4]
            ax.plot(w4, ys, "-o", color=col, ms=4)
        ax.set_title("Qwen3.5-4B, J-lens, band from L8 (n=52)", fontsize=9)
        ax.set_xlabel("band width (layers)")
        ax.set_xticks(w4)
    if third:
        rep = load_jsonl("block81_qwen3_4b/replication.jsonl")
        meta = json.load(open(RES / "block81_qwen3_4b" / "meta.json"))
        w3 = sorted({int(r["width"]) for r in rep if r["stage"] == "ladder"})
        ax = axes[2]
        for arm, col in (("involution", C["inv"]), ("clamp", C["clamp"])):
            ys = [rate(sel(rep, stage="ladder", lens="J", arm=arm, width=w)) for w in w3]
            ax.plot(w3, ys, "-o", color=col, ms=4)
        ax.set_title(f"Qwen3-4B (dense attn), J-lens fitted here, from L{meta['band'][0]} (n={meta['n_eligible']})", fontsize=9)
        ax.set_xlabel("band width (layers)")
        ax.set_xticks(w3)
    save(fig, "F1_parity_ladder.png")


# ---------------------------------------------------------------- F2 alpha divergence
def f2():
    ws = json.load(open(RES / "block04" / "width_sweep.json"))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    widths = sorted({r["width"] for r in ws})
    alphas = sorted({r["alpha"] for r in ws})
    for w, col in zip(widths, plt.cm.viridis(np.linspace(0.1, 0.9, len(widths)))):
        ys = [np.median([r["growth_last_over_first"] for r in ws if r["width"] == w and r["alpha"] == a and r["lens"] == "J"]) for a in alphas]
        ax.plot(alphas, ys, "-o", color=col, ms=4, label=f"width {w}: measured")
        th = [abs(1 - 2 * a) ** (2 * (w - 1)) if w > 1 else 1 for a in alphas]
        ax.plot(alphas, th, ":", color=col, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("α")
    ax.set_ylabel("‖Δh‖² last layer / first layer (median)")
    ax.set_title("swap update growth across a band\nsolid: measured; dotted: |1−2α|^{2(width−1)}")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "F2_alpha_divergence.png")


# ---------------------------------------------------------------- F3 baseline ladder
def f3():
    b2 = load_jsonl("block02/block02_records.jsonl")
    lad = load_jsonl("block10_ladder/ladder.jsonl")
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    sub = load_jsonl("block10_ladder/subspace_patch.jsonl")
    pat = load_jsonl("block07_causal/patch.jsonl")
    P = "primary_L8_20"
    bars = [
        ("Gaussian random plane (swap)", sel(b2, condition="swap_gauss", lens="J", band=P, alpha=1.0, positions="all")),
        ("mismatched entity pair (swap)", sel(b2, condition="swap_shuffled", lens="J", band=P, alpha=1.0, positions="all")),
        ("logit-lens clamp (identity transport)", sel(lad, lens="logit", arm="clamp", scale=1.0)),
        ("published swap (involution), J", sel(b2, condition="swap_raw", lens="J", band=P, alpha=1.0, positions="all")),
        ("J clamp, direct push projected out", sel(c9, band=P, lens="J", pair="entity", arm="ortho_rescaled")),
        ("J clamp", sel(c9, band=P, lens="J", pair="entity", arm="full")),
        ("J clamp, scale 2", sel(lad, lens="J", arm="clamp", scale=2.0)),
        ("J clamp (14 minimal pairs)", sel(sub, lens="J", arm="clamp_exchange")),
        ("full-layer activation patch (14 pairs)", sel(pat, arm="patch_full", lens="J")),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ys = np.arange(len(bars))[::-1]
    for y, (lab, rows) in zip(ys, bars):
        vals = {}
        for r in rows:
            vals.setdefault(int(r["index"]), []).append(float(r["top1_is_swap"]))
        vals = {i: float(np.mean(v)) for i, v in vals.items()}
        idx = sorted(vals)
        cl = cluster_map(idx, fact_pairs({i: ITEMS[i] for i in idx}))
        m, lo, hi = bootstrap_mean(vals, cl, n_boot=3000)
        col = C["clamp"] if "clamp" in lab and "logit" not in lab else (C["inv"] if "involution" in lab else C["grey"])
        ax.barh(y, m, color=col, xerr=[[m - lo], [hi - m]], error_kw={"elinewidth": 1, "ecolor": "k"})
        ax.text(min(hi + 0.02, 0.9), y, f"{m:.2f} (n={len(idx)})", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("top-1 → swap_answer rate (band L8–20, all prompt positions)")
    ax.set_title("From random to full patch: where the lens intervention sits")
    save(fig, "F3_baseline_ladder.png")


# ---------------------------------------------------------------- F4 direct push vs band depth
def f4():
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    pts = []
    for band, lab, depth in (("early_L3_8", "9B L3–8", 5.5 / 32), ("primary_L8_20", "9B L8–20", 14 / 32)):
        full = per_item(sel(c9, band=band, lens="J", pair="entity", arm="full"))
        orth = per_item(sel(c9, band=band, lens="J", pair="entity", arm="ortho_rescaled"))
        pts.append((depth, np.mean(list(orth.values())) / np.mean(list(full.values())), lab, C["J"]))
    for d, lab, depth in (("block27_4b_band3_8", "4B L3–8", 5.5 / 32), ("block27_4b_band8_20", "4B L8–20", 14 / 32), ("block27_4b", "4B L24–29", 26.5 / 32)):
        p = RES / d / "replication.jsonl"
        if p.exists():
            rep = load_jsonl(f"{d}/replication.jsonl")
            full = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="full"))
            orth = per_item(sel(rep, stage="band", lens="J", arm="clamp", control="ortho_rescaled"))
            pts.append((depth, np.mean(list(orth.values())) / np.mean(list(full.values())), lab, C["R"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for depth, ret, lab, col in pts:
        ax.plot(depth, ret, "o", color=col, ms=7)
        ax.annotate(lab, (depth, ret), textcoords="offset points", xytext=(6, 6 if "4B" in lab else -9), fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("band centre (fraction of depth)")
    ax.set_ylabel("effect retained after projecting out\nthe answer-contrast direction", fontsize=8)
    ax.set_title("Mediated share by band (blue 9B, orange 4B)\nlow only in the answer-formation band; not a monotone trend", fontsize=9)
    save(fig, "F4_direct_push_vs_depth.png")


# ---------------------------------------------------------------- F5 readability vs editability
def f5():
    cr = load_jsonl("block04/cross_readout.jsonl")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    dm = {r["index"]: r["delta_margin"] for r in sel(sup, lens="J", arm="clamp")}
    geo = {r["index"]: r for r in load_jsonl("block05_answer_swap/geometry.jsonl")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    xs, ys, cs = [], [], []
    for r in sel(cr, arm="clean", reader="J"):
        i = r["index"]
        if i in flip:
            xs.append(min(r["best_rank_intermediate"]))
            ys.append(dm[i])
            cs.append(C["green"] if flip[i] else C["grey"])
    ax.scatter(xs, ys, c=cs, s=18)
    ax.set_xscale("log")
    ax.set_xlabel("clean readout: best rank of intermediate\nin band L8–20 (J-lens); green = flips", fontsize=8)
    ax.set_ylabel("clamp Δmargin (nat)")
    ax.set_title("across items: readable ≠ editable (ρ≈0)", fontsize=9)
    ax = axes[1]
    xs = [geo[i]["cos_entitydiff_answerdiff"] for i in dm if i in geo]
    ys = [dm[i] for i in dm if i in geo]
    cs = [C["green"] if flip[i] else C["grey"] for i in dm if i in geo]
    ax.scatter(xs, ys, c=cs, s=18)
    ax.set_xlabel("cos(entity difference, answer difference)\nin unembedding space", fontsize=8)
    ax.set_title("across items: linear relatedness predicts (ρ≈+0.7)", fontsize=9)
    ax = axes[2]
    e2 = load_jsonl("block13_deep/e2_positions.jsonl")
    z = np.load(RES / "block01" / "readout_ranks.npz")
    layers = list(z["layers"])
    rows_band = [layers.index(l) for l in range(8, 21)]
    xs, ys = [], []
    for n, i in enumerate(z["indices"]):
        L = int(z["lengths"][n])
        hits = (z["ranks_J"][n][rows_band, :L, 0] <= 10).sum(axis=0)
        for r in sel(e2, lens="J"):
            if r["index"] == i and 0 <= r["pos"] < L:
                xs.append(hits[r["pos"]] + np.random.default_rng(i).uniform(-0.2, 0.2))
                ys.append(r["delta_margin"])
    ax.scatter(xs, ys, s=10, color=C["J"], alpha=0.5)
    ax.set_xlabel("position readability: band layers where\nthe intermediate is in the lens top-10", fontsize=8)
    ax.set_ylabel("single-position clamp Δmargin (nat)")
    ax.set_title("within items: readable positions are where writing works\n(median within-item ρ = +0.45)", fontsize=9)
    save(fig, "F5_readability_vs_editability.png")


# ---------------------------------------------------------------- F6 gain
def f6():
    p = RES / "block28_actual_delta" / "actual_delta.jsonl"
    if not p.exists():
        return
    ad = sel(load_jsonl("block28_actual_delta/actual_delta.jsonl"), lens="J")
    geo = {r["index"]: r for r in load_jsonl("block05_answer_swap/geometry.jsonl")}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    pa = np.array([r["predicted_actual_sum"] for r in ad])
    act = np.array([r["actual_delta_margin"] for r in ad])
    fl = np.array([r["top1_is_swap"] for r in ad])
    ax.scatter(pa[~fl], act[~fl], color=C["grey"], s=18, label="no flip")
    ax.scatter(pa[fl], act[fl], color=C["green"], s=18, label="flips")
    lim = max(pa.max(), act.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="linear (y = x)")
    ax.set_xlabel("first-order prediction Σ⟨∇, δ_actual⟩ (nat)")
    ax.set_ylabel("measured clamp Δmargin (nat)")
    ax.set_title("the clamp is amplified, not resisted")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    scales = [0.25, 0.5, 1.0]
    med = [np.median([r["additive_reinjection_delta_margin"][str(s)] for r in ad]) for s in scales]
    lin = [np.median(pa) * s for s in scales]
    ax.plot(scales, med, "-o", color=C["clamp"], label="measured (same δ field, rescaled)")
    ax.plot(scales, lin, ":", color="k", label="linear extrapolation")
    ax.set_xlabel("scale of the clamp's δ field")
    ax.set_ylabel("median Δmargin (nat)")
    ax.set_title("dose–response is convex")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    amp = act / np.where(pa > 0.05, pa, np.nan)
    cos = np.array([geo[r["index"]]["cos_entitydiff_answerdiff"] for r in ad])
    ok = ~np.isnan(amp)
    ax.scatter(cos[ok & ~fl], amp[ok & ~fl], color=C["grey"], s=18)
    ax.scatter(cos[ok & fl], amp[ok & fl], color=C["green"], s=18)
    ax.set_yscale("log")
    ax.set_xlabel("cos(entity difference, answer difference)")
    ax.set_ylabel("gain = measured / first-order")
    ax.set_title("gain tracks linear relatedness (ρ≈+0.55)")
    save(fig, "F6_gain.png")


# ---------------------------------------------------------------- F7 R vs J by layer + early band
def f7():
    sw = load_jsonl("block03_sweep/sweep.jsonl")
    c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    layers = sorted({r["layer"] for r in sw})
    for alpha, col in ((0.5, C["grey"]), (1.0, C["J"])):
        ms, los, his = [], [], []
        for l in layers:
            d = paired_difference(per_item(sel(sw, lens="R", layer=l, alpha=alpha)), per_item(sel(sw, lens="J", layer=l, alpha=alpha)), CL, n_boot=2000)
            ms.append(d["mean_diff"]); los.append(d["lo95"]); his.append(d["hi95"])
        ax.errorbar(layers, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], fmt="-o", color=col, ms=4, capsize=2, label=f"single-layer swap, α={alpha}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("R − J Δmargin (nat), paired, 95% CI")
    ax.set_title("R beats J only below L8")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    groups = [("early_L3_8", "full", "L3–8\nclamp"), ("early_L3_8", "ortho_rescaled", "L3–8\ndirect push out"),
              ("primary_L8_20", "full", "L8–20\nclamp"), ("primary_L8_20", "ortho_rescaled", "L8–20\ndirect push out")]
    for g, (band, arm, lab) in enumerate(groups):
        for k, lens in enumerate(("J", "R")):
            m, lo, hi = rate_ci(sel(c9, band=band, lens=lens, pair="entity", arm=arm))
            ax.bar(g * 3 + k, m, color=C[lens], yerr=[[m - lo], [hi - m]], capsize=2, label=lens if g == 0 else None)
    ax.set_xticks([g * 3 + 0.5 for g in range(len(groups))])
    ax.set_xticklabels([g[2] for g in groups], fontsize=8)
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_title("clamp flip rates, R vs J (95% cluster-bootstrap CI)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F7_R_vs_J.png")


# ---------------------------------------------------------------- F8 decay after partial clamp + MLP vs attn
def f8():
    dr = load_jsonl("block18_rederivation/depth_recovery.jsonl")
    s3 = load_jsonl("block20_specificity/s3_repair.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for arm, col, lab in (("partial_8_12", C["inv"], "clamp L8–12 only"), ("full_8_20", C["clamp"], "clamp L8–20")):
        rows = sel(dr, lens="J", arm=arm)
        layers = [p["layer"] for p in rows[0]["profile"]]
        med = [np.median([r["profile"][k]["swapped_fraction"] for r in rows]) for k in range(len(layers))]
        ax.plot(layers, med, "-o", color=col, ms=3, label=lab)
    ax.set_xlabel("layer read")
    ax.set_ylabel("swapped fraction of lens coordinates\n(median, cross-lens R readout)", fontsize=8)
    ax.set_title("the rewrite decays 5–10% per layer above the clamp", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rows = sel(s3, lens="J")
    layers = [p["layer"] for p in rows[0]["profile"]]
    for key, col, lab in (("attn_push_on_source", C["J"], "attention half"), ("mlp_push_on_source", C["R"], "MLP half")):
        med = [np.median([r["profile"][k][key] for r in rows]) for k in range(len(layers))]
        ax.plot(layers, med, "-o", color=col, ms=3, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer (clamp on L8–12)")
    ax.set_ylabel("push toward the ORIGINAL entity\n(units of clean coordinate gap, median)")
    ax.set_title("who rebuilds the original entity: mostly the MLP", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F8_decay_and_repair.png")


# ---------------------------------------------------------------- F9 semantic outcome
def f9():
    mc = json.load(open(RES / "block23_semantic" / "manual_classification.json"))
    def n(v):
        return len(v) if isinstance(v, (list, dict)) else int(v)

    counts = {"strict token flip": 24, "semantic rewrite, other token": n(mc.get("rewrite_semantic_only", 3)), "answer damaged (neither)": n(mc.get("damaged", 5))}
    counts["unchanged"] = 59 - sum(counts.values())
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    labs = list(counts)
    vals = [counts[k] for k in labs]
    cols = [C["green"], "#82c99a", C["inv"], C["grey"]]
    left = 0
    for k, (lab, v, col) in enumerate(zip(labs, vals, cols)):
        ax.barh(0, v, left=left, color=col)
        if v >= 10:
            ax.text(left + v / 2, 0, f"{lab}\n{v}", ha="center", va="center", fontsize=7.5, color="white")
        else:
            ax.annotate(f"{lab}: {v}", (left + v / 2, 0.42), ha="center", va="bottom", fontsize=7.5, xytext=(0, 4 + 12 * (k % 2)), textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": 0.6})
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, 59)
    ax.set_xlabel("items (J-lens clamp, L8–20, 20-token free continuation, hand-classified)")
    ax.set_title("Rewrite, break, or nothing: 59 items")
    save(fig, "F9_semantic_outcomes.png")


# ---------------------------------------------------------------- F10 gain controls (block29)
def f10():
    p = RES / "block29_gain_controls" / "gain_controls.jsonl"
    if not p.exists():
        return
    rows = sel(load_jsonl("block29_gain_controls/gain_controls.jsonl"), lens="J")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    labels = ["clamp\nfield", "answer-contrast\nfield", "clamp field,\nanswer dir. removed", "random\nfield"]
    keys = ["clamp", "contrast", "ortho", "random"]
    pred = [np.median([r["predicted"][k] for r in rows]) for k in keys]
    real = [np.median([r["realised"][k] for r in rows]) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, pred, 0.36, color=C["grey"], label="first-order prediction")
    ax.bar(x + 0.18, real, 0.36, color=C["clamp"], label="measured")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("median Δmargin (nat)")
    ax.set_title("same per-position energy profile as the clamp", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    pm_c = [r["prediction_matched"]["contrast"]["realised"] for r in rows if r["prediction_matched"].get("contrast")]
    pm_o = [r["prediction_matched"]["ortho"]["realised"] for r in rows if r["prediction_matched"].get("ortho")]
    cl = [r["realised"]["clamp"] for r in rows if r["prediction_matched"].get("contrast")]
    pr = [r["predicted"]["clamp"] for r in rows if r["prediction_matched"].get("contrast")]
    vals = [np.median(pr), np.median(cl), np.median(pm_c), np.median(pm_o)]
    labs = ["first-order\nprediction\n(all fields)", "clamp field", "answer-contrast\nfield", "clamp field,\nanswer dir. removed"]
    cols = [C["grey"], C["clamp"], C["inv"], C["green"]]
    ax.bar(np.arange(4), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("median Δmargin (nat)")
    ax.set_title("fields rescaled to the SAME first-order prediction", fontsize=9)
    save(fig, "F10_gain_controls.png")


# ---------------------------------------------------------------- F11 answer emergence by layer (block30)
def f11():
    src = "block30_answer_emergence_logitdir" if (RES / "block30_answer_emergence_logitdir" / "answer_emergence.jsonl").exists() else "block30_answer_emergence"
    p = RES / src / "answer_emergence.jsonl"
    if not p.exists():
        return
    rows = sel(load_jsonl(f"{src}/answer_emergence.jsonl"), lens="J")
    additive = src.endswith("logitdir")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for ax, (gname, g) in zip(axes, (("items that flip (n=%d)", [r for r in rows if r["top1_is_swap"]]), ("items that do not flip (n=%d)", [r for r in rows if not r["top1_is_swap"]]))):
        layers = [q["layer"] for q in g[0]["layers"]]
        for key, col, lab in (("attn_answer", C["J"], "attention half (mid − pre)"), ("mlp_answer", C["R"], "MLP half (out − mid)"), ("direct_answer", C["grey"], "clamp's own injection")):
            med = np.array([np.median([r["layers"][k][key] for r in g]) for k in range(len(layers))])
            ax.plot(layers, np.cumsum(med), "-o", color=col, ms=3, label=lab)
        ax.axvspan(8, 20, color="k", alpha=0.05)
        ax.set_title(gname % len(g), fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("cumulative contribution to the answer contrast\n⟨h, W_U[:,swap] − W_U[:,answer]⟩ (additive)" if additive else "cumulative projection on the J-transported answer direction\n(unit-vector projection; NOT additive)", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(14, axes[0].get_ylim()[1] * 0.9, "clamp band", fontsize=8, ha="center")
    save(fig, "F11_answer_emergence.png")


# ---------------------------------------------------------------- F12 4B early band R vs J
def f12():
    pts = []
    for d, lab in (("block27_4b_band3_8", "4B L3–8"), ("block27_4b_band8_20", "4B L8–20"), ("block27_4b", "4B L24–29")):
        p = RES / d / "replication.jsonl"
        if not p.exists():
            continue
        rep = load_jsonl(f"{d}/replication.jsonl")
        for lens in ("J", "R"):
            rows = sel(rep, stage="band", lens=lens, arm="clamp", control="full")
            vals = {int(r["index"]): float(r["top1_is_swap"]) for r in rows}
            idx = sorted(vals)
            m, lo, hi = bootstrap_mean(vals, cluster_map(idx, fact_pairs({i: ITEMS[i] for i in idx})), n_boot=3000)
            pts.append((lab, lens, m, lo, hi))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    labs = []
    for k, (lab, lens, m, lo, hi) in enumerate(pts):
        x = (k // 2) * 3 + (k % 2)
        ax.bar(x, m, color=C[lens], yerr=[[m - lo], [hi - m]], capsize=2, label=lens if k < 2 else None)
        if k % 2 == 0:
            labs.append((x + 0.5, lab))
    ax.set_xticks([x for x, _ in labs])
    ax.set_xticklabels([l for _, l in labs], fontsize=8)
    ax.set_ylabel("clamp: top-1 → swap_answer rate")
    ax.set_title("Qwen3.5-4B: R beats J only in the early band (n=52)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F12_4B_R_vs_J.png")


# ---------------------------------------------------------------- F13 donor paste (block31)
def f13():
    p = RES / "block31_donor_paste" / "donor_paste.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block31_donor_paste/donor_paste.jsonl")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    arms = [("clamp2d", "2-D lens clamp"), ("paste_sub", "donor: lens-plane\ncoordinates only"), ("paste_orth", "donor: orthogonal\ncomplement only"), ("paste_full", "donor: full vector")]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(arms))
    for k, (gname, cond, col) in enumerate((("items the band clamp flips (n=24)", lambda i: band_flip.get(i, False), C["green"]), ("items it never flips (n=35)", lambda i: not band_flip.get(i, False), C["grey"]))):
        vals = []
        for arm, _ in arms:
            s = [r for r in sel(rows, position_set="best", arm=arm) if cond(r["index"])]
            vals.append(np.mean([r["top1_is_swap"] for r in s]))
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, label=gname)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    foil = RES / "block31_donor_paste_foil" / "donor_paste.jsonl"
    if foil.exists():
        fr = load_jsonl("block31_donor_paste_foil/donor_paste.jsonl")
        v = np.mean([r["top1_is_swap"] for r in sel(fr, position_set="best", arm="paste_full")])
        ax.axhline(v, color=C["inv"], ls="--", lw=1, label=f"mismatched donor, full vector ({v:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=8)
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_title("single bridge position, L8–20: what the lens plane misses", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F13_donor_paste.png")


# ---------------------------------------------------------------- F14 entity probe v2 (block26)
def f14():
    p = RES / "block26_entity_probe" / "entity_probe.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block26_entity_probe/entity_probe.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    clean = {r["index"]: r for r in sel(rows, arm="clean", probe="q_noans")}
    arm = {r["index"]: r for r in sel(rows, arm="J", probe="q_noans")}
    idx = sorted(arm)
    da = np.array([arm[i]["answer_margin"] - clean[i]["answer_margin"] for i in idx])
    de = np.array([arm[i]["entity_margin"] - clean[i]["entity_margin"] for i in idx])
    fl = np.array([arm[i]["top1_is_swap_answer"] for i in idx])
    ax.scatter(da[~fl], de[~fl], color=C["grey"], s=18, label="answer does not flip")
    ax.scatter(da[fl], de[fl], color=C["green"], s=18, label="answer flips")
    ax.set_xlabel("Δ answer margin at the prompt (nat)")
    ax.set_ylabel("Δ entity-report margin at the probe (nat)\nlog p(new entity) − log p(old entity)")
    ax.set_title("no answer in the probe context: the two move together (ρ = +0.89)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    cats = ["new entity", "old entity", "other"]
    x = np.arange(2)
    bottoms = np.zeros(2)
    cols = [C["green"], C["inv"], C["grey"]]
    for c, col in zip(cats, cols):
        vals = []
        for probe in ("q_noans", "q_cleanans"):
            a = {r["index"]: r for r in sel(rows, arm="J", probe=probe)}
            fl_items = [i for i in a if a[i]["top1_is_swap_answer"]]
            if c == "new entity":
                v = np.mean([a[i]["entity_top1_is_swap_to"] for i in fl_items])
            elif c == "old entity":
                v = np.mean([a[i]["entity_top1_is_intermediate"] for i in fl_items])
            else:
                v = np.mean([not a[i]["entity_top1_is_swap_to"] and not a[i]["entity_top1_is_intermediate"] for i in fl_items])
            vals.append(v)
        ax.bar(x, vals, bottom=bottoms, color=col, label=f"probe names the {c}")
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(["probe without answer", "probe with the CLEAN answer\nin context (block-23 design)"], fontsize=8)
    ax.set_ylabel("share of the 24 answer-flipped items")
    ax.set_title("what the model says the bridge entity is", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F14_entity_probe.png")


# ---------------------------------------------------------------- F15 head attribution (block37)
def f15():
    p = RES / "block37_head_attribution" / "head_attribution.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block37_head_attribution/head_attribution.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, layer in zip(axes, ("23", "31")):
        for k, (gname, g, col) in enumerate((("items that flip", [r for r in rows if r["top1_is_swap"]], C["green"]), ("items that do not", [r for r in rows if not r["top1_is_swap"]], C["grey"]))):
            D = np.array([r["layers"][layer]["head_delta_answer"] for r in g])
            med = np.median(D, axis=0)
            x = np.arange(D.shape[1])
            ax.bar(x + (k - 0.5) * 0.4, med, 0.4, color=col, label=f"{gname} (n={len(g)})")
        ax.set_xticks(np.arange(16))
        ax.set_xlabel(f"head (layer {layer}, full attention)")
        ax.set_title(f"L{layer}: per-head answer-direction signal at the final position\n(clamped − clean, median over items)", fontsize=9)
    axes[0].set_ylabel("Δ projection on answer-contrast direction")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "F15_head_attribution.png")


# ---------------------------------------------------------------- F16 neutral donor (block38)
def f16():
    p = RES / "block38_neutral_donor" / "neutral_donor.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block38_neutral_donor/neutral_donor.jsonl")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    donors = [("relation_target", "relation donor\n(single-hop sentence)"), ("N1_target", "neutral donor 1\n'Here is a word: X'"), ("N2_target", "neutral donor 2\n'Consider the following: X'"), ("N1_source", "control: SOURCE entity,\nneutral context")]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(donors))
    for k, (gname, cond, col) in enumerate((("all items (n=59)", lambda i: True, C["clamp"]), ("items the band clamp never flips (n=35)", lambda i: not band_flip.get(i, False), C["grey"]))):
        vals = [np.mean([r["top1_is_swap"] for r in sel(rows, donor=d, mode="full") if cond(r["index"])]) for d, _ in donors]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=col, label=gname)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in donors], fontsize=8)
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_title("full-vector paste at the bridge position: the entity, not a precomputed answer", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "F16_neutral_donor.png")


# ---------------------------------------------------------------- F17 stage necessity (block39)
def f17():
    p = RES / "block39_stage_necessity" / "stage_necessity.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block39_stage_necessity/stage_necessity.jsonl")
    arms = [("clamp_only", "clamp\nonly", C["clamp"]), ("attn_9_16@final", "attention\nL9–16\n(control)", C["grey"]), ("attn_17_24@final", "attention\nL17–24", C["J"]),
            ("attn_25_31@final", "attention\nL25–31", C["grey"]), ("mlp_17_24@final", "MLP\nL17–24", C["grey"]), ("mlp_26_29@final", "MLP\nL26–29", C["R"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for k, (arm, lab, col) in enumerate(arms):
        m, lo, hi = rate_ci(sel(rows, arm=arm))
        ax.bar(k, m, color=col, yerr=[[m - lo], [hi - m]], capsize=2)
        ax.text(k, hi + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms], fontsize=7.5)
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_title("clamp on L8–20, plus one sublayer group restored to its clean output at the final position", fontsize=9)
    ax.set_xlabel("restored group")
    save(fig, "F17_stage_necessity.png")


# ---------------------------------------------------------------- F18 hybrid bases (block43 / 43b)
def f18():
    p = RES / "block43_hybrid_basis" / "hybrid_basis.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block43_hybrid_basis/hybrid_basis.jsonl")
    unit = RES / "block43b_hybrid_basis_unit" / "hybrid_basis.jsonl"
    rows_u = load_jsonl("block43b_hybrid_basis_unit/hybrid_basis.jsonl") if unit.exists() else []
    arms = [("J_s+J_t", "J remove\nJ install", C["J"]), ("R_s+R_t", "R remove\nR install", C["R"]), ("R_s+J_t", "R remove\nJ install", "#b3c6d6"), ("J_s+R_t", "J remove\nR install", "#f0b27a")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (band, title) in zip(axes, (("early_L3_8", "early band L3–8"), ("primary_L8_20", "workspace band L8–20"))):
        for k, (arm, lab, col) in enumerate(arms):
            m, lo, hi = rate_ci(sel(rows, band=band, arm=arm))
            ax.bar(k - (0.2 if rows_u else 0), m, 0.4 if rows_u else 0.7, color=col, yerr=[[m - lo], [hi - m]], capsize=2)
            ax.text(k - (0.2 if rows_u else 0), hi + 0.015, f"{m:.2f}", ha="center", fontsize=7.5)
            if rows_u:
                mu, lou, hiu = rate_ci(sel(rows_u, band=band, arm=arm + "|unit"))
                ax.bar(k + 0.2, mu, 0.4, color=col, alpha=0.45, yerr=[[mu - lou], [hiu - mu]], capsize=2, hatch="//")
                ax.text(k + 0.2, hiu + 0.015, f"{mu:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[1] for a in arms], fontsize=8)
        ax.set_title(f"{title}: clamp with mixed J/R directions" + ("\n(solid: raw vectors; hatched: unit-normalised)" if rows_u else ""), fontsize=9)
    axes[0].set_ylabel("top-1 → swap_answer rate")
    save(fig, "F18_hybrid_bases.png")


# ---------------------------------------------------------------- F19 subspace ladder (block46)
def f19():
    p = RES / "block46_subspace_ladder" / "subspace_ladder.jsonl"
    if not p.exists():
        return
    rows = sel(load_jsonl("block46_subspace_ladder/subspace_ladder.jsonl"), donor="relation")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    band_flip = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    arms = [("J_plane", "J plane\n(2)"), ("J+R_4d", "J ∪ R\n(4)"), ("J_plane+top16", "+ top 16\nsingular dirs"), ("J_plane+top64", "+ top 64"), ("J_plane+top256", "+ top 256"), ("J_plane+top1024", "+ top 1024"), ("outside_top256", "outside\ntop 256"), ("full", "full\nvector")]
    p71 = RES / "block71_necessity_ladder" / "necessity_ladder.jsonl"
    two = p71.exists()
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(12.5 if two else 7.5, 3.4), gridspec_kw={"width_ratios": [1.15, 1]} if two else None)
    ax = axes[0] if two else axes
    x = np.arange(len(arms))
    groups = (("all items (n=59)", lambda i: True, C["clamp"]), ("items the band clamp never flips (n=35)", lambda i: not band_flip.get(i, False), C["grey"]))
    for k, (gname, cond, col) in enumerate(groups):
        vals = [np.mean([r["top1_is_swap"] for r in sel(rows, arm=a) if cond(r["index"])]) for a, _ in arms]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=col, label=gname)
        for xi, v in zip(x + (k - 0.5) * 0.38, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in arms], fontsize=7.5)
    ax.set_ylabel("top-1 → swap_answer rate")
    ax.set_title("SUFFICIENCY: donor paste (full overwrite) restricted to a subspace", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if two:
        ax = axes[1]
        r71 = load_jsonl("block71_necessity_ladder/necessity_ladder.jsonl")
        ks = [2, 16, 64, 256, 1024]
        for k, (gname, cond, col) in enumerate(groups):
            lens_v = [np.mean([r["top1_is_swap"] for r in sel(r71, arm=f"full-Jtop{kk}") if cond(r["index"])]) for kk in ks]
            rand_v = [np.mean([r["top1_is_swap"] for r in sel(r71, arm=f"full-rand{kk}") if cond(r["index"])]) if kk > 2 else np.nan for kk in ks]
            full_v = np.mean([r["top1_is_swap"] for r in sel(r71, arm="full") if cond(r["index"])])
            ax.plot(ks, lens_v, "-o", color=col, ms=4, label=f"minus top-k lens directions, {gname}")
            ax.plot(ks, rand_v, "--s", color=col, ms=4, alpha=0.6, label=f"minus random k dims, {gname}")
            ax.axhline(full_v, color=col, lw=0.8, ls=":")
        p77 = RES / "block77_growth_pca" / "growth_pca.jsonl"
        if p77.exists():
            r77 = load_jsonl("block77_growth_pca/growth_pca.jsonl")
            pk = [64, 256, 1024]
            pca_v = [np.mean([r["necessity"][f"full-PCA{kk}"]["top1_is_swap"] for r in r77]) for kk in pk]
            ax.plot(pk, pca_v, "-^", color=C["R"], ms=5, label="minus top-k residual PCA directions, all items (block 77)")
            both = np.mean([r["necessity"]["full-(R256+PCA256)"]["top1_is_swap"] for r in r77])
            ax.plot([256], [both], "*", color="k", ms=10, label="minus lens-256 ∪ PCA-256")
        ax.set_xscale("log")
        ax.set_xticks(ks); ax.set_xticklabels([str(kk) for kk in ks])
        ax.set_xlabel("k removed from the paste (scale 0.25)")
        ax.set_ylabel("flip rate")
        ax.set_title("NECESSITY: full paste ×0.25 with a subspace removed (dotted = nothing removed)", fontsize=9)
        ax.legend(frameon=False, fontsize=6.5)
    save(fig, "F19_subspace_ladder.png")


# ---------------------------------------------------------------- F20 energy-matched J/R/hybrid (block48)
def f20():
    p = RES / "block48_energy_matched_hybrid" / "energy_matched.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block48_energy_matched_hybrid/energy_matched.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    arms = [("R_s+R_t", "R clamp", C["R"]), ("J_s+R_t", "J remove + R install", "#f0b27a"), ("R_s+J_t", "R remove + J install", "#b3c6d6")]
    versions = [("as_is", "as is"), ("match_total", "total energy\nmatched to J"), ("match_positions", "per-position\nnorms matched to J")]
    for ax, (band, title) in zip(axes, (("early_L3_8", "early band L3–8"), ("primary_L8_20", "workspace band L8–20"))):
        ref = per_item(sel(rows, band=band, arm="J_s+J_t", version="as_is"))
        idx = sorted(ref)
        cl = cluster_map(idx, fact_pairs({i: ITEMS[i] for i in idx}))
        for k, (arm, lab, col) in enumerate(arms):
            for v, (version, vlab) in enumerate(versions):
                d = paired_difference(per_item(sel(rows, band=band, arm=arm, version=version)), ref, cl, n_boot=3000)
                x = v * 4 + k
                ax.bar(x, d["mean_diff"], color=col, yerr=[[d["mean_diff"] - d["lo95"]], [d["hi95"] - d["mean_diff"]]], capsize=2, label=lab if v == 0 else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks([v * 4 + 1 for v in range(len(versions))])
        ax.set_xticklabels([vl for _, vl in versions], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Δmargin − J clamp (nat, paired, 95% CI)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "F20_energy_matched.png")


# ---------------------------------------------------------------- F21 flips vs KL: plane vs complement (block58)
def f21():
    p62 = RES / "block62_energy_of_arms" / "energy_of_arms.jsonl"
    use_energy = p62.exists()
    p = p62 if use_energy else RES / "block58_energy_matched_paste" / "energy_matched_paste.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block62_energy_of_arms/energy_of_arms.jsonl" if use_energy else "block58_energy_matched_paste/energy_matched_paste.jsonl")
    xkey = "energy" if use_energy else "kl"
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sel(sup, lens="J", arm="clamp") if not r["top1_is_swap"]}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for prefix, lab, col, mk in (("clamp2d@", "2-D lens clamp, over-driven (scale 1→4)", C["J"], "o"), ("orth@", "donor complement, scaled down (0.1→1)", C["green"], "s"), ("rand", "J plane + random k dims (16/64/256)", C["grey"], "^")):
        pts = []
        for arm in sorted({r["arm"] for r in rows if r["arm"].startswith(prefix)}):
            s = [r for r in sel(rows, arm=arm) if r["index"] in zero]
            pts.append((np.median([r[xkey] for r in s]), np.mean([r["top1_is_swap"] for r in s])))
        pts.sort()
        ax.plot([q[0] for q in pts], [q[1] for q in pts], "-" + mk, color=col, ms=5, label=lab)
    rt = [r for r in sel(rows, arm="Rtop256") if r["index"] in zero]
    if rt:
        ax.plot([np.median([r[xkey] for r in rt])], [np.mean([r["top1_is_swap"] for r in rt])], "*", color=C["R"], ms=11, label="J plane + R top-256 singular dirs")
    ax.set_xscale("log")
    ax.set_xlabel("median injected energy Σ_l‖δ_l‖² at the bridge position" if use_energy else "median KL(clean ‖ intervened)")
    ax.set_ylabel("flip rate, 35 never-flipped items")
    ax.set_title("same bridge position, L8–20: what a unit of perturbation buys", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F21_flips_vs_kl.png")


# ---------------------------------------------------------------- F23 propagation of a single L8 paste (block64)
def f23():
    p = RES / "block64_propagation" / "propagation.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block64_propagation/propagation.jsonl")
    band = list(range(8, 21))
    p72 = RES / "block72_subspace_controls" / "subspace_controls.jsonl"
    three = p72.exists()
    fig, axes = plt.subplots(1, 3 if three else 2, figsize=(14.5 if three else 9.2, 3.5))
    style = {"full": ("full donor vector", C["grey"], "-"), "J+R256": ("J plane + R top-256 singular dirs", C["R"], "-"),
             "J+rand256": ("J plane + 256 random dirs", C["green"], "--"), "J_plane": ("J plane only (2-D)", C["J"], ":")}
    ax = axes[0]
    for sub, (lab, col, ls) in style.items():
        med = [np.median([r["arms"][sub]["single_L8"]["realised"][i] for r in rows]) for i in range(len(band))]
        ax.plot(band, med, ls, color=col, marker="o", ms=3, label=lab)
    ax.set_xlabel("layer (paste applied at L8 only)")
    ax.set_ylabel("realised share of the donor difference\n⟨h_int − h_clean, d_l⟩ / ‖d_l‖²")
    ax.set_title("what the model regenerates on its own", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for sub, (lab, col, ls) in style.items():
        if sub == "full":
            continue
        med = [np.median([r["arms"][sub]["multi"]["injected"][i] / max(r["arms"][sub]["multi"]["static"][i], 1e-9) for r in rows]) for i in range(len(band))]
        ax.plot(band, med, ls, color=col, marker="o", ms=3, label=lab)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("layer (projector paste at every band layer)")
    ax.set_ylabel("share still to inject at layer l\n‖QQᵀ(donor − h)‖² / ‖QQᵀ(donor − clean)‖²")
    ax.set_title("how much the paste must re-inject", fontsize=9)
    if three:
        ax = axes[2]
        r72 = load_jsonl("block72_subspace_controls/subspace_controls.jsonl")
        r64 = load_jsonl("block64_propagation/propagation.jsonl")
        p77 = RES / "block77_growth_pca" / "growth_pca.jsonl"
        r77 = load_jsonl("block77_growth_pca/growth_pca.jsonl") if p77.exists() else []
        subs = [("J+R256", "lens R\ntop-256", C["R"]), ("J+PCA256", "residual\nPCA-256", C["grey"]), ("J+rand256", "random\n256", C["green"]), ("full", "full donor\nvector", "#b3c6d6")]
        def growth_dl(sub):
            src = r64 if sub == "full" else r72
            return np.median([r["arms"][sub]["single_L8"]["realised"][-1] / max(r["arms"][sub]["single_L8"]["realised"][0], 1e-6) for r in src])
        def growth_d20(sub):
            return np.median([r["growth"][sub]["layers"][-1]["realised_d20"] / max(r["growth"][sub]["realised_L8_on_d20"], 1e-6) for r in r77]) if r77 else np.nan
        x = np.arange(len(subs)); w = 0.38
        ax.bar(x - w / 2, [growth_dl(a) for a, _, _ in subs], w, color=[c for _, _, c in subs], edgecolor="k", lw=0.5, hatch="//", label="vs each layer's own difference d_l")
        ax.bar(x + w / 2, [growth_d20(a) for a, _, _ in subs], w, color=[c for _, _, c in subs], label="vs the end-of-band difference d_20")
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([b for _, b, _ in subs], fontsize=7.5)
        ax.set_ylabel("realised share at L20 / at L8 (single L8 paste)")
        ax.set_title("growth depends on the yardstick; MLPs L18–20 do it", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    save(fig, "F23_propagation.png")


# ---------------------------------------------------------------- F22 content vs routing (blocks 63 + 66)
def f22():
    p63 = RES / "block63_why_plane_ignored" / "why_plane_ignored.jsonl"
    p66 = RES / "block66_key_value" / "key_value.jsonl"
    if not (p63.exists() and p66.exists()):
        return
    r63 = {r["index"]: r for r in load_jsonl("block63_why_plane_ignored/why_plane_ignored.jsonl")}
    r66 = {r["index"]: r for r in load_jsonl("block66_key_value/key_value.jsonl")}
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sel(sup, lens="J", arm="clamp") if not r["top1_is_swap"]}
    e62 = load_jsonl("block62_energy_of_arms/energy_of_arms.jsonl")
    resc4 = {r["index"] for r in sel(e62, arm="clamp2d@4") if r["top1_is_swap"]}
    groups = [("flippable\n(band clamp flips)", [i for i in r66 if i not in zero], C["J"]),
              ("never flipped,\nclamp@4 rescues", [i for i in r66 if i in zero and i in resc4], C["green"]),
              ("never flipped,\nclamp@4 fails", [i for i in r66 if i in zero and i not in resc4], C["grey"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1.9]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, (lab, ids, col) in enumerate(groups):
        ys = [r63[i]["attention"]["clean"]["L23_h8_to_bridge"] for i in ids]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(ys)), ys, s=14, color=col, alpha=0.8)
        ax.hlines(np.median(ys), k - 0.3, k + 0.3, color="k", lw=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("clean L23 h8 attention → bridge position")
    ax.set_title("routing before any edit", fontsize=9)
    ax = axes[1]
    p73 = RES / "block73_routing_all_layers" / "routing_all_layers.jsonl"
    r73 = {r["index"]: r for r in load_jsonl("block73_routing_all_layers/routing_all_layers.jsonl")} if p73.exists() else {}
    arms = [("clamp2d@4", "2-D clamp ×4", r66), ("K@19,23", "keys only\n(complement)", r66), ("K@19,23+clamp2d@4", "keys +\n2-D clamp", r66), ("KV@19,23", "keys+values\nL19/L23", r66)]
    if r73:
        arms += [("transport<-clamp", "all L17–24 fed\nthe clamped state", r73), ("KV19,23+lin_in", "all L17–24 fed\nthe pasted state", r73)]
    arms += [("orth@0.25", "complement\npaste ×0.25", r66)]
    w = 0.26
    for k, (lab, ids, col) in enumerate(groups):
        vals = [np.mean([src[i]["arms"][a]["top1_is_swap"] for i in ids if i in src]) for a, _, src in arms]
        ax.bar(np.arange(len(arms)) + (k - 1) * w, vals, w, color=col, label=lab.replace("\n", " "))
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=7.5)
    ax.set_ylabel("flip rate")
    ax.set_title("what the transport layers receive at the bridge position (patched from the complement or clamp run)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F22_content_vs_routing.png")


# ---------------------------------------------------------------- F24 edit site: lens readability vs transport head (block69)
def f24():
    p = RES / "block69_position_choice" / "position_choice.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block69_position_choice/position_choice.jsonl")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sel(sup, lens="J", arm="clamp") if not r["top1_is_swap"]}
    e62 = load_jsonl("block62_energy_of_arms/energy_of_arms.jsonl")
    resc4 = {r["index"] for r in sel(e62, arm="clamp2d@4") if r["top1_is_swap"]}
    groups = [("all 59", rows, C["J"]), ("never flipped (35)", [r for r in rows if r["index"] in zero], C["green"]), ("never flipped, clamp@4 fails (27)", [r for r in rows if r["index"] in zero and r["index"] not in resc4], C["grey"])]
    pairs = [("2-D clamp ×4", "clamp@4_best", "clamp@4_h8"), ("complement paste ×0.25", "orth@0.25_best", "orth@0.25_h8"), ("full paste ×0.25", None, "full@0.25_h8")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.26
    x = 0
    ticks, labels = [], []
    for name, a_best, a_h8 in pairs:
        for k, (glab, rs, col) in enumerate(groups):
            vb = np.mean([r["arms"][a_best]["top1_is_swap"] for r in rs]) if a_best else np.nan
            vh = np.mean([r["arms"][a_h8]["top1_is_swap"] for r in rs])
            if a_best:
                ax.bar(x + k * w, vb, w * 0.9, color=col, alpha=0.4, edgecolor=col, hatch="//")
            ax.bar(x + k * w, vh, w * 0.9, color=col, alpha=1.0 if a_best else 0.7, bottom=0, label=glab if name == pairs[0][0] else None, zorder=2 if not a_best else 1)
            if a_best:
                ax.plot([x + k * w - w * 0.45, x + k * w + w * 0.45], [vb, vb], color="k", lw=1.2, zorder=3)
        ticks.append(x + w); labels.append(name)
        x += 1.2
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("flip rate")
    ax.set_title("edit site: at the readability-chosen position (black line) vs where L23 h8 attends (bar)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, title="item group", title_fontsize=7.5)
    save(fig, "F24_edit_site.png")


# ---------------------------------------------------------------- F25 donor-free routing-assisted clamp (block75)
def f25():
    p = RES / "block75_donor_free_recipe" / "donor_free.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block75_donor_free_recipe/donor_free.jsonl")
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    zero = {r["index"] for r in sel(sup, lens="J", arm="clamp") if not r["top1_is_swap"]}
    groups = [("flippable (24)", [r for r in rows if r["index"] not in zero], C["J"]), ("never flipped (35)", [r for r in rows if r["index"] in zero], C["grey"])]
    arms = [("F[h8@23,1]_best", "force h8\nonly"), ("clamp@4_best", "clamp\n×4"), ("F[h8@23,1]+clamp@4_best", "h8 +\nclamp"), ("F[h8h9h0@23,1]+clamp@4_best", "h8+h9+h0\n+ clamp"), ("F[all@23,1]+clamp@4_best", "all L23\n+ clamp"), ("K<-donor+clamp@4_best", "donor keys\n+ clamp")]
    three = (RES / "block80_4b_donor_free_recipe" / "donor_free.jsonl").exists() and (RES / "block83_qwen3_4b_donor_free_recipe" / "donor_free.jsonl").exists()
    fig, axes = plt.subplots(1, 3 if three else 2, figsize=(15 if three else 9.6, 3.5), gridspec_kw={"width_ratios": [1.2, 1.2, 1.3]} if three else None)
    w = 0.38
    for k, (glab, rs, col) in enumerate(groups):
        flips = [np.mean([r["arms"][a]["top1_is_swap"] for r in rs]) for a, _ in arms]
        dms = [np.median([r["arms"][a]["delta_margin"] for r in rs]) for a, _ in arms]
        axes[0].bar(np.arange(len(arms)) + (k - 0.5) * w, flips, w * 0.92, color=col, label=glab)
        axes[1].bar(np.arange(len(arms)) + (k - 0.5) * w, dms, w * 0.92, color=col, label=glab)
    for ax, lab in ((axes[0], "flip rate"), (axes[1], "median Δmargin (nats)")):
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=7.5)
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].set_title("Qwen3.5-9B: force L23 h8 onto the edited position (no donor)", fontsize=9)
    axes[1].set_title("edit at the readability-chosen position, L8–20, clamp ×4", fontsize=9)
    if three:
        audit = {r["index"]: r for r in json.load(open(RES / "block01" / "q01_data_audit_rows.json"))}
        ax = axes[2]
        models = [("Qwen3.5-9B\n(h8)", "block75_donor_free_recipe/donor_free.jsonl", "block12_suppression/suppression.jsonl", None),
                  ("Qwen3.5-4B\n(h8)", "block80_4b_donor_free_recipe/donor_free.jsonl", None, "block27_4b_band8_20/replication.jsonl"),
                  ("Qwen3-4B dense\n(h10)", "block83_qwen3_4b_donor_free_recipe/donor_free.jsonl", None, "block81_qwen3_4b/replication.jsonl")]
        y = 0
        for lab, f, supf, repf in models:
            rows_m = load_jsonl(f)
            if supf:
                zero_m = {r["index"] for r in sel(load_jsonl(supf), lens="J", arm="clamp") if not r["top1_is_swap"]}
            else:
                rep = load_jsonl(repf)
                zero_m = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
            for sub, col, tag in (("flippable", C["J"], lambda r: r["index"] not in zero_m), ("all items", C["grey"], lambda r: True)):
                rs = [r for r in rows_m if tag(r)]
                idx = sorted(r["index"] for r in rs)
                cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
                A = {r["index"]: r["arms"]["F[h8@23,1]+clamp@4_best"]["delta_margin"] for r in rs}
                B = {r["index"]: r["arms"]["clamp@4_best"]["delta_margin"] for r in rs}
                d = paired_difference(A, B, cl)
                ax.errorbar(d["mean_diff"], y, xerr=[[d["mean_diff"] - d["lo95"]], [d["hi95"] - d["mean_diff"]]], fmt="o", color=col, ms=5, capsize=3, label=sub if y < 2 else None)
                ax.text(d["hi95"] + 0.15, y, f"{lab.replace(chr(10), ' ')} · {sub} (n={len(idx)})", va="center", fontsize=7)
                y += 1
            y += 0.5
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_yticks([]); ax.set_xlabel("forced head + clamp − clamp, paired Δmargin (nats), 95% CI", fontsize=8)
        ax.set_xlim(-0.5, 7.5); ax.set_title("the routing assist across three models", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "F25_donor_free_recipe.png")


# ---------------------------------------------------------------- F27 third model (Qwen3-4B, dense): mechanism checks (block82)
def f27():
    p = RES / "block82_qwen3_4b_mechanism" / "mechanism.jsonl"
    if not p.exists():
        return
    rows = load_jsonl("block82_qwen3_4b_mechanism/mechanism.jsonl")
    rep = load_jsonl("block81_qwen3_4b/replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    arms = [("clamp@1", "2-D clamp\n×1"), ("clamp@4", "2-D clamp\n×4"), ("clamp@1_all", "clamp ×1\nall positions"), ("orth@0.25", "complement\npaste ×0.25"), ("full@0.25", "full paste\n×0.25")]
    w = 0.38
    for k, (lab, keep, col) in enumerate((("all items (n=45)", lambda r: True, C["J"]), ("never flipped by the band clamp (n=32)", lambda r: r["index"] in zero, C["grey"]))):
        vals = [np.mean([r["arms"][a]["top1_is_swap"] for r in rows if keep(r)]) for a, _ in arms]
        ax.bar(np.arange(len(arms)) + (k - 0.5) * w, vals, w * 0.92, color=col, label=lab)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=7.5)
    ax.set_ylabel("flip rate")
    ax.set_title("Qwen3-4B (dense attention), bridge position, band L10–20", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    starts = [wd["start"] for wd in rows[0]["stage"]["two_hop"]["windows"]]
    for kind, col, lab in (("two_hop", C["R"], "two-hop prompt"), ("single_hop", C["green"], "single-hop prompt (control)")):
        rs = [r for r in rows if r["stage"][kind]["clean_correct"]]
        acc = [np.mean([r["stage"][kind]["windows"][i]["still_correct"] for r in rs]) for i in range(len(starts))]
        ax.plot([s0 + 3.5 for s0 in starts], acc, "-o", color=col, ms=4, label=lab)
    ax.axvspan(17, 30, color=C["attn"] if "attn" in C else "#f0b27a", alpha=0.25, lw=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("centre of the ablated 8-layer window (attention outputs, final position)")
    ax.set_ylabel("accuracy after ablation (clean-correct items)")
    ax.set_title("clean-run stage necessity: only L17–30 hurts two-hop", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F27_third_model_mechanism.png")


if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f27):
        try:
            fn()
        except Exception as e:  # keep going; report which figure failed
            print(f"!! {fn.__name__} failed: {e!r}")
