#!/usr/bin/env python3
"""Offline analysis of the audit-driven runs (blocks 57-60). CPU only.  Usage: a14_analyze_audit_runs.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import per_item, sel  # noqa: E402
from rlens.analysis import cluster_map, fact_pairs, load_records, paired_difference  # noqa: E402


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    bf = {r["index"]: r["top1_is_swap"] for r in sel(sup, lens="J", arm="clamp")}
    audit = {r["index"]: r for r in json.load(open(root / "block01" / "q01_data_audit_rows.json"))}

    # block57: additive attribution along the fixed final direction
    p = root / "block30_answer_emergence_logitdir" / "answer_emergence.jsonl"
    if p.exists():
        rows = sel(load_records(p), lens="J")
        print("===== block57: attribution along the FIXED final direction w = W_U[:,s]-W_U[:,a] (adds up to Δ<h_out31, w>) =====")
        for gname, g in (("flipped", [r for r in rows if r["top1_is_swap"]]), ("unflipped", [r for r in rows if not r["top1_is_swap"]])):
            A = np.array([[q["attn_answer"] for q in r["layers"]] for r in g]); M = np.array([[q["mlp_answer"] for q in r["layers"]] for r in g]); D = np.array([[q["direct_answer"] for q in r["layers"]] for r in g])
            R = np.array([r["layers"][-1]["resid_answer_delta"] for r in g])
            layers = [q["layer"] for q in g[0]["layers"]]
            tot = A.sum(1) + M.sum(1) + D.sum(1)
            print(f"  {gname} (n={len(g)}): median attn {np.median(A.sum(1)):+.2f}  mlp {np.median(M.sum(1)):+.2f}  direct {np.median(D.sum(1)):+.2f}  | sum {np.median(tot):+.2f} vs Δ<h_out31,w> {np.median(R):+.2f} | measured ΔM {np.median([r['delta_margin'] for r in g]):+.2f}")
            share_attn = np.median(A.sum(1) / np.where(np.abs(tot) > 1e-6, tot, np.nan)); share_direct = np.median(D.sum(1) / np.where(np.abs(tot) > 1e-6, tot, np.nan))
            print(f"    per-item share: attention {share_attn:.2f}, direct {share_direct:.2f}; top attention layers by median: " + ", ".join(f"L{layers[i]} {np.median(A[:, i]):+.2f}" for i in np.argsort(-np.median(A, 0))[:4]) + "; top MLP layers: " + ", ".join(f"L{layers[i]} {np.median(M[:, i]):+.2f}" for i in np.argsort(-np.median(M, 0))[:4]))

    # block58: energy-matched paste vs clamp
    p = root / "block58_energy_matched_paste" / "energy_matched_paste.jsonl"
    if p.exists():
        rows = load_records(p)
        print("\n===== block58: single-position 2-D clamp (over-driven) vs complement paste (scaled down); flips at matched KL =====")
        print(f"  {'arm':12s} {'flip all':>8s} {'flip zero35':>11s} {'neither':>8s} {'KL med':>7s} {'ΔM med':>7s}")
        order = ["clamp2d@1", "clamp2d@1.5", "clamp2d@2", "clamp2d@3", "clamp2d@4", "orth@0.1", "orth@0.25", "orth@0.5", "orth@1", "full@0.25", "full@0.5", "full@1", "rand16", "rand64", "rand256", "Rtop256"]
        for arm in order:
            s = sel(rows, arm=arm)
            if not s:
                continue
            z = [r for r in s if not bf.get(r["index"], False)]
            print(f"  {arm:12s} {np.mean([r['top1_is_swap'] for r in s]):8.2f} {np.mean([r['top1_is_swap'] for r in z]):11.2f} {np.mean([not r['top1_is_swap'] and not r['top1_is_answer'] for r in s]):8.2f} {np.median([r['kl'] for r in s]):7.2f} {np.median([r['delta_margin'] for r in s]):7.2f}")

    # block59: gain controls with in-subspace random field
    p = root / "block59_gain_controls_subspace" / "gain_controls.jsonl"
    if p.exists():
        rows = sel(load_records(p), lens="J")
        print("\n===== block59: gain controls incl. random field INSIDE J's top-256 subspace (norm-matched to the clamp field) =====")
        for k in ("clamp", "contrast", "ortho", "random", "random_top256"):
            pred = np.array([r["predicted"][k] for r in rows]); real = np.array([r["realised"][k] for r in rows])
            ok = pred > 0.05
            print(f"  {k:14s} pred med {np.median(pred):6.2f}  real med {np.median(real):6.2f}  gain(med ratio) {np.median(real[ok]/pred[ok]) if ok.any() else float('nan'):6.2f}  (n ok={ok.sum()})  flipped real {np.median(real[[r['top1_is_swap'] for r in rows]]):.2f}")

    # block60: self-consistent scaling + clean-run transfer-stage ablation
    p = root / "block60_selfconsistent" / "selfconsistent.jsonl"
    if p.exists():
        rows = load_records(p)
        idx = sorted({r["index"] for r in rows}); cl = cluster_map(idx, fact_pairs({i: audit[i] for i in idx}))
        print("\n===== block60a: early band, REAL clamps at global scales (energy / KL matched) =====")
        ref = per_item(sel(rows, part="a", arm="J@1"))
        for arm in sorted({r["arm"] for r in sel(rows, part="a")}):
            s = sel(rows, part="a", arm=arm); d = paired_difference(per_item(s), ref, cl)
            print(f"  {arm:16s} flip {np.mean([r['top1_is_swap'] for r in s]):.2f}  energy med {np.median([r['energy'] for r in s]):6.1f}  KL med {np.median([r['kl'] for r in s]):.3f}  ΔM−J@1: {d['mean_diff']:+.2f} [{d['lo95']:+.2f},{d['hi95']:+.2f}]")
        print("===== block60b: CLEAN run, mean-ablate a stage at the final position =====")
        for kind in ("two_hop", "single_hop"):
            for arm in ("attn_17_24", "mlp_26_29", "attn_9_16"):
                s = sel(rows, part="b", prompt_kind=kind, arm=arm); cc = [r for r in s if r["clean_correct"]]
                print(f"  {kind:10s} {arm:10s} clean acc {np.mean([r['clean_correct'] for r in s]):.2f} -> after {np.mean([r['still_correct'] for r in cc]):.2f}  Δlogp(ans) med {np.median([r['delta_logp_answer'] for r in cc]):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
