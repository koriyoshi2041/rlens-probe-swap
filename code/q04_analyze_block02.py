#!/usr/bin/env python
"""Analyze block-02 records and write the summary tables the write-up needs."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from rlens import config_block02 as cfg
from rlens.analysis import (
    bootstrap_mean,
    by_index,
    cluster_map,
    load_records,
    paired_difference,
    select,
    summarize_condition,
)
from rlens.paths import RESULTS_DIR


def main() -> int:
    out_dir = RESULTS_DIR / (sys.argv[1] if len(sys.argv) > 1 else "block02")
    records = load_records(out_dir / "block02_records.jsonl")
    audit = json.loads((RESULTS_DIR / "block01" / "q01_data_audit_summary.json").read_text(encoding="utf-8"))
    indices = sorted({int(r["index"]) for r in records})
    clusters = cluster_map(indices, audit["reverse_pairs"])
    n_clusters = len(set(clusters[i] for i in indices))
    print(f"items={len(indices)} clusters={n_clusters} records={len(records)}")

    report = {"n_items": len(indices), "n_clusters": n_clusters, "n_records": len(records), "conditions": {}, "contrasts": {}}

    # --- headline and dose-response tables -------------------------------
    print("\n=== swap_raw: per band / lens / alpha (all positions) ===")
    print(f"{'band':16s} {'lens':6s} {'a':>4s} {'ΔM mean [95% CI]':>26s} {'ΔM med':>7s} {'Δans':>7s} {'Δswap':>7s} {'top1=swap':>9s} {'neither':>8s} {'KL':>7s} {'energy':>9s}")
    for band in cfg.BANDS:
        for lens in cfg.LENSES_SWAP:
            for alpha in cfg.ALPHAS_ALL:
                rows = select(records, condition="swap_raw", band=band, lens=lens, alpha=alpha, positions="all")
                s = summarize_condition(rows, clusters)
                if not s:
                    continue
                report["conditions"][f"swap_raw|{band}|{lens}|{alpha}"] = s
                print(f"{band:16s} {lens:6s} {alpha:4} {s['delta_margin_mean']:+8.2f} [{s['delta_margin_lo95']:+6.2f},{s['delta_margin_hi95']:+6.2f}] {s['delta_margin_median']:+7.2f} {s['delta_logp_answer_median']:+7.2f} {s['delta_logp_swap_median']:+7.2f} {s['top1_swap_rate']:9.2f} {s['top1_neither_rate']:8.2f} {s['kl_median']:7.2f} {s['energy_median']:9.3g}")

    print("\n=== controls at alpha in", cfg.CONTROL_ALPHAS, "===")
    print(f"{'condition':16s} {'band':16s} {'lens':6s} {'a':>4s} {'ΔM mean [95% CI]':>26s} {'top1=swap':>9s} {'energy':>9s}")
    for cond in ("swap_unit", "swap_shuffled", "swap_gauss", "swap_foil"):
        for band in cfg.BANDS:
            for lens in cfg.LENSES_CONTROL:
                for alpha in cfg.CONTROL_ALPHAS:
                    rows = select(records, condition=cond, band=band, lens=lens, alpha=alpha)
                    s = summarize_condition(rows, clusters)
                    if not s:
                        continue
                    report["conditions"][f"{cond}|{band}|{lens}|{alpha}"] = s
                    print(f"{cond:16s} {band:16s} {lens:6s} {alpha:4} {s['delta_margin_mean']:+8.2f} [{s['delta_margin_lo95']:+6.2f},{s['delta_margin_hi95']:+6.2f}] {s['top1_swap_rate']:9.2f} {s['energy_median']:9.3g}")
    for band in cfg.BANDS:
        for lens in cfg.LENSES_CONTROL:
            rows = select(records, condition="ablate_src", band=band, lens=lens)
            s = summarize_condition(rows, clusters)
            if s:
                report["conditions"][f"ablate_src|{band}|{lens}"] = s
                print(f"{'ablate_src':16s} {band:16s} {lens:6s} {'-':>4s} {s['delta_margin_mean']:+8.2f} [{s['delta_margin_lo95']:+6.2f},{s['delta_margin_hi95']:+6.2f}] {s['top1_swap_rate']:9.2f} {s['energy_median']:9.3g}")

    # --- the pre-registered contrasts ------------------------------------
    print("\n=== paired contrasts (cluster bootstrap, 10k) ===")
    def contrast(name, rows_a, rows_b, field="delta_margin"):
        a, b = by_index(rows_a, field), by_index(rows_b, field)
        if not a or not b:
            return
        d = paired_difference(a, b, clusters)
        report["contrasts"][name] = d
        print(f"{name:52s} n={d['n']:2d} mean={d['mean_diff']:+7.3f} [{d['lo95']:+7.3f},{d['hi95']:+7.3f}] med={d['median_diff']:+7.3f} win={d['win_rate']:.2f}")

    for band in cfg.BANDS:
        for alpha in cfg.ALPHAS_MAIN:
            contrast(
                f"R-J | swap_raw | {band} | a={alpha}",
                select(records, condition="swap_raw", band=band, lens="R", alpha=alpha, positions="all"),
                select(records, condition="swap_raw", band=band, lens="J", alpha=alpha, positions="all"),
            )
        for lens in cfg.LENSES_CONTROL:
            contrast(
                f"{lens} swap - shuffled | {band} | a=1.0",
                select(records, condition="swap_raw", band=band, lens=lens, alpha=1.0, positions="all"),
                select(records, condition="swap_shuffled", band=band, lens=lens, alpha=1.0),
            )
            contrast(
                f"{lens} swap - ablate_src | {band} | a=1.0",
                select(records, condition="swap_raw", band=band, lens=lens, alpha=1.0, positions="all"),
                select(records, condition="ablate_src", band=band, lens=lens),
            )
            contrast(
                f"{lens} swap - foil | {band} | a=1.0",
                select(records, condition="swap_raw", band=band, lens=lens, alpha=1.0, positions="all"),
                select(records, condition="swap_foil", band=band, lens=lens, alpha=1.0),
            )
            contrast(
                f"{lens} unit - raw | {band} | a=1.0",
                select(records, condition="swap_unit", band=band, lens=lens, alpha=1.0),
                select(records, condition="swap_raw", band=band, lens=lens, alpha=1.0, positions="all"),
            )
    for alpha in cfg.ALPHAS_MAIN:
        contrast(
            f"R-J | swap_unit | {cfg.PRIMARY_BAND} | a={alpha}",
            select(records, condition="swap_unit", band=cfg.PRIMARY_BAND, lens="R", alpha=alpha),
            select(records, condition="swap_unit", band=cfg.PRIMARY_BAND, lens="J", alpha=alpha),
        ) if alpha in cfg.CONTROL_ALPHAS else None

    # --- single-layer profile --------------------------------------------
    print("\n=== single-layer swap profile (alpha=1, all positions) ===")
    print(f"{'L':>3s} {'J ΔM':>8s} {'R ΔM':>8s} {'R-J':>8s} {'J top1sw':>9s} {'R top1sw':>9s} {'J E':>8s} {'R E':>8s}")
    profile = []
    for layer in cfg.SINGLE_LAYER_PROFILE:
        rj = {}
        for lens in cfg.LENSES_CONTROL:
            rows = select(records, condition="swap_single_layer", lens=lens, layer=layer)
            rj[lens] = summarize_condition(rows, clusters)
        if not rj.get("J") or not rj.get("R"):
            continue
        d = paired_difference(
            by_index(select(records, condition="swap_single_layer", lens="R", layer=layer), "delta_margin"),
            by_index(select(records, condition="swap_single_layer", lens="J", layer=layer), "delta_margin"),
            clusters,
            n_boot=2000,
        )
        profile.append({"layer": layer, "J": rj["J"], "R": rj["R"], "R_minus_J": d})
        print(f"{layer:3d} {rj['J']['delta_margin_median']:+8.2f} {rj['R']['delta_margin_median']:+8.2f} {d['mean_diff']:+8.3f} {rj['J']['top1_swap_rate']:9.2f} {rj['R']['top1_swap_rate']:9.2f} {rj['J']['energy_median']:8.3g} {rj['R']['energy_median']:8.3g}")
    report["single_layer_profile"] = profile

    # --- position sensitivity --------------------------------------------
    print("\n=== position sensitivity (primary band, alpha=1) ===")
    for pos in ["all"] + list(cfg.POSITION_SENSITIVITY):
        for lens in cfg.LENSES_CONTROL:
            rows = select(records, condition="swap_raw", band=cfg.PRIMARY_BAND, lens=lens, alpha=1.0, positions=pos)
            s = summarize_condition(rows, clusters)
            if s:
                report["conditions"][f"pos|{pos}|{lens}"] = s
                print(f"{pos:6s} {lens:2s} ΔM={s['delta_margin_mean']:+7.2f} [{s['delta_margin_lo95']:+6.2f},{s['delta_margin_hi95']:+6.2f}] top1sw={s['top1_swap_rate']:.2f} E={s['energy_median']:9.3g}")

    # --- stop rule --------------------------------------------------------
    stop = cfg.STOP_RULE
    verdict = {}
    for lens in cfg.LENSES_CONTROL:
        rows = select(records, condition="swap_raw", band=stop["band"], lens=lens, alpha=stop["alpha"], positions="all")
        s = summarize_condition(rows, clusters)
        verdict[lens] = {"top1_swap_rate": s["top1_swap_rate"], "median_delta_margin": s["delta_margin_median"],
                         "passes": s["top1_swap_rate"] >= stop["min_top1_swap_rate"] and s["delta_margin_median"] >= stop["min_median_delta_margin"]}
    report["stop_rule"] = {"rule": stop, "verdict": verdict, "any_pass": any(v["passes"] for v in verdict.values())}
    print("\n=== pre-registered stop rule ===")
    print(json.dumps(report["stop_rule"], indent=1))

    (out_dir / "analysis_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir / 'analysis_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
