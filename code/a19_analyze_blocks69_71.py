#!/usr/bin/env python3
"""Offline analysis of blocks 69 (position choice + foil keys), 71 (necessity ladder) and 70 (4B key/value).
CPU only.  Usage: a19_analyze_blocks69_71.py <results_root>"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rlens.analysis import load_records  # noqa: E402


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def groups_9b(root):
    sup = load_records(root / "block12_suppression" / "suppression.jsonl")
    zero = {r["index"] for r in sup if r["lens"] == "J" and r["arm"] == "clamp" and not r["top1_is_swap"]}
    e62 = load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e62 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    return zero, resc4


def block69(root):
    p = root / "block69_position_choice" / "position_choice.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, resc4 = groups_9b(root)
    groups = {"all": rows, "flipped by band clamp": [r for r in rows if r["index"] not in zero],
              "zero-flip (35)": [r for r in rows if r["index"] in zero],
              "zero: clamp2d@4 fails (27)": [r for r in rows if r["index"] in zero and r["index"] not in resc4]}
    print("===== block69: where does h8 look, and does editing THERE help? =====")
    for g, rs in groups.items():
        same = np.mean([r["p_best"] == r["p_h8"] for r in rs])
        print(f"  [{g}] n={len(rs)}  p_h8 == p_best: {same:.2f}; h8 mass median: best {med([r['h8_mass']['best'] for r in rs]):.2f} h8pos {med([r['h8_mass']['h8pos'] for r in rs]):.2f} final {med([r['h8_mass']['final'] for r in rs]):.2f} pos0 {med([r['h8_mass']['pos0'] for r in rs]):.2f}; readability hits: best {med([r['hits_best'] for r in rs]):.0f} h8pos {med([r['hits_h8'] for r in rs]):.0f}")
    rs27 = groups["zero: clamp2d@4 fails (27)"]
    print("  tokens h8 attends to (27 fails): " + ", ".join(f"{r['name']}:{r['tok_h8']!r}@{r['p_h8']}/{r['n_prompt']}" for r in rs27[:27]))
    arms = ["clamp@1_best", "clamp@4_best", "clamp@1_h8", "clamp@4_h8", "clamp@4_both", "clamp@1_all", "clamp@4_all", "orth@0.25_best", "orth@0.25_h8", "full@0.25_h8",
            "foil_orth@0.25_best", "foil_full@0.25_best", "K<-donor", "K<-foil", "K<-donor+clamp@4_best", "K<-foil+clamp@4_best", "K<-foil+clamp@1_best"]
    for g, rs in groups.items():
        print(f"\n  --- [{g}] n={len(rs)}: flip | top1=answer | median ΔM | h8→best | h8→h8pos ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs]
            print(f"    {arm:22s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ans {np.mean([q['top1_is_answer'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h8→best {med([q['h8_best'] for q in a]):.3f}  h8→h8pos {med([q['h8_h8pos'] for q in a]):.3f}")
    # foil: does the output become the foil's answer?
    def is_foil_answer(r, arm):
        t = r["arms"][arm]["top1_str"].strip().lower()
        f = r["foil_answer"].strip().lower()
        return bool(t) and (f.startswith(t) or t.startswith(f[: max(1, len(t))]))
    for arm in ("foil_orth@0.25_best", "foil_full@0.25_best", "K<-foil"):
        print(f"  {arm:22s}: top1 = foil's answer in {np.mean([is_foil_answer(r, arm) for r in rows]):.2f} of items")
    # consistency with block 62 / 66
    e62 = {(r["index"], r["arm"]): r["delta_margin"] for r in load_records(root / "block62_energy_of_arms" / "energy_of_arms.jsonl")}
    d = [abs(r["arms"]["clamp@4_best"]["delta_margin"] - e62[(r["index"], "clamp2d@4")]) for r in rows]
    print(f"  consistency clamp@4_best vs block62 clamp2d@4: max |diff| {max(d):.4f}")
    k66 = {r["index"]: r["arms"]["K@19,23+clamp2d@4"]["delta_margin"] for r in load_records(root / "block66_key_value" / "key_value.jsonl")}
    d = [abs(r["arms"]["K<-donor+clamp@4_best"]["delta_margin"] - k66[r["index"]]) for r in rows if r["index"] in k66]
    print(f"  consistency K<-donor+clamp@4 vs block66: max |diff| {max(d):.4f}")


def block71(root):
    p = root / "block71_necessity_ladder" / "necessity_ladder.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    zero, _ = groups_9b(root)
    print("\n===== block71: necessity ladder — full paste (x0.25) minus a subspace, at the best bridge position =====")
    print(f"  {'arm':14s} {'flip':>5s} {'flipZ':>5s} {'ΔM med':>7s} {'KL med':>7s} {'E med':>7s}")
    order = ["full", "full-Jtop2", "full-Jtop16", "full-Jtop64", "full-Jtop256", "full-Jtop1024", "full-Rtop256", "full-rand16", "full-rand64", "full-rand256", "full-rand1024"]
    for arm in order:
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        z = [r for r in s if r["index"] in zero]
        print(f"  {arm:14s} {np.mean([r['top1_is_swap'] for r in s]):5.2f} {np.mean([r['top1_is_swap'] for r in z]):5.2f} {med([r['delta_margin'] for r in s]):+7.2f} {med([r['kl'] for r in s]):7.2f} {med([r['energy'] for r in s]):7.1f}")


def block70(root):
    p = root / "block70_4b_key_value" / "key_value.jsonl"
    if not p.exists():
        return
    rows = load_records(p)
    check = json.load(open(root / "block70_4b_key_value" / "self_check.json"))
    print("\n===== block70: Qwen3.5-4B key/value patching =====")
    print("  self-check:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in check.items()})
    rep_dir = root / "block27_4b_band8_20" if (root / "block27_4b_band8_20" / "replication.jsonl").exists() else root / "block27_4b"
    rep = load_records(rep_dir / "replication.jsonl")
    zero = {r["index"] for r in rep if r["stage"] == "band" and r["lens"] == "J" and r["arm"] == "clamp" and r["control"] == "full" and not r["top1_is_swap"]}
    e65 = load_records(root / "block65_4b_energy_of_arms" / "energy_of_arms.jsonl")
    resc4 = {r["index"] for r in e65 if r["arm"] == "clamp2d@4" and r["top1_is_swap"]}
    groups = {"all": rows, "flipped by 4B band clamp": [r for r in rows if r["index"] not in zero], "zero: clamp2d@4 fails": [r for r in rows if r["index"] in zero and r["index"] not in resc4]}
    arms = ["clean", "clamp2d@4", "orth@0.25", "K@19,23", "V@19,23", "KV@19,23", "PAT@19,23", "K@19,23+clamp2d@4", "PAT@19,23+clamp2d@4", "KV@19,23+clamp2d@4"]
    for g, rs in groups.items():
        print(f"  --- [{g}] n={len(rs)}: flip | median ΔM | h8→bridge ---")
        for arm in arms:
            a = [r["arms"][arm] for r in rs]
            print(f"    {arm:22s} flip {np.mean([q['top1_is_swap'] for q in a]):.2f}  ΔM {med([q['delta_margin'] for q in a]):+6.2f}  h8 {med([q['L23_h8_to_bridge'] for q in a]):.3f}")


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    block69(root)
    block71(root)
    block70(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
