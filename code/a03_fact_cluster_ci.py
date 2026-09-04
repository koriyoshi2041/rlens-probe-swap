#!/usr/bin/env python3
"""Recompute the headline paired contrasts with FACT-LEVEL clusters (audit finding M3).

Old clusters: reverse pairs only (55 clusters among the 59 eligible items).
New clusters: items sharing the same unordered {intermediate, swap_to} pair, transitive
closure (51 clusters). Prints old vs new side by side. CPU only, reads the synced results.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "audit_recompute"))
from audit_lib import RES, load_jsonl, per_item, sel  # noqa: E402
from rlens.analysis import cluster_map, fact_pairs, paired_difference  # noqa: E402

rows_audit = json.load(open(RES / "block01" / "q01_data_audit_rows.json"))
ITEMS = {int(r["index"]): {"intermediate": r["intermediate"], "swap_to": r["swap_to"]} for r in rows_audit}
ELIG = json.load(open(RES / "block01" / "eligible_hybrid_indices.json"))
REV = json.load(open(RES / "block01" / "q01_data_audit_summary.json"))["reverse_pairs"]
OLD = cluster_map(ELIG, REV)
NEW = cluster_map(ELIG, fact_pairs({i: ITEMS[i] for i in ELIG}))
print(f"clusters: old {len(set(OLD.values()))}, new {len(set(NEW.values()))}")
merged = {}
for i in ELIG:
    merged.setdefault(NEW[i], []).append(i)
print("multi-item fact clusters:", {k: v for k, v in merged.items() if len(v) > 1})


def both(label, a, b):
    o = paired_difference(a, b, OLD)
    n = paired_difference(a, b, NEW)
    flag = "  <-- CI crosses 0 differently" if ((o["lo95"] > 0) != (n["lo95"] > 0)) or ((o["hi95"] < 0) != (n["hi95"] < 0)) else ""
    print(f"{label:48s} old {o['mean_diff']:+.3f} [{o['lo95']:+.3f},{o['hi95']:+.3f}] | new {n['mean_diff']:+.3f} [{n['lo95']:+.3f},{n['hi95']:+.3f}] (n={n['n']}){flag}")


P, E = "primary_L8_20", "early_L3_8"
b2 = load_jsonl("block02/block02_records.jsonl")
sw = lambda lens, band, alpha: per_item(sel(b2, condition="swap_raw", lens=lens, band=band, alpha=alpha, positions="all"))
try:
    both("R-J involution early a=0.5", sw("R", E, 0.5), sw("J", E, 0.5))
    both("R-J involution early a=1", sw("R", E, 1.0), sw("J", E, 1.0))
    both("R-J involution primary a=1", sw("R", P, 1.0), sw("J", P, 1.0))
except Exception as e:  # schema differs: fall back to whatever keys exist
    print("block02 schema note:", e, "; keys:", sorted(b2[0].keys()))
for kind in ("J", "R"):
    try:
        both(f"{kind} swap - shuffled pair (primary a=1)", sw(kind, P, 1.0), per_item(sel(b2, condition="swap_shuffled", lens=kind, band=P, alpha=1.0, positions="all")))
        both(f"{kind} swap - projection ablation (primary)", sw(kind, P, 1.0), per_item(sel(b2, condition="ablate_source", lens=kind, band=P, positions="all")))
    except Exception as e:
        print("control schema note:", e)
        conds = sorted({r.get("condition") for r in b2})
        print("conditions:", conds)
        break

dec = load_jsonl("block03_explore/explore_decomposition.jsonl")
for kind in ("J", "R"):
    both(f"{kind} early swap - install", per_item(sel(dec, band=E, lens=kind, mode="swap")), per_item(sel(dec, band=E, lens=kind, mode="install")))
ort = load_jsonl("block03_ortho/ortho.jsonl")
both("J primary ortho_rescaled - full", per_item(sel(ort, group=P, lens="J", arm="ortho_rescaled")), per_item(sel(ort, group=P, lens="J", arm="full")))
both("R-J ortho early", per_item(sel(ort, group=E, lens="R", arm="ortho_rescaled")), per_item(sel(ort, group=E, lens="J", arm="ortho_rescaled")))
c9 = load_jsonl("block09_clamp_controls/clamp_controls.jsonl")
both("early clamp R-J (full)", per_item(sel(c9, band=E, lens="R", pair="entity", arm="full")), per_item(sel(c9, band=E, lens="J", pair="entity", arm="full")))
both("early clamp R-J (ortho)", per_item(sel(c9, band=E, lens="R", pair="entity", arm="ortho_rescaled")), per_item(sel(c9, band=E, lens="J", pair="entity", arm="ortho_rescaled")))
both("primary clamp R-J (full)", per_item(sel(c9, band=P, lens="R", pair="entity", arm="full")), per_item(sel(c9, band=P, lens="J", pair="entity", arm="full")))
sup = load_jsonl("block12_suppression/suppression.jsonl")
for kind in ("J", "R"):
    clamp = per_item(sel(sup, lens=kind, arm="clamp"))
    inst = per_item(sel(sup, lens=kind, arm="install"))
    rem = per_item(sel(sup, lens=kind, arm="remove"))
    add = {i: inst[i] + rem[i] for i in clamp if i in inst and i in rem}
    both(f"{kind} synergy clamp - (install+remove)", clamp, add)
lad = load_jsonl("block10_ladder/ladder.jsonl")
both("J clamp - logit clamp (scale 1)", per_item(sel(lad, lens="J", arm="clamp", scale=1.0)), per_item(sel(lad, lens="logit", arm="clamp", scale=1.0)))
m3 = load_jsonl("block11_mechanism/m3_parity.jsonl")
for kind in ("J", "R"):
    for arm in ("involution", "clamp"):
        odd = {}
        even = {}
        for r in sel(m3, lens=kind, arm=arm):
            (odd if int(r["width"]) % 2 else even).setdefault(int(r["index"]), []).append(float(r["delta_margin"]))
        both(f"{kind} {arm} odd-even (widths 1..13)", {i: float(np.mean(v)) for i, v in odd.items()}, {i: float(np.mean(v)) for i, v in even.items()})
lev = load_jsonl("block19_ladder_eval/ladder_eval.jsonl")
try:
    keys = sorted(lev[0].keys())
    print("ladder_eval keys:", keys)
    n25 = per_item(sel(lev, source="fit_n25", arm="clamp")) if "source" in keys else None
    n400 = per_item(sel(lev, source="fit_n400", arm="clamp")) if "source" in keys else None
    pub = per_item(sel(lev, source="published", arm="clamp")) if "source" in keys else None
    if n25 and n400:
        both("self-fit n400 - n25 (clamp)", n400, n25)
    if n25 and pub:
        both("self-fit n25 - published (clamp)", n25, pub)
except Exception as e:
    print("ladder_eval note:", e)
