"""Result loaders and cluster-bootstrap helpers shared by the write-up figures.

Every number drawn by a figure is computed here from the synced JSONL results; nothing
is typed in by hand. CIs are fact-level cluster bootstraps (items sharing an entity pair
form one cluster), the same convention as the log and the claims pack.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "audit_recompute"))
from audit_lib import RES, load_jsonl, per_item, sel  # noqa: E402,F401
from rlens.analysis import bootstrap_mean, cluster_map, fact_pairs, paired_difference  # noqa: E402

N_BOOT = 5000

_audit_rows = json.load(open(RES / "block01" / "q01_data_audit_rows.json"))
ITEMS: Dict[int, dict] = {int(r["index"]): r for r in _audit_rows}


def clusters(indices: Iterable[int]) -> Dict[int, int]:
    idx = sorted(int(i) for i in indices)
    return cluster_map(idx, fact_pairs({i: ITEMS[i] for i in idx}))


def item_values(rows: Iterable[dict], field: str = "top1_is_swap") -> Dict[int, float]:
    return per_item(rows, field)


def rate_ci(rows: Iterable[dict], field: str = "top1_is_swap", seed: int = 0) -> Tuple[float, float, float, int]:
    vals = item_values(rows, field)
    if not vals:
        return np.nan, np.nan, np.nan, 0
    m, lo, hi = bootstrap_mean(vals, clusters(vals), n_boot=N_BOOT, seed=seed)
    return m, lo, hi, len(vals)


def paired_ci(a: Dict[int, float], b: Dict[int, float], cl: Dict[int, int] | None = None, seed: int = 0) -> dict:
    shared = sorted(set(a) & set(b))
    return paired_difference(a, b, cl or clusters(shared), n_boot=N_BOOT, seed=seed)


def flip_rate(rows: Iterable[dict], field: str = "top1_is_swap") -> float:
    vals = item_values(rows, field)
    return float(np.mean(list(vals.values()))) if vals else np.nan


def zero_set_9b() -> set:
    sup = load_jsonl("block12_suppression/suppression.jsonl")
    return {int(r["index"]) for r in sel(sup, lens="J", arm="clamp") if not r["top1_is_swap"]}


def zero_set_from_replication(rel: str) -> set:
    rep = load_jsonl(rel)
    rows = sel(rep, stage="band", lens="J", arm="clamp", control="full")
    return {int(r["index"]) for r in rows if not r["top1_is_swap"]}


def rescued_by_clamp4() -> set:
    e62 = load_jsonl("block62_energy_of_arms/energy_of_arms.jsonl")
    return {int(r["index"]) for r in sel(e62, arm="clamp2d@4") if r["top1_is_swap"]}


def newitems() -> Tuple[Dict[str, dict], Dict[str, str]]:
    """LLM-written non-geographic items (block 55 v2): name -> item, and name -> fact key."""
    items = json.load(open(RES / "block55_newitems_v2" / "newitems.json"))
    by_name = {it["name"]: it for it in items}
    fact = {n: "|".join(sorted((it["intermediate"].strip().lower(), it["swap_to"].strip().lower()))) for n, it in by_name.items()}
    return by_name, fact


def clusters_by_key(names: Iterable[str], key_of: Dict[str, str]) -> Dict[str, str]:
    return {n: key_of[n] for n in names}


def bootstrap_named(values: Dict[str, float], cl: Dict[str, str], seed: int = 0) -> Tuple[float, float, float]:
    """Cluster bootstrap for string-keyed items (the out-of-sample set)."""
    groups: Dict[str, List[float]] = {}
    for k, v in values.items():
        groups.setdefault(cl[k], []).append(v)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(keys), len(keys))
        draws[b] = float(np.mean([v for k in pick for v in groups[keys[k]]]))
    flat = [v for g in groups.values() for v in g]
    return float(np.mean(flat)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
