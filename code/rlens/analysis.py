"""Analysis of block-02 records: paired J-vs-R comparison with cluster bootstrap.

Clusters: reverse pairs share a fact, so they are resampled together. Everything here
is descriptive; the pre-registered headline is delta_margin at alpha=1 in the primary band.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def load_records(path: pathlib.Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(records: Iterable[Dict[str, object]], **filters) -> List[Dict[str, object]]:
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def by_index(rows: Iterable[Dict[str, object]], field: str) -> Dict[int, float]:
    """One value per item; if several rows share an index (e.g. seeds) their mean is used."""
    acc: Dict[int, List[float]] = {}
    for r in rows:
        acc.setdefault(int(r["index"]), []).append(float(r[field]))
    return {i: float(np.mean(v)) for i, v in acc.items()}


def cluster_map(indices: Sequence[int], pairs: Sequence[Sequence[int]]) -> Dict[int, int]:
    """Assign a cluster id per item: items linked by any chain of pairs share one (transitive closure).

    The earlier version assigned ``cluster[b] = cluster[a]`` pair by pair, which is not a
    closure: for pairs (1, 2), (3, 2) item 1 kept its own id. Union-find fixes that.
    """
    parent = {i: i for i in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in indices}


def fact_pairs(items_by_index: Dict[int, Dict[str, str]]) -> List[List[int]]:
    """Pairs of items that share the same unordered {intermediate, swap_to} entity pair.

    This is the fact-level dependence structure: reverse pairs (France->Italy and
    Italy->France) AND same-direction duplicates (France->Italy asked as capital and as
    language) both share the bridge fact, so they are resampled together. ``items_by_index``
    maps item index -> dict with keys ``intermediate`` and ``swap_to``.
    """
    key_to_items: Dict[tuple, List[int]] = {}
    for index, item in items_by_index.items():
        key = tuple(sorted((item["intermediate"].strip().lower(), item["swap_to"].strip().lower())))
        key_to_items.setdefault(key, []).append(int(index))
    pairs: List[List[int]] = []
    for members in key_to_items.values():
        members = sorted(members)
        pairs.extend([members[0], other] for other in members[1:])
    return pairs


def bootstrap_mean(
    values: Dict[int, float],
    clusters: Dict[int, int],
    n_boot: int = 10000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Cluster bootstrap of the mean; returns (mean, lo95, hi95)."""
    groups: Dict[int, List[float]] = {}
    for index, value in values.items():
        groups.setdefault(clusters.get(index, index), []).append(value)
    keys = list(groups)
    flat = np.array([v for g in groups.values() for v in g])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(keys), len(keys))
        vals = [v for k in picked for v in groups[keys[k]]]
        draws[b] = float(np.mean(vals))
    return float(flat.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_difference(
    a: Dict[int, float],
    b: Dict[int, float],
    clusters: Dict[int, int],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap of mean(a - b) over the items present in both."""
    shared = sorted(set(a) & set(b))
    diff = {i: a[i] - b[i] for i in shared}
    mean, lo, hi = bootstrap_mean(diff, clusters, n_boot=n_boot, seed=seed)
    wins = sum(1 for i in shared if diff[i] > 0)
    return {
        "n": len(shared),
        "mean_diff": mean,
        "lo95": lo,
        "hi95": hi,
        "median_diff": float(np.median([diff[i] for i in shared])),
        "win_rate": wins / len(shared) if shared else float("nan"),
    }


def rate(rows: Iterable[Dict[str, object]], field: str) -> float:
    vals = [bool(r[field]) for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_condition(rows: List[Dict[str, object]], clusters: Dict[int, int], seed: int = 0) -> Dict[str, float]:
    if not rows:
        return {}
    margin = by_index(rows, "delta_margin")
    mean, lo, hi = bootstrap_mean(margin, clusters, seed=seed)
    return {
        "n_items": len(margin),
        "delta_margin_mean": mean,
        "delta_margin_lo95": lo,
        "delta_margin_hi95": hi,
        "delta_margin_median": float(np.median(list(margin.values()))),
        "delta_logp_answer_median": float(np.median(list(by_index(rows, "delta_logp_answer").values()))),
        "delta_logp_swap_median": float(np.median(list(by_index(rows, "delta_logp_swap_answer").values()))),
        "top1_swap_rate": rate(rows, "top1_is_swap"),
        "top1_answer_rate": rate(rows, "top1_is_answer"),
        "top1_neither_rate": 1.0 - rate(rows, "top1_is_swap") - rate(rows, "top1_is_answer"),
        "kl_median": float(np.median([r["kl_clean_to_hooked"] for r in rows])),
        "energy_median": float(np.median([r["total_dh2"] for r in rows])),
    }
