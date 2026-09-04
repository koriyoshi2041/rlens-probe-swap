"""Shared helpers for the independent recomputation audit (read-only on project files)."""
from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]  # the folder that holds 正式研究/ and 正式研究同步/ (original layout)
_repo_results = pathlib.Path(__file__).resolve().parents[2] / "results"  # open-source layout: <repo>/results next to <repo>/code
RES = pathlib.Path(os.environ["RLENS_RESULTS"]) if os.environ.get("RLENS_RESULTS") else (_repo_results if _repo_results.exists() else ROOT / "正式研究同步" / "results")

ELIGIBLE = json.loads((RES / "block01" / "eligible_hybrid_indices.json").read_text())
AUDIT = json.loads((RES / "block01" / "q01_data_audit_summary.json").read_text())
REVERSE_PAIRS = AUDIT["reverse_pairs"]


def load_jsonl(rel: str) -> List[dict]:
    path = RES / rel
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sel(rows: Iterable[dict], **f) -> List[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in f.items())]


def per_item(rows: Iterable[dict], field: str = "delta_margin") -> Dict[int, float]:
    """One value per item index; multiple rows per index are averaged (mirrors seeds handling)."""
    acc: Dict[int, List[float]] = {}
    for r in rows:
        acc.setdefault(int(r["index"]), []).append(float(r[field]))
    return {i: float(np.mean(v)) for i, v in acc.items()}


def clusters_for(indices: Sequence[int], pairs=REVERSE_PAIRS) -> Dict[int, int]:
    cl = {i: i for i in indices}
    for a, b in pairs:
        if a in cl and b in cl:
            cl[b] = cl[a]
    return cl


def cboot(values: Dict[int, float], n_boot: int = 10000, seed: int = 0, pairs=REVERSE_PAIRS):
    """Cluster bootstrap of the pooled mean. Returns (mean, lo95, hi95, n_items, n_clusters)."""
    idx = sorted(values)
    cl = clusters_for(idx, pairs)
    groups: Dict[int, List[float]] = {}
    for i in idx:
        groups.setdefault(cl[i], []).append(values[i])
    keys = list(groups)
    gvals = [np.asarray(groups[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        pooled = np.concatenate([gvals[k] for k in pick])
        draws[b] = pooled.mean()
    allv = np.concatenate(gvals)
    return float(allv.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), len(idx), len(keys)


def paired(a: Dict[int, float], b: Dict[int, float], **kw):
    shared = sorted(set(a) & set(b))
    diff = {i: a[i] - b[i] for i in shared}
    m, lo, hi, n, nc = cboot(diff, **kw)
    wins = sum(1 for i in shared if diff[i] > 0) / len(shared)
    return {"n": n, "nclus": nc, "mean": m, "lo": lo, "hi": hi, "median": float(np.median(list(diff.values()))), "win": wins}


def fmt(d) -> str:
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n={d['n']}, clusters={d['nclus']}, median {d['median']:+.3f}, win {d['win']:.2f})"


def rate(rows, field="top1_is_swap"):
    v = [bool(r[field]) for r in rows]
    return float(np.mean(v)) if v else float("nan"), len(v)


def med(rows, field):
    return float(np.median([float(r[field]) for r in rows]))


def mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows]))


def spearman(x, y):
    from math import isnan

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = _rank(x)
    ry = _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a):
    order = a.argsort()
    ranks = np.empty(len(a), float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def check_items(rows, label):
    idx = sorted({int(r["index"]) for r in rows})
    msg = ""
    if idx != sorted(ELIGIBLE):
        msg = f"  !! {label}: item set differs from eligible-59 (n={len(idx)}; missing={sorted(set(ELIGIBLE)-set(idx))[:10]}, extra={sorted(set(idx)-set(ELIGIBLE))[:10]})"
    return msg
