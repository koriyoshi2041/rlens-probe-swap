"""CPU tests for the cluster bootstrap helpers (previously untested, flagged by the code audit)."""
from __future__ import annotations

import numpy as np

from rlens.analysis import bootstrap_mean, cluster_map, fact_pairs, paired_difference


def test_cluster_map_is_a_transitive_closure():
    # chained pairs (1,2) and (3,2): all three must share one cluster id
    clusters = cluster_map([1, 2, 3, 4], [[1, 2], [3, 2]])
    assert clusters[1] == clusters[2] == clusters[3]
    assert clusters[4] != clusters[1]
    # order of the pairs must not matter
    other = cluster_map([1, 2, 3, 4], [[3, 2], [1, 2]])
    assert other[1] == other[2] == other[3]


def test_cluster_map_ignores_pairs_outside_the_index_set():
    clusters = cluster_map([1, 2], [[1, 9], [2, 3]])
    assert clusters[1] != clusters[2]


def test_fact_pairs_merge_reverse_and_same_direction_duplicates():
    items = {
        0: {"intermediate": "France", "swap_to": "Italy"},
        1: {"intermediate": "Italy", "swap_to": "France"},  # reverse
        2: {"intermediate": "France", "swap_to": "Italy"},  # same direction, different relation
        3: {"intermediate": "Canada", "swap_to": "France"},
    }
    pairs = fact_pairs(items)
    clusters = cluster_map(list(items), pairs)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[0]
    assert len(set(clusters.values())) == 2


def test_bootstrap_mean_point_estimate_and_ci_are_sane():
    rng = np.random.default_rng(1)
    values = {i: float(v) for i, v in enumerate(rng.normal(2.0, 1.0, 60))}
    clusters = cluster_map(list(values), [])
    mean, lo, hi = bootstrap_mean(values, clusters, n_boot=2000)
    assert abs(mean - np.mean(list(values.values()))) < 1e-9
    assert lo < mean < hi
    assert 0.1 < (hi - lo) < 1.0


def test_clustering_widens_the_ci_when_items_within_a_cluster_agree():
    # 30 clusters of two identical values: the effective n is 30, so the CI must be wider than with 60 clusters
    values = {}
    rng = np.random.default_rng(2)
    for c in range(30):
        v = float(rng.normal(0.0, 1.0))
        values[2 * c] = v
        values[2 * c + 1] = v
    independent = cluster_map(list(values), [])
    paired = cluster_map(list(values), [[2 * c, 2 * c + 1] for c in range(30)])
    _, lo_i, hi_i = bootstrap_mean(values, independent, n_boot=3000)
    _, lo_p, hi_p = bootstrap_mean(values, paired, n_boot=3000)
    assert (hi_p - lo_p) > (hi_i - lo_i)


def test_paired_difference_uses_shared_items_only():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {1: 1.0, 2: 1.0, 3: 100.0}
    out = paired_difference(a, b, cluster_map([0, 1, 2, 3], []), n_boot=500)
    assert out["n"] == 2
    assert abs(out["mean_diff"] - 1.5) < 1e-9
    assert out["win_rate"] == 1.0
