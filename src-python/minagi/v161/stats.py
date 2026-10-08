"""Shared deterministic statistics for campaign qualification.

bootstrap_ci is used identically by the campaign runner (RESULT.json)
and the independent qualifier (recomputed from receipts), so the CI in
the persisted record is byte-identical to the independently derived one.
The resampling RNG is seeded — same inputs always yield the same CI.
"""
from __future__ import annotations

import random


def bootstrap_ci(samples, resamples: int, alpha: float,
                 rng_seed: int = 20241006) -> dict:
    """Percentile bootstrap CI for the mean of `samples`.

    Returns {"lower","upper","mean","n","resamples","alpha","rng_seed"}.
    Deterministic: resampling is driven by random.Random(rng_seed).
    """
    xs = [float(x) for x in samples]
    if not xs:
        raise ValueError("bootstrap_ci requires at least one sample")
    rng = random.Random(rng_seed)
    n = len(xs)
    means = sorted(
        sum(xs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(int(resamples)))
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return {"lower": lo, "upper": hi, "mean": sum(xs) / n, "n": n,
            "resamples": int(resamples), "alpha": float(alpha),
            "rng_seed": int(rng_seed)}


def cluster_bootstrap_ci(samples, clusters, resamples: int, alpha: float,
                         rng_seed: int = 20241006) -> dict:
    """Cluster bootstrap: resample CLUSTERS (e.g. task families) with
    replacement and pool their members — the honest uncertainty unit
    when rows generated from the same template are correlated.

    samples[i] pairs with clusters[i]. Deterministic rng_seed as above.
    Returns bootstrap_ci-shaped dict plus n_clusters.
    """
    xs = [float(x) for x in samples]
    cs = list(clusters)
    if len(xs) != len(cs):
        raise ValueError("samples and clusters must align")
    if not xs:
        raise ValueError("cluster_bootstrap_ci requires samples")
    by_cluster: dict = {}
    for x, c in zip(xs, cs):
        by_cluster.setdefault(str(c), []).append(x)
    units = sorted(by_cluster)
    rng = random.Random(rng_seed)
    n_u = len(units)
    means = []
    for _ in range(int(resamples)):
        drawn = [units[rng.randrange(n_u)] for _ in range(n_u)]
        pooled = [x for u in drawn for x in by_cluster[u]]
        means.append(sum(pooled) / len(pooled))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return {"lower": lo, "upper": hi, "mean": sum(xs) / len(xs),
            "n": len(xs), "n_clusters": n_u,
            "resamples": int(resamples), "alpha": float(alpha),
            "rng_seed": int(rng_seed)}


def false_activation_rate(base_outputs: dict, arm_outputs: dict,
                          golds: dict, scorer) -> dict:
    """Rows where adaptation fired in error: baseline (L1) produced the
    correct answer and the adapted arm changed its output to an
    incorrect one — learned behavior applying to a non-target task.

    base_outputs / arm_outputs / golds: {task_id: text} over the same
    hidden partition. scorer(pred, gold) -> float; a row scores 1.0-ish
    correct. Returns {"n_pairs", "n_false_activation",
    "false_activation_rate", "changed_ids"}."""
    ids = sorted(set(base_outputs) & set(arm_outputs) & set(golds))
    hits = []
    for i in ids:
        if arm_outputs[i] == base_outputs[i]:
            continue
        if float(scorer(base_outputs[i], golds[i])) >= 1.0 \
                and float(scorer(arm_outputs[i], golds[i])) < 1.0:
            hits.append(i)
    return {"n_pairs": len(ids), "n_false_activation": len(hits),
            "false_activation_rate": (len(hits) / len(ids)) if ids else 0.0,
            "changed_ids": hits}
