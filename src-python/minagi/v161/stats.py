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
