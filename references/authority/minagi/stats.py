"""Small deterministic statistics helpers for qualification gates.

These routines are intentionally dependency-light. They operate on paired
measurements because candidate/control comparisons are strongest when both arms
are evaluated on the same tasks/worlds/seeds. The bootstrap is deterministic
for a supplied seed so qualification records can be reproduced exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import random


@dataclass(frozen=True)
class PairedEvidence:
    n: int
    mean_delta: float
    median_delta: float
    lcb: float
    ucb: float
    p_signflip: float
    confidence: float
    seed: int

    def as_dict(self):
        return asdict(self)


def _clean(xs):
    out = [float(x) for x in xs]
    if not out:
        raise ValueError("paired evidence requires at least one observation")
    if any(not math.isfinite(x) for x in out):
        raise ValueError("paired evidence contains non-finite values")
    return out


def paired_deltas(candidate, control):
    a = _clean(candidate)
    b = _clean(control)
    if len(a) != len(b):
        raise ValueError("candidate/control paired measurements must have equal length")
    return [x - y for x, y in zip(a, b)]


def _quantile(sorted_values, q):
    if not sorted_values:
        raise ValueError("empty quantile")
    q = min(1.0, max(0.0, float(q)))
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    p = q * (len(sorted_values) - 1)
    lo = int(math.floor(p)); hi = int(math.ceil(p))
    if lo == hi:
        return float(sorted_values[lo])
    w = p - lo
    return float(sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w)


def paired_bootstrap(candidate, control, *, confidence=0.95, resamples=5000,
                     seed=0, signflip_samples=20000):
    """Return deterministic paired-bootstrap CI and a two-sided sign-flip p.

    Delta is candidate-control, so positive values mean the candidate is better
    when the supplied metric is defined as "higher is better". Callers using a
    loss should negate it before passing values here.
    """
    d = paired_deltas(candidate, control)
    n = len(d)
    confidence = float(confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")
    resamples = max(1, int(resamples))
    rng = random.Random(int(seed))
    means = []
    for _ in range(resamples):
        means.append(sum(d[rng.randrange(n)] for _j in range(n)) / n)
    means.sort()
    alpha = 1.0 - confidence
    lcb = _quantile(means, alpha / 2.0)
    ucb = _quantile(means, 1.0 - alpha / 2.0)

    obs = abs(sum(d) / n)
    # Exact enumeration is cheap up to 16 pairs; beyond that use a deterministic
    # Monte-Carlo sign-flip test. +1 smoothing prevents a zero p-value.
    if n <= 16:
        total = 1 << n
        extreme = 0
        for mask in range(total):
            m = sum((x if (mask >> i) & 1 else -x) for i, x in enumerate(d)) / n
            if abs(m) >= obs - 1e-15:
                extreme += 1
        p = extreme / total
    else:
        trials = max(1000, int(signflip_samples))
        extreme = 0
        for _ in range(trials):
            m = sum((x if rng.random() < 0.5 else -x) for x in d) / n
            if abs(m) >= obs - 1e-15:
                extreme += 1
        p = (extreme + 1.0) / (trials + 1.0)

    sd = sorted(d)
    return PairedEvidence(
        n=n,
        mean_delta=float(sum(d) / n),
        median_delta=_quantile(sd, 0.5),
        lcb=float(lcb),
        ucb=float(ucb),
        p_signflip=float(p),
        confidence=confidence,
        seed=int(seed),
    )
