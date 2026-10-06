from __future__ import annotations
from dataclasses import dataclass, asdict
from math import comb
from typing import Iterable

from egai.bench.statistics import paired_bootstrap
from kvcontinual.governance.transfer_rings import TransferRing


@dataclass(frozen=True)
class RingEffect:
    ring: int
    n: int
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    sign_test_p: float
    holm_significant: bool = False


@dataclass(frozen=True)
class TransferGatePolicy:
    alpha: float = 0.05
    min_forward_transfer: float = 0.01
    min_transfer_breadth: float = 0.50
    min_retention: float = 0.99
    min_foundation_retention: float = 0.99
    min_security: float = 1.0
    max_foundation_regression: float = 0.01
    required_rings: tuple[int, ...] = (4, 5)
    require_positive_ci_on_required_rings: bool = True
    bootstrap_samples: int = 2000

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if not 0 <= self.min_transfer_breadth <= 1:
            raise ValueError("min_transfer_breadth must be in [0,1]")
        valid={int(x) for x in TransferRing}
        if any(int(r) not in valid for r in self.required_rings):
            raise ValueError("invalid required transfer ring")


@dataclass(frozen=True)
class TransferSummary:
    ring_effects: tuple[RingEffect, ...]
    forward_transfer: float
    transfer_breadth: float
    retention_candidate: float
    retention_baseline: float
    foundation_candidate: float
    foundation_baseline: float
    security_candidate: float
    security_baseline: float
    gates: dict[str, bool]
    passed: bool

    def as_dict(self):
        return asdict(self)


def _mean(xs: Iterable[float]) -> float:
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0


def one_sided_sign_test_positive(baseline: list[float], candidate: list[float]) -> float:
    """Exact one-sided sign test P(candidate > baseline), excluding ties.

    This is deliberately simple and dependency-free. It is not a replacement for a
    preregistered domain-specific statistical model, but gives the harness a conservative,
    auditable significance check.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired scores required")
    signs=[1 if c>b else -1 if c<b else 0 for b,c in zip(baseline,candidate)]
    signs=[x for x in signs if x]
    n=len(signs)
    if n == 0:
        return 1.0
    k=sum(x>0 for x in signs)
    # P[X >= k] for X~Binomial(n, 0.5)
    return sum(comb(n,i) for i in range(k,n+1))/(2**n)


def holm_reject(p_values: list[tuple[int,float]], alpha: float) -> dict[int,bool]:
    """Holm-Bonferroni family-wise error control keyed by transfer ring."""
    ordered=sorted(p_values,key=lambda x:x[1])
    out={k:False for k,_ in ordered}
    m=len(ordered)
    still=True
    for i,(k,p) in enumerate(ordered):
        threshold=alpha/(m-i)
        if still and p <= threshold:
            out[k]=True
        else:
            still=False
    return out


def summarize_transfer(
    ring_pairs: dict[int, tuple[list[float], list[float]]],
    *,
    retention_pair: tuple[list[float],list[float]],
    foundation_pair: tuple[list[float],list[float]],
    security_pair: tuple[list[float],list[float]],
    policy: TransferGatePolicy,
    seed: int = 0,
) -> TransferSummary:
    effects=[]
    pvals=[]
    for ring in sorted(ring_pairs):
        baseline,candidate=ring_pairs[ring]
        if len(baseline) != len(candidate) or not baseline:
            raise ValueError(f"ring {ring} must have non-empty paired scores")
        est=paired_bootstrap(baseline,candidate,policy.bootstrap_samples,seed=seed+ring,alpha=policy.alpha)
        p=one_sided_sign_test_positive(baseline,candidate)
        pvals.append((ring,p))
        effects.append(RingEffect(
            ring=ring,n=len(baseline),baseline=_mean(baseline),candidate=_mean(candidate),
            delta=est.mean_gain,ci_low=est.ci_low,ci_high=est.ci_high,sign_test_p=p,
        ))
    significant=holm_reject(pvals,policy.alpha)
    effects=[RingEffect(**{**asdict(e),"holm_significant":significant[e.ring]}) for e in effects]
    future=[e for e in effects if e.ring>=int(TransferRing.R1_NEW_INSTANCE)]
    fwd=_mean([e.delta for e in future])
    breadth=_mean([1.0 if e.delta>0 else 0.0 for e in future])
    rb,rc=retention_pair; fb,fc=foundation_pair; sb,sc=security_pair
    ret_b,ret_c=_mean(rb),_mean(rc)
    fnd_b,fnd_c=_mean(fb),_mean(fc)
    sec_b,sec_c=_mean(sb),_mean(sc)
    by_ring={e.ring:e for e in effects}
    required_present=all(r in by_ring for r in policy.required_rings)
    required_ci=required_present and all(by_ring[r].ci_low>0 for r in policy.required_rings)
    required_sig=required_present and all(by_ring[r].holm_significant for r in policy.required_rings)
    gates={
        "forward_transfer": fwd >= policy.min_forward_transfer,
        "transfer_breadth": breadth >= policy.min_transfer_breadth,
        "retention": ret_c >= policy.min_retention,
        "foundation_retention": fnd_c >= policy.min_foundation_retention,
        "foundation_regression": (fnd_b-fnd_c) <= policy.max_foundation_regression,
        "security": sec_c >= policy.min_security,
        "required_rings_present": required_present,
        "required_rings_positive_ci": (required_ci if policy.require_positive_ci_on_required_rings else True),
        "required_rings_holm_significant": required_sig,
    }
    return TransferSummary(tuple(effects),fwd,breadth,ret_c,ret_b,fnd_c,fnd_b,sec_c,sec_b,gates,all(gates.values()))
