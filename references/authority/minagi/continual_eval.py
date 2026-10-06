"""Continual-learning evaluation primitives.

These metrics make release claims falsifiable: a new capability gain is not a
success if unrelated old capabilities regress beyond policy limits.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import math


@dataclass(frozen=True)
class ContinualMetrics:
    old_mean_before: float
    old_mean_after: float
    old_domain_regression: float
    new_mean_before: float
    new_mean_after: float
    new_capability_gain: float
    backward_transfer: float
    forgetting_max: float
    retained_fraction: float

    def as_dict(self): return asdict(self)


def _mean(xs):
    xs=list(xs)
    return sum(float(x) for x in xs)/max(1,len(xs))


def continual_metrics(old_before: Mapping[str,float], old_after: Mapping[str,float],
                      new_before: Mapping[str,float], new_after: Mapping[str,float], *,
                      retention_tolerance: float = 0.02) -> ContinualMetrics:
    if set(old_before) != set(old_after):
        raise ValueError("old-domain task sets differ")
    if set(new_before) != set(new_after):
        raise ValueError("new-domain task sets differ")
    ob={k:float(v) for k,v in old_before.items()}; oa={k:float(v) for k,v in old_after.items()}
    nb={k:float(v) for k,v in new_before.items()}; na={k:float(v) for k,v in new_after.items()}
    old_d=[oa[k]-ob[k] for k in ob]
    new_d=[na[k]-nb[k] for k in nb]
    forgetting=max([max(0.0,-d) for d in old_d], default=0.0)
    retained=sum(1 for d in old_d if d >= -float(retention_tolerance))/max(1,len(old_d))
    return ContinualMetrics(
        old_mean_before=_mean(ob.values()), old_mean_after=_mean(oa.values()),
        old_domain_regression=max(0.0, _mean(ob.values())-_mean(oa.values())),
        new_mean_before=_mean(nb.values()), new_mean_after=_mean(na.values()),
        new_capability_gain=_mean(new_d), backward_transfer=_mean(old_d),
        forgetting_max=float(forgetting), retained_fraction=float(retained),
    )


def temporal_fact_accuracy(cases: Sequence[dict], predictor) -> dict:
    """Evaluate current-vs-historical fact discrimination.

    Each case: ``query``, ``as_of``, ``expected``. ``predictor(query, as_of)``
    returns a string/value. This is intentionally model-agnostic.
    """
    if not cases: return {"n":0,"accuracy":0.0}
    ok=0
    for c in cases:
        got=predictor(c["query"], c.get("as_of"))
        ok += int(str(got).strip() == str(c["expected"]).strip())
    return {"n":len(cases),"accuracy":ok/len(cases)}
