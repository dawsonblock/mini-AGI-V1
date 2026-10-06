"""Counterfactual authority for expert-growth experiments.

Production growth should be promoted because it beats a matched no-growth
control, not merely because validation happened to improve after an expert was
born.  v4.2 retains the scalar decision helper and adds paired task/world
qualification with a deterministic bootstrap confidence gate.
"""
from dataclasses import dataclass, asdict

from .stats import paired_bootstrap


@dataclass
class GrowthArm:
    before: float
    after: float
    old_domain_regression: float = 0.0
    params_added: int = 0

    @property
    def gain(self):
        return float(self.before) - float(self.after)


@dataclass
class GrowthDecision:
    promote: bool
    causal_gain: float
    candidate_gain: float
    control_gain: float
    reason: str
    paired_evidence: dict | None = None

    def as_dict(self): return asdict(self)


def decide(control: GrowthArm, candidate: GrowthArm, min_causal_gain=0.002,
           max_old_domain_regression=0.02):
    causal = candidate.gain - control.gain
    if candidate.old_domain_regression > max_old_domain_regression:
        return GrowthDecision(False, causal, candidate.gain, control.gain,
                              f"old-domain regression {candidate.old_domain_regression:.4f} exceeds {max_old_domain_regression:.4f}")
    if causal < min_causal_gain:
        return GrowthDecision(False, causal, candidate.gain, control.gain,
                              f"paired causal gain {causal:.4f} below {min_causal_gain:.4f}")
    return GrowthDecision(True, causal, candidate.gain, control.gain,
                          f"paired causal gain {causal:.4f} passed with regression {candidate.old_domain_regression:.4f}")


def decide_paired(control_scores, candidate_scores, *, higher_is_better=True,
                  min_causal_gain=0.0, confidence=0.95, min_n=8, seed=0,
                  old_domain_regression=0.0, max_old_domain_regression=0.02):
    """Qualify growth on matched tasks/seeds with a confidence lower bound.

    ``control_scores`` and ``candidate_scores`` must be measurements from the
    same tasks/worlds/seeds in the same order.  Losses may be supplied by
    setting ``higher_is_better=False``; they are negated before qualification.
    """
    c = [float(x) for x in control_scores]
    a = [float(x) for x in candidate_scores]
    if not higher_is_better:
        c = [-x for x in c]
        a = [-x for x in a]
    ev = paired_bootstrap(a, c, confidence=confidence, seed=seed)
    if old_domain_regression > max_old_domain_regression:
        return GrowthDecision(
            False, ev.mean_delta, ev.mean_delta, 0.0,
            f"old-domain regression {old_domain_regression:.4f} exceeds {max_old_domain_regression:.4f}",
            paired_evidence=ev.as_dict())
    if ev.n < int(min_n):
        return GrowthDecision(
            False, ev.mean_delta, ev.mean_delta, 0.0,
            f"paired evidence n={ev.n} below required {int(min_n)}",
            paired_evidence=ev.as_dict())
    if ev.lcb < float(min_causal_gain):
        return GrowthDecision(
            False, ev.mean_delta, ev.mean_delta, 0.0,
            f"paired {confidence:.1%} lower bound {ev.lcb:.6g} below {float(min_causal_gain):.6g}",
            paired_evidence=ev.as_dict())
    return GrowthDecision(
        True, ev.mean_delta, ev.mean_delta, 0.0,
        f"paired mean gain {ev.mean_delta:.6g}; {confidence:.1%} lower bound {ev.lcb:.6g} passed",
        paired_evidence=ev.as_dict())
