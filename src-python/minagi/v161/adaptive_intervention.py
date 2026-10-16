"""Evidence-driven adaptive intervention selection (v4.2.0 / Phase 6).

``capability_controller.select_intervention`` ships a hypothesis-driven
policy with a STATIC cost table and prior-only benefit estimates. This
module replaces the static assumptions with measured outcomes from
qualified experiments:

* every intervention attempt produces a versioned ``OutcomeRecord``
  carrying measured deltas (within-family, cross-family, retention
  regression), measured resources, and ``evidence_receipt_digest``;
* a record counts as *verified evidence* only when its referenced
  receipt independently verifies — an operator-supplied
  ``qualification_passed`` boolean is an outcome field, never proof;
* selection is deliberately interpretable: shrunken empirical benefit
  minus regression-risk and measured-cost penalties, plus a bounded
  UCB-style exploration bonus. Compare against ``FixedPolicy`` and
  ``RandomPolicy`` — the adaptive policy must beat both on a recorded
  or simulated outcome stream before claiming adaptive benefit (G9).

The learning controller PROPOSES interventions; it never authorizes its
own qualification. ``candidate_controlled`` stays True on every emitted
decision.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from egai.common.canonical import digest

SCHEMA = "mini-agi-v4.2-outcome-record-v1"
POLICY_SCHEMA = "mini-agi-v4.2-adaptive-decision-v1"


@dataclass(frozen=True)
class OutcomeRecord:
    """One executed intervention's measured outcome.

    ``evaluation`` fields mirror the plan's outcome model:
    within_family_delta / cross_family_delta are accuracy deltas on the
    corresponding surfaces, retention_regression is the measured
    destruction of previously-correct behavior (nonneg = damage), and
    qualification_passed records the INDEPENDENT qualifier's verdict —
    informational only; verification comes from the receipt.
    """
    intervention: str
    target_capability: str
    evaluation: Mapping[str, float | bool]
    resources: Mapping[str, float]
    evidence_receipt_digest: str
    schema: str = SCHEMA
    run_id: str = ""
    seed: int | None = None
    recorded_utc: str = ""

    def __post_init__(self) -> None:
        if not self.intervention or not self.target_capability:
            raise ValueError("intervention and target_capability required")
        ev = dict(self.evaluation)
        for k in ("within_family_delta", "cross_family_delta",
                  "retention_regression", "qualification_passed"):
            if k not in ev:
                raise ValueError(f"evaluation.{k} required")
        for k, v in ev.items():
            if k != "qualification_passed" and not isinstance(
                    v, (int, float)):
                raise ValueError(f"evaluation.{k} must be numeric")
        res = dict(self.resources)
        for k in ("training_seconds", "gpu_seconds",
                  "evaluation_seconds"):
            if k not in res or not isinstance(res[k], (int, float)):
                raise ValueError(f"resources.{k} (seconds) required")
        if not str(self.evidence_receipt_digest):
            raise ValueError("evidence_receipt_digest required")

    @property
    def benefit(self) -> float:
        """Primary benefit signal: within-family gain, with
        cross-family transfer credited at half weight."""
        ev = self.evaluation
        return float(ev["within_family_delta"]) \
            + 0.5 * float(ev["cross_family_delta"])

    @property
    def regression(self) -> float:
        return max(0.0, float(self.evaluation["retention_regression"]))

    @property
    def total_seconds(self) -> float:
        r = self.resources
        return float(r["training_seconds"]) + float(r["gpu_seconds"]) \
            + float(r["evaluation_seconds"])


def load_outcomes(path) -> list[OutcomeRecord]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if ln.strip():
            out.append(OutcomeRecord(**json.loads(ln)))
    return out


def append_outcome(path, rec: OutcomeRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(asdict(rec), sort_keys=True) + "\n")


#: receipt_verifier signature: (OutcomeRecord) -> bool. In production
#: bind ``capability_outcomes.verify_outcome_receipt`` against the run
#: dir the record's digest resolves to and require digest agreement.
ReceiptVerifier = Callable[[OutcomeRecord], bool]


def verify_outcomes(records: Iterable[OutcomeRecord],
                    receipt_verifier: ReceiptVerifier | None,
                    ) -> dict[str, list[OutcomeRecord]]:
    """Partition records into verified / unverified under the caller's
    receipt check. NO verifier supplied => everything is unverified
    (fail closed): a signature that cannot be checked is not
    verification."""
    verified: list[OutcomeRecord] = []
    unverified: list[OutcomeRecord] = []
    for r in records:
        ok = False
        if receipt_verifier is not None:
            try:
                ok = bool(receipt_verifier(r))
            except Exception:
                ok = False
        (verified if ok else unverified).append(r)
    return {"verified": verified, "unverified": unverified}


@dataclass(frozen=True)
class ArmEstimate:
    intervention: str
    n_verified: int
    mean_benefit: float
    shrunken_benefit: float
    mean_regression: float
    mean_cost_seconds: float
    explore_bonus: float
    score: float
    per_run_deltas: tuple[float, ...] = ()


class EmpiricalSelector:
    """Cost-normalized empirical selection with a bounded explore bonus.

    score_i = shrunken_benefit_i + beta * sqrt(ln(N+1)/(n_i+1))
              - risk_penalty * mean_regression_i
              - cost_weight * mean_cost_seconds_i / cost_scale

    ``shrunken_benefit`` pulls sparse arms toward ``prior_benefit``
    (default 0 — conservative) with ``prior_weight`` pseudo-observations,
    so one lucky run cannot dominate the estimate. The explore bonus is
    finite at n_i = 0 (untested arms are tried, but a tested arm with a
    real measured advantage can outrank them immediately).
    """

    def __init__(self, *, prior_benefit: float = 0.0,
                 prior_weight: float = 3.0, beta: float = 0.1,
                 risk_penalty: float = 1.0, cost_weight: float = 0.05,
                 cost_scale: float = 3600.0) -> None:
        self.prior_benefit = float(prior_benefit)
        self.prior_weight = float(prior_weight)
        self.beta = float(beta)
        self.risk_penalty = float(risk_penalty)
        self.cost_weight = float(cost_weight)
        self.cost_scale = float(cost_scale)

    def estimate(self, intervention: str,
                 verified: Sequence[OutcomeRecord],
                 total_verified: int) -> ArmEstimate:
        recs = [r for r in verified if r.intervention == intervention]
        n = len(recs)
        mean_b = sum(r.benefit for r in recs) / n if n else 0.0
        shrunk = ((n * mean_b + self.prior_weight * self.prior_benefit)
                  / (n + self.prior_weight))
        mean_r = sum(r.regression for r in recs) / n if n else 0.0
        mean_c = sum(r.total_seconds for r in recs) / n if n else 0.0
        # ln(N+2) keeps the bonus strictly positive at cold start, so an
        # untried arm outranks an unproven one; at n=0 the bonus is
        # bounded (beta * sqrt(ln 2) ~ 0.83 beta), not infinite — a
        # measured advantage beats exploration immediately.
        bonus = self.beta * math.sqrt(
            math.log(total_verified + 2) / (n + 1))
        score = (shrunk + bonus - self.risk_penalty * mean_r
                 - self.cost_weight * mean_c / self.cost_scale)
        return ArmEstimate(
            intervention=intervention, n_verified=n,
            mean_benefit=round(mean_b, 6),
            shrunken_benefit=round(shrunk, 6),
            mean_regression=round(mean_r, 6),
            mean_cost_seconds=round(mean_c, 3),
            explore_bonus=round(bonus, 6), score=round(score, 6),
            per_run_deltas=tuple(round(r.benefit, 6) for r in recs))

    def select(self, candidates: Sequence[str],
               verified: Sequence[OutcomeRecord], *,
               refuse_below: float = 0.0) -> dict:
        """Rank candidates; ``no_intervention`` anchors at score 0."""
        verified = list(verified)
        total = len(verified)
        ests = [self.estimate(c, verified, total) for c in candidates
                if c != "no_intervention"]
        best = max(ests, key=lambda e: e.score, default=None)
        selected = "no_intervention"
        if best is not None and (best.score > refuse_below
                                 or "no_intervention" not in candidates):
            # refuse_below only applies when doing nothing is actually
            # on the table; a candidate set without no_intervention
            # means the caller requires an intervention, so take the
            # argmax rather than emitting an unavailable choice
            selected = best.intervention
        return {
            "schema": POLICY_SCHEMA,
            "selected": selected,
            "candidate_controlled": True,
            "estimates": [asdict(e) for e in sorted(
                ests, key=lambda e: (-e.score, e.intervention))],
            "policy": {
                "kind": "empirical_ucb",
                "prior_benefit": self.prior_benefit,
                "prior_weight": self.prior_weight,
                "beta": self.beta,
                "risk_penalty": self.risk_penalty,
                "cost_weight": self.cost_weight,
                "cost_scale": self.cost_scale,
                "n_verified_records": total,
            },
        }

    # policy-protocol hook used by evaluate_policies / bandit_simulation
    def __call__(self, candidates: Sequence[str],
                 history: Sequence[OutcomeRecord]) -> str:
        return str(self.select(candidates, history)["selected"])


class FixedPolicy:
    """Baseline: always selects the same intervention."""

    def __init__(self, intervention: str) -> None:
        self.intervention = intervention

    def __call__(self, candidates: Sequence[str],
                 history: Sequence[OutcomeRecord]) -> str:
        return self.intervention


class RandomPolicy:
    """Baseline: uniform random selection over non-empty candidates."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def __call__(self, candidates: Sequence[str],
                 history: Sequence[OutcomeRecord]) -> str:
        pool = [c for c in candidates if c != "no_intervention"] \
            or list(candidates)
        return self._rng.choice(pool)


def bandit_simulation(true_model: Mapping[str, Mapping[str, float]],
                      policies: Mapping[str, Callable], *,
                      rounds: int = 50, seed: int = 0) -> dict:
    """Compare policies on a synthetic intervention environment.

    ``true_model`` maps intervention -> {"benefit", "regression",
    "cost_seconds", "noise_sd"}. Each round, every policy independently
    picks an arm (seeing only ITS OWN accumulated history — policies
    learn from what they tried), draws a noisy benefit, and banks the
    outcome. Reports per-policy mean realized benefit, total cost,
    total regression, and the number of distinct arms tried.
    """
    rng = random.Random(seed)
    report: dict[str, dict] = {}
    for name, policy in policies.items():
        history: list[OutcomeRecord] = []
        realized: list[float] = []
        arms = sorted(true_model)
        for t in range(rounds):
            choice = policy(arms, history)
            if choice not in true_model:
                realized.append(0.0)
                continue
            spec = true_model[choice]
            draw = rng.gauss(float(spec["benefit"]),
                             float(spec.get("noise_sd", 0.05)))
            reg = max(0.0, rng.gauss(float(spec.get("regression", 0.0)),
                                     0.01))
            cost = float(spec.get("cost_seconds", 60.0))
            history.append(OutcomeRecord(
                intervention=choice, target_capability="sim",
                evaluation={"within_family_delta": draw,
                            "cross_family_delta": 0.0,
                            "retention_regression": reg,
                            "qualification_passed": draw > 0},
                resources={"training_seconds": cost, "gpu_seconds": 0.0,
                           "evaluation_seconds": 0.0},
                evidence_receipt_digest=digest(
                    {"sim": True, "round": t, "arm": choice})))
            realized.append(draw)
        report[name] = {
            "mean_realized_benefit": round(
                sum(realized) / len(realized), 6) if realized else 0.0,
            "cumulative_benefit": round(sum(realized), 4),
            "total_cost_seconds": round(
                sum(r.total_seconds for r in history), 1),
            "total_regression": round(
                sum(r.regression for r in history), 4),
            "distinct_arms_tried": len({r.intervention
                                        for r in history}),
        }
    return {"schema": "mini-agi-v4.2-policy-comparison-v1",
            "rounds": rounds, "policies": report}


def decision_digest(dec: Mapping) -> str:
    d = dict(dec)
    d.pop("digest", None)
    return digest(d)
