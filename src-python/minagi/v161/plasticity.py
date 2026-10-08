"""Governed adaptive plasticity (Phase 7 / REPAIR-037..041 + Phase 5).

The system must NOT train weights for every unfamiliar task. This
module implements the decision layer: diagnose the observed failure,
select the cheapest plausible mechanism, and emit a proposal that can
only become an executed change through the existing preregistration →
execution → independent qualification → promotion chain.

Nothing here authorizes a change — it produces signed, digest-bound
proposals for the plan authority. Promotion still requires the
qualification record the proposal claimed.

v16.6.0 (Phase 5) — evidence-driven selection. Beyond the deterministic
ladder, the controller ranks candidate mechanisms by a cost-aware
objective

    U = ΔQ - λ_C·C - λ_R·R - λ_L·L

where ΔQ is estimated quality improvement, C resource cost, R
regression risk, and L additional inference latency. The weights are
externally configured (`ObjectiveWeights`, frozen per campaign — never
rewritten by the agent being evaluated). Hard safety constraints
(regression-risk cap, confidence floor) are applied *before* ranking
and cannot be traded for utility. When no candidate clears the floor
the controller selects a diagnostic experiment instead of adapting on
an uncertain cause. Every attempt produces a machine-verifiable
`AttemptReceipt` chaining

    FailureEvidence → MechanismProposal → AttemptReceipt → EvaluationBundle.

Determinism: diagnosis and selection are pure functions of their
inputs — identical inputs produce identical decisions.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from egai.common.canonical import digest, validate_digest


class FailureKind(str, Enum):
    MISSING_INFORMATION = "missing_information"
    REPEATABLE_PROCEDURE = "repeatable_procedure"
    PERSISTENT_DEFICIENCY = "persistent_deficiency"
    HARNESS_DEFECT = "harness_defect"


class Mechanism(str, Enum):
    RETRIEVAL = "retrieval"                    # cheapest: no state change
    SKILL = "skill_compile"                    # reusable procedure, no weights
    WEIGHTS = "weight_adaptation"              # LoRA — model state changes
    HARNESS = "harness_change"                 # most expensive/risky


# Cheapest-first ordering: a mechanism may be proposed only after every
# cheaper mechanism has been tried and shown insufficient (or ruled out
# by the diagnosis). This is the cost-ladder the plan requires.
MECHANISM_LADDER = (Mechanism.RETRIEVAL, Mechanism.SKILL,
                    Mechanism.WEIGHTS, Mechanism.HARNESS)

_KIND_TO_MECHANISM = {
    FailureKind.MISSING_INFORMATION: Mechanism.RETRIEVAL,
    FailureKind.REPEATABLE_PROCEDURE: Mechanism.SKILL,
    FailureKind.PERSISTENT_DEFICIENCY: Mechanism.WEIGHTS,
    FailureKind.HARNESS_DEFECT: Mechanism.HARNESS,
}


@dataclass(frozen=True)
class FailureEvidence:
    """Digest-bound summary of an observed task failure. Content lives
    in the referenced evidence artifacts; this record carries only the
    fields the diagnosis rules inspect."""
    task_digest: str
    failure_signature: str          # canonical failure pattern digest
    retrieval_found: bool           # did retrieval return relevant evidence
    retrieval_helped: bool          # did the retrieved evidence fix it
    repeats_prior_signature: bool   # same signature seen on earlier tasks
    harness_error: bool             # tool-call/format/transport failure
    schema: str = "mini-agi-v16.7-failure-evidence-v1"

    def __post_init__(self):
        validate_digest(self.task_digest)
        validate_digest(self.failure_signature)

    @property
    def digest(self) -> str:
        return digest(self)


def diagnose(ev: FailureEvidence) -> tuple[FailureKind, Mechanism]:
    """Deterministic diagnosis. Harness defects outrank everything —
    a broken tool path cannot be fixed by retrieval or weights.
    Retrieval evidence that resolves the failure stays the cheapest
    mechanism; unhelpful retrieval on a repeating signature escalates
    to skill compilation; only a deficiency that survives cheaper
    mechanisms escalates to weight adaptation."""
    if ev.harness_error:
        return FailureKind.HARNESS_DEFECT, Mechanism.HARNESS
    if ev.retrieval_found and ev.retrieval_helped:
        return FailureKind.MISSING_INFORMATION, Mechanism.RETRIEVAL
    if ev.repeats_prior_signature and not ev.retrieval_helped:
        return FailureKind.REPEATABLE_PROCEDURE, Mechanism.SKILL
    return FailureKind.PERSISTENT_DEFICIENCY, Mechanism.WEIGHTS


@dataclass(frozen=True)
class PlasticityProposal:
    """A proposal, not an authorization. `mechanism` is the cheapest
    mechanism the diagnosis did not rule out; `prior_attempts` binds
    the cheaper mechanisms already shown insufficient — a proposal
    skipping ladder steps without recorded attempts is invalid."""
    proposal_id: str
    evidence_digest: str
    failure_kind: str
    mechanism: str
    candidate_config_digest: str   # config of the proposed change
    baseline_digest: str           # frozen comparator the candidate beats
    prior_attempts: tuple[str, ...]  # digests of cheaper mechanism attempts
    schema: str = "mini-agi-v16.7-plasticity-proposal-v1"

    def __post_init__(self):
        validate_digest(self.evidence_digest)
        validate_digest(self.candidate_config_digest)
        validate_digest(self.baseline_digest)
        for d in self.prior_attempts:
            validate_digest(d)
        if self.failure_kind not in {k.value for k in FailureKind}:
            raise ValueError("unknown failure_kind")
        mech = Mechanism(self.mechanism)
        required_prior = MECHANISM_LADDER[:MECHANISM_LADDER.index(mech)]
        if len(self.prior_attempts) != len(required_prior):
            raise ValueError(
                f"{mech.value} proposal requires {len(required_prior)} "
                f"prior cheaper-mechanism attempts, got "
                f"{len(self.prior_attempts)}")

    @property
    def digest(self) -> str:
        return digest(self)


def propose(ev: FailureEvidence, *, proposal_id: str,
            candidate_config_digest: str, baseline_digest: str,
            prior_attempts: tuple[str, ...] = ()) -> PlasticityProposal:
    kind, mech = diagnose(ev)
    return PlasticityProposal(
        proposal_id=proposal_id, evidence_digest=ev.digest,
        failure_kind=kind.value, mechanism=mech.value,
        candidate_config_digest=candidate_config_digest,
        baseline_digest=baseline_digest,
        prior_attempts=prior_attempts)


# ---------------------------------------------------------------------------
# Dynamic-rank LoRA — experimental mechanism (REPAIR-038..041).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicLoraPolicyV1:
    """Signed policy for dynamic-rank LoRA. Rank is allocated, not
    unbounded: every allocation is clamped to the per-task and total
    budgets; growth happens only in rank_growth_step increments; when
    activation_protection is set, rank may not grow while the
    false-activation gate is breached (learned behavior firing off-task
    means more capacity must not be allocated)."""
    base_rank: int
    max_rank: int
    rank_growth_step: int
    per_task_rank_budget: int
    total_rank_budget: int
    activation_protection: bool = True
    schema: str = "mini-agi-v16.7-dynamic-lora-policy-v1"

    def __post_init__(self):
        if not 0 < self.base_rank <= self.max_rank:
            raise ValueError("require 0 < base_rank <= max_rank")
        if self.rank_growth_step <= 0:
            raise ValueError("rank_growth_step must be > 0")
        if self.per_task_rank_budget <= 0 or self.total_rank_budget <= 0:
            raise ValueError("rank budgets must be > 0")
        if self.per_task_rank_budget > self.max_rank:
            raise ValueError("per-task budget cannot exceed max_rank")

    @property
    def digest(self) -> str:
        return digest(self)


class RankAllocator:
    """Deterministic rank allocator under a DynamicLoraPolicyV1.

    allocate() returns the rank granted to a task — never above the
    per-task budget or what remains of the total budget. grow() is the
    only way an existing allocation increases, and only in
    rank_growth_step increments while activation_protection allows it
    (caller supplies the current false-activation breach flag).
    """

    def __init__(self, policy: DynamicLoraPolicyV1):
        self.policy = policy
        self._alloc: dict[str, int] = {}
        self._spent = 0

    def allocate(self, task_id: str, requested: int = 0) -> int:
        if task_id in self._alloc:
            return self._alloc[task_id]
        want = requested or self.policy.base_rank
        granted = min(want, self.policy.per_task_rank_budget,
                      self.policy.max_rank,
                      self.policy.total_rank_budget - self._spent)
        if granted <= 0:
            raise RuntimeError("total rank budget exhausted")
        self._alloc[task_id] = granted
        self._spent += granted
        return granted

    def grow(self, task_id: str, *, false_activation_breach: bool = False) -> int:
        """Increase an existing allocation by at most rank_growth_step,
        bounded by the per-task budget and by what the total budget can
        still fund *on top of the current rank*:

            headroom = total_rank_budget - spent
            maximum_allowed = min(per_task_rank_budget, max_rank,
                                  current_rank + headroom)

        FIX-002: clamping to `headroom` alone let a task's rank fall
        (even to zero) when the total budget was exhausted, violating
        the no-decrease invariant. Growth is now strictly
        non-decreasing; when nothing can be granted, the allocation is
        returned unchanged."""
        if task_id not in self._alloc:
            raise KeyError(f"no allocation for task {task_id}")
        cur = self._alloc[task_id]
        if self.policy.activation_protection and false_activation_breach:
            return cur  # refuse growth while misfiring
        headroom = self.policy.total_rank_budget - self._spent
        maximum_allowed = min(self.policy.per_task_rank_budget,
                              self.policy.max_rank,
                              cur + headroom)
        new = min(cur + self.policy.rank_growth_step, maximum_allowed)
        if new <= cur:  # blocked growth leaves state unchanged
            return cur
        self._alloc[task_id] = new
        self._spent += new - cur
        return new

    @property
    def allocations(self) -> dict[str, int]:
        return dict(self._alloc)

    @property
    def spent(self) -> int:
        return self._spent


# ---------------------------------------------------------------------------
# v16.6.0 Phase 5 — evidence-driven, cost-aware mechanism selection.
# ---------------------------------------------------------------------------

DIAGNOSTIC_EXPERIMENT = "diagnostic_experiment"


class AttemptOutcome(str, Enum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    FAILED = "failed"
    INVALID = "invalid"          # evidence invalid / policy violation


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class MechanismEstimate:
    """Cost-aware estimate for one candidate mechanism. Estimates are
    external inputs (from recorded attempts and measured costs); the
    controller only ranks them — it never invents them."""
    mechanism: str
    predicted_gain: float          # ΔQ: estimated quality improvement
    expected_cost: float           # C: resource cost (normalized)
    confidence: float              # [0,1] in the estimate itself
    regression_risk: float         # R: [0,1]
    latency_cost: float            # L: additional inference latency
    rationale: str = ""            # human-auditable provenance

    def __post_init__(self):
        Mechanism(self.mechanism)
        _finite("predicted_gain", self.predicted_gain)
        _finite("expected_cost", self.expected_cost)
        _finite("latency_cost", self.latency_cost)
        if self.expected_cost < 0 or self.latency_cost < 0:
            raise ValueError("expected_cost/latency_cost must be >= 0")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if not 0.0 <= float(self.regression_risk) <= 1.0:
            raise ValueError("regression_risk must be in [0,1]")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class ObjectiveWeights:
    """U = ΔQ - λ_C·C - λ_R·R - λ_L·L. Externally configured and frozen
    for each campaign; the agent under evaluation never rewrites it."""
    lambda_cost: float = 1.0
    lambda_risk: float = 1.0
    lambda_latency: float = 1.0
    schema: str = "mini-agi-v16.6-objective-weights-v1"

    def __post_init__(self):
        for name in ("lambda_cost", "lambda_risk", "lambda_latency"):
            if _finite(name, getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")

    def utility(self, est: MechanismEstimate) -> float:
        return (float(est.predicted_gain)
                - self.lambda_cost * float(est.expected_cost)
                - self.lambda_risk * float(est.regression_risk)
                - self.lambda_latency * float(est.latency_cost))

    @property
    def digest(self) -> str:
        return digest(self)


def rank_candidates(estimates, weights: ObjectiveWeights, *,
                    max_regression_risk: float | None = None,
                    min_confidence: float = 0.0):
    """Rank candidate estimates by U descending, with a deterministic
    tie-break (cheapest-first ladder order, then mechanism name).

    Hard safety constraints are applied BEFORE ranking and are never
    traded for utility: candidates whose regression risk exceeds
    `max_regression_risk`, or whose confidence is below
    `min_confidence`, are excluded with a reason. Returns
    (ranked, excluded)."""
    ranked, excluded = [], []
    for est in estimates:
        if max_regression_risk is not None \
                and float(est.regression_risk) > max_regression_risk:
            excluded.append((est.mechanism,
                             "regression_risk above hard cap"))
        elif float(est.confidence) < min_confidence:
            excluded.append((est.mechanism,
                             "confidence below floor"))
        else:
            ranked.append(est)
    ranked.sort(key=lambda e: (-weights.utility(e),
                               MECHANISM_LADDER.index(Mechanism(e.mechanism)),
                               e.mechanism))
    return ranked, excluded


@dataclass(frozen=True)
class MechanismDecision:
    """Machine-verifiable record of one selection: what was chosen, why,
    what was excluded, and the objective value of the winner."""
    failure_kind: str
    chosen: str                    # Mechanism value or DIAGNOSTIC_EXPERIMENT
    utility: float
    ranked: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]
    rationale: str
    schema: str = "mini-agi-v16.6-mechanism-decision-v1"

    def __post_init__(self):
        if self.failure_kind not in {k.value for k in FailureKind}:
            raise ValueError("unknown failure_kind")
        if self.chosen != DIAGNOSTIC_EXPERIMENT:
            Mechanism(self.chosen)
        _finite("utility", self.utility)
        for name in self.ranked:
            Mechanism(name)
        for name, _reason in self.excluded:
            Mechanism(name)

    @property
    def digest(self) -> str:
        return digest(self)


def select_mechanism(ev: FailureEvidence, estimates, weights: ObjectiveWeights,
                     *, prior_attempts: tuple[str, ...] = (),
                     max_regression_risk: float | None = None,
                     min_confidence: float = 0.5) -> MechanismDecision:
    """Evidence-driven mechanism selection.

    The cheapest-first ladder still binds: a WEIGHTS estimate is only
    admissible when the cheaper-mechanism attempts it presupposes are
    recorded (retrieval + skill — the same rule `PlasticityProposal`
    enforces). Safety constraints are applied before ranking; when no
    candidate clears the confidence floor, a diagnostic experiment is
    selected instead of adapting on an uncertain cause."""
    kind, _mech = diagnose(ev)
    required = MECHANISM_LADDER[
        :MECHANISM_LADDER.index(Mechanism.WEIGHTS)]
    for est in estimates:
        if Mechanism(est.mechanism) is Mechanism.WEIGHTS \
                and len(prior_attempts) != len(required):
            raise ValueError(
                "weight-adaptation estimates require recorded attempts at "
                f"{[m.value for m in required]} first (got "
                f"{len(prior_attempts)} prior attempts)")
    ranked, excluded = rank_candidates(
        estimates, weights, max_regression_risk=max_regression_risk,
        min_confidence=min_confidence)
    if not ranked:
        return MechanismDecision(
            failure_kind=kind.value, chosen=DIAGNOSTIC_EXPERIMENT,
            utility=0.0, ranked=(),
            excluded=tuple(excluded),
            rationale="no candidate cleared the confidence floor / hard "
                      "safety cap — run a diagnostic experiment instead of "
                      "adapting on an uncertain cause")
    best = ranked[0]
    return MechanismDecision(
        failure_kind=kind.value, chosen=best.mechanism,
        utility=weights.utility(best),
        ranked=tuple(e.mechanism for e in ranked),
        excluded=tuple(excluded),
        rationale=f"highest objective utility {weights.utility(best):.6f} "
                  f"under frozen weights {weights.digest}")


@dataclass(frozen=True)
class AttemptReceipt:
    """Machine-verifiable record of one mechanism attempt, chaining
    FailureEvidence → MechanismProposal → AttemptReceipt →
    EvaluationBundle by digest. Observed gain/cost come from the
    independent evaluation, never from the proposal itself."""
    attempt_id: str
    proposal_digest: str
    mechanism: str
    evidence_digest: str
    outcome: str                   # AttemptOutcome value
    observed_gain: float
    cost: float
    evaluation_bundle_digest: str
    schema: str = "mini-agi-v16.6-attempt-receipt-v1"

    def __post_init__(self):
        if not self.attempt_id:
            raise ValueError("attempt_id required")
        for field_name, value in (("proposal_digest", self.proposal_digest),
                                  ("evidence_digest", self.evidence_digest),
                                  ("evaluation_bundle_digest",
                                   self.evaluation_bundle_digest)):
            validate_digest(value)
        if self.mechanism != DIAGNOSTIC_EXPERIMENT:
            Mechanism(self.mechanism)
        if self.outcome not in {o.value for o in AttemptOutcome}:
            raise ValueError("unknown attempt outcome")
        _finite("observed_gain", self.observed_gain)
        if _finite("cost", self.cost) < 0:
            raise ValueError("cost must be >= 0")

    @property
    def digest(self) -> str:
        return digest(self)
