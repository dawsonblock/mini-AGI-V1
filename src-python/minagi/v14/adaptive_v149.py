from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import Iterable, Sequence

from egai.common.canonical import digest, validate_digest


class PlasticityMechanismV149(str, Enum):
    CURRENT_CONTEXT = "current_context"
    RETRIEVAL = "retrieval"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    SKILL = "skill"
    SKILL_COMPOSITION = "skill_composition"
    JIT_POLICY = "jit_policy"
    TEMPORARY_ACTIVATION = "temporary_activation"
    MODULAR_ADAPTER = "modular_adapter"
    FOUNDATION_UPDATE = "foundation_update"


class PermanenceLevelV149(IntEnum):
    CURRENT_CONTEXT = 0
    RETRIEVAL = 1
    EPISODIC_MEMORY = 2
    SEMANTIC_MEMORY = 3
    SKILL = 4
    SKILL_COMPOSITION = 5
    JIT_POLICY = 6
    TEMPORARY_ACTIVATION = 7
    MODULAR_ADAPTER = 8
    FOUNDATION_UPDATE = 9


MECHANISM_LEVEL_V149 = {
    PlasticityMechanismV149.CURRENT_CONTEXT: PermanenceLevelV149.CURRENT_CONTEXT,
    PlasticityMechanismV149.RETRIEVAL: PermanenceLevelV149.RETRIEVAL,
    PlasticityMechanismV149.EPISODIC_MEMORY: PermanenceLevelV149.EPISODIC_MEMORY,
    PlasticityMechanismV149.SEMANTIC_MEMORY: PermanenceLevelV149.SEMANTIC_MEMORY,
    PlasticityMechanismV149.SKILL: PermanenceLevelV149.SKILL,
    PlasticityMechanismV149.SKILL_COMPOSITION: PermanenceLevelV149.SKILL_COMPOSITION,
    PlasticityMechanismV149.JIT_POLICY: PermanenceLevelV149.JIT_POLICY,
    PlasticityMechanismV149.TEMPORARY_ACTIVATION: PermanenceLevelV149.TEMPORARY_ACTIVATION,
    PlasticityMechanismV149.MODULAR_ADAPTER: PermanenceLevelV149.MODULAR_ADAPTER,
    PlasticityMechanismV149.FOUNDATION_UPDATE: PermanenceLevelV149.FOUNDATION_UPDATE,
}

DEFAULT_ORDER_V149 = tuple(PlasticityMechanismV149)


@dataclass(frozen=True)
class EmpiricalPlasticityPolicyV149:
    min_success_rate: float = 0.80
    min_retention: float = 0.95
    max_interference: float = 0.02
    max_security_regressions: int = 0
    max_latency_ms: float = 120000.0
    max_compute_cost: float = 1.0
    require_causal_gain: bool = True
    min_causal_gain: float = 0.01
    allow_adapter: bool = False
    allow_foundation_update: bool = False
    schema: str = "mini-agi-v14.1-alpha9-plasticity-policy-v1"

    def __post_init__(self):
        for value in (self.min_success_rate, self.min_retention):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("rates must be in [0,1]")
        if not 0.0 <= float(self.max_interference) <= 1.0:
            raise ValueError("max_interference must be in [0,1]")
        if self.max_security_regressions < 0 or self.max_latency_ms < 0 or self.max_compute_cost < 0:
            raise ValueError("policy limits may not be negative")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class PlasticityExperimentCommitmentV149:
    request_digest: str
    evaluator_id: str
    run_id: str
    policy_digest: str
    ordered_mechanisms: tuple[PlasticityMechanismV149, ...] = DEFAULT_ORDER_V149
    schema: str = "mini-agi-v14.1-alpha9-plasticity-commitment-v1"

    def __post_init__(self):
        validate_digest(self.request_digest)
        validate_digest(self.policy_digest)
        if not self.evaluator_id or not self.run_id:
            raise ValueError("evaluator_id and run_id are required")
        mechanisms = tuple(PlasticityMechanismV149(m) for m in self.ordered_mechanisms)
        if not mechanisms or len(set(mechanisms)) != len(mechanisms):
            raise ValueError("ordered_mechanisms must be non-empty and unique")
        levels = [int(MECHANISM_LEVEL_V149[m]) for m in mechanisms]
        if levels != sorted(levels):
            raise ValueError("ordered_mechanisms must follow permanence order")
        if mechanisms != DEFAULT_ORDER_V149[:len(mechanisms)]:
            raise ValueError("ordered_mechanisms must be a contiguous prefix of the permanence ladder")
        object.__setattr__(self, "ordered_mechanisms", mechanisms)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class CapabilityClosureAttemptV149:
    commitment_digest: str
    request_digest: str
    run_id: str
    evaluator_id: str
    mechanism: PlasticityMechanismV149
    predicted_success_rate: float
    predicted_interference: float
    predicted_compute_cost: float
    realized_success_rate: float
    realized_retention: float
    realized_interference: float
    realized_compute_cost: float
    causal_ablated_success_rate: float
    security_regressions: int = 0
    latency_ms: float = 0.0
    evidence_digests: tuple[str, ...] = ()
    schema: str = "mini-agi-v14.1-alpha9-capability-closure-attempt-v1"

    def __post_init__(self):
        validate_digest(self.commitment_digest)
        validate_digest(self.request_digest)
        if not self.run_id or not self.evaluator_id:
            raise ValueError("run_id and evaluator_id are required")
        object.__setattr__(self, "mechanism", PlasticityMechanismV149(self.mechanism))
        for value in (
            self.predicted_success_rate, self.predicted_interference, self.realized_success_rate,
            self.realized_retention, self.realized_interference, self.causal_ablated_success_rate,
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("rates must be finite and in [0,1]")
        for value in (self.predicted_compute_cost, self.realized_compute_cost, self.latency_ms):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("costs must be finite and non-negative")
        if self.security_regressions < 0:
            raise ValueError("security_regressions may not be negative")
        for value in self.evidence_digests:
            validate_digest(value)

    @property
    def causal_gain(self) -> float:
        return float(self.realized_success_rate) - float(self.causal_ablated_success_rate)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class PlasticityCalibrationReceiptV149:
    run_id: str
    commitment_digest: str
    attempt_digests: tuple[str, ...]
    mean_abs_success_error: float
    mean_abs_interference_error: float
    mean_abs_compute_error: float
    per_mechanism: dict
    schema: str = "mini-agi-v14.1-alpha9-plasticity-calibration-v1"

    def __post_init__(self):
        validate_digest(self.commitment_digest)
        if not self.run_id or not self.attempt_digests:
            raise ValueError("run_id and attempts are required")
        for value in self.attempt_digests:
            validate_digest(value)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class EmpiricalPlasticityReportV149:
    run_id: str
    request_digest: str
    evaluator_id: str
    commitment_digest: str
    policy_digest: str
    attempt_digests: tuple[str, ...]
    selected_mechanism: str
    decision: str
    reasons: tuple[str, ...]
    calibration_digest: str
    schema: str = "mini-agi-v14.1-alpha9-plasticity-report-v1"

    def __post_init__(self):
        for value in (self.request_digest, self.commitment_digest, self.policy_digest, self.calibration_digest):
            validate_digest(value)
        for value in self.attempt_digests:
            validate_digest(value)
        if self.decision not in {"PASS", "NO_CLOSURE"}:
            raise ValueError("decision must be PASS or NO_CLOSURE")
        if self.decision == "PASS" and not self.selected_mechanism:
            raise ValueError("PASS requires selected_mechanism")
        if self.selected_mechanism:
            PlasticityMechanismV149(self.selected_mechanism)

    @property
    def digest(self) -> str:
        return digest(self)


class EmpiricalPlasticityRouterV149:
    """Selects the least-permanent mechanism empirically demonstrated sufficient.

    Forecasts are recorded for calibration only. They cannot make a mechanism
    sufficient and cannot skip lower-permanence experiments.
    """

    def __init__(self, policy: EmpiricalPlasticityPolicyV149 | None = None):
        self.policy = policy or EmpiricalPlasticityPolicyV149()

    def is_sufficient(self, attempt: CapabilityClosureAttemptV149) -> tuple[bool, tuple[str, ...]]:
        p = self.policy
        reasons = []
        if attempt.realized_success_rate < p.min_success_rate:
            reasons.append("success_below_threshold")
        if attempt.realized_retention < p.min_retention:
            reasons.append("retention_below_threshold")
        if attempt.realized_interference > p.max_interference:
            reasons.append("interference_above_threshold")
        if attempt.security_regressions > p.max_security_regressions:
            reasons.append("security_regression")
        if attempt.latency_ms > p.max_latency_ms:
            reasons.append("latency_above_threshold")
        if attempt.realized_compute_cost > p.max_compute_cost:
            reasons.append("compute_above_threshold")
        if p.require_causal_gain and attempt.causal_gain < p.min_causal_gain:
            reasons.append("causal_gain_below_threshold")
        if attempt.mechanism is PlasticityMechanismV149.MODULAR_ADAPTER and not p.allow_adapter:
            reasons.append("adapter_disabled")
        if attempt.mechanism is PlasticityMechanismV149.FOUNDATION_UPDATE and not p.allow_foundation_update:
            reasons.append("foundation_update_disabled")
        return not reasons, tuple(reasons)

    def calibrate(self, commitment: PlasticityExperimentCommitmentV149,
                  attempts: Sequence[CapabilityClosureAttemptV149]) -> PlasticityCalibrationReceiptV149:
        if not attempts:
            raise ValueError("attempts required")
        by_mechanism = {}
        success_errors, interference_errors, compute_errors = [], [], []
        for a in attempts:
            success_err = abs(float(a.predicted_success_rate) - float(a.realized_success_rate))
            int_err = abs(float(a.predicted_interference) - float(a.realized_interference))
            cost_err = abs(float(a.predicted_compute_cost) - float(a.realized_compute_cost))
            success_errors.append(success_err); interference_errors.append(int_err); compute_errors.append(cost_err)
            by_mechanism[a.mechanism.value] = {
                "predicted_success_rate": a.predicted_success_rate,
                "realized_success_rate": a.realized_success_rate,
                "success_abs_error": success_err,
                "predicted_interference": a.predicted_interference,
                "realized_interference": a.realized_interference,
                "interference_abs_error": int_err,
                "predicted_compute_cost": a.predicted_compute_cost,
                "realized_compute_cost": a.realized_compute_cost,
                "compute_abs_error": cost_err,
            }
        return PlasticityCalibrationReceiptV149(
            run_id=commitment.run_id,
            commitment_digest=commitment.digest,
            attempt_digests=tuple(a.digest for a in attempts),
            mean_abs_success_error=math.fsum(success_errors) / len(success_errors),
            mean_abs_interference_error=math.fsum(interference_errors) / len(interference_errors),
            mean_abs_compute_error=math.fsum(compute_errors) / len(compute_errors),
            per_mechanism=by_mechanism,
        )

    def evaluate(self, *, commitment: PlasticityExperimentCommitmentV149,
                 attempts: Sequence[CapabilityClosureAttemptV149]) -> tuple[EmpiricalPlasticityReportV149, PlasticityCalibrationReceiptV149]:
        attempts = tuple(attempts)
        if not attempts:
            raise ValueError("attempts required")
        if commitment.policy_digest != self.policy.digest:
            raise ValueError("commitment policy digest does not match active plasticity policy")
        ordered = commitment.ordered_mechanisms
        if len(attempts) > len(ordered):
            raise ValueError("too many attempts")
        for idx, attempt in enumerate(attempts):
            if attempt.commitment_digest != commitment.digest or attempt.request_digest != commitment.request_digest:
                raise ValueError("attempt commitment/request mismatch")
            if attempt.run_id != commitment.run_id or attempt.evaluator_id != commitment.evaluator_id:
                raise ValueError("attempt run/evaluator mismatch")
            if attempt.mechanism is not ordered[idx]:
                raise ValueError("capability-closure experiments may not skip permanence levels")

        selected = ""
        reasons = []
        for idx, attempt in enumerate(attempts):
            sufficient, why = self.is_sufficient(attempt)
            if sufficient:
                selected = attempt.mechanism.value
                if idx != len(attempts) - 1:
                    raise ValueError("experiments continued after a sufficient lower-permanence mechanism")
                reasons.append(f"closure:{selected}")
                break
            reasons.extend(f"{attempt.mechanism.value}:{r}" for r in why)

        calibration = self.calibrate(commitment, attempts)
        decision = "PASS" if selected else "NO_CLOSURE"
        report = EmpiricalPlasticityReportV149(
            run_id=commitment.run_id,
            request_digest=commitment.request_digest,
            evaluator_id=commitment.evaluator_id,
            commitment_digest=commitment.digest,
            policy_digest=self.policy.digest,
            attempt_digests=tuple(a.digest for a in attempts),
            selected_mechanism=selected,
            decision=decision,
            reasons=tuple(reasons),
            calibration_digest=calibration.digest,
        )
        return report, calibration


@dataclass(frozen=True)
class JITPolicySampleV149:
    context_digest: str
    action: str
    reward: float
    outcome_digest: str
    source_trajectory_digest: str
    evidence_digest: str
    schema: str = "mini-agi-v14.1-alpha9-jit-policy-sample-v1"

    def __post_init__(self):
        for value in (self.context_digest, self.outcome_digest, self.source_trajectory_digest, self.evidence_digest):
            validate_digest(value)
        if not self.action:
            raise ValueError("action required")
        if not math.isfinite(float(self.reward)) or not -1.0 <= float(self.reward) <= 1.0:
            raise ValueError("reward must be finite and in [-1,1]")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class JITActionEstimateV149:
    action: str
    sample_count: int
    mean_reward: float
    posterior_reward: float
    advantage: float
    sample_digests: tuple[str, ...]


@dataclass(frozen=True)
class JITPolicyDecisionReceiptV149:
    context_digest: str
    chosen_action: str
    estimates: tuple[JITActionEstimateV149, ...]
    estimator_digest: str
    source_sample_digests: tuple[str, ...]
    schema: str = "mini-agi-v14.1-alpha9-jit-policy-decision-v1"

    def __post_init__(self):
        validate_digest(self.context_digest)
        validate_digest(self.estimator_digest)
        for value in self.source_sample_digests:
            validate_digest(value)
        if not self.chosen_action:
            raise ValueError("chosen_action required")

    @property
    def digest(self) -> str:
        return digest(self)


class JITNonParametricPolicyV149:
    """Ephemeral trajectory-memory action estimator; never mutates model weights.

    A caller may provide ``sample_validator`` to bind samples to an external
    evidence/trajectory authority. Production runtime requires that validator.
    """

    def __init__(self, *, prior_mean: float = 0.0, prior_strength: float = 2.0, min_samples: int = 1, sample_validator=None):
        if prior_strength < 0 or min_samples < 1:
            raise ValueError("invalid estimator configuration")
        self.prior_mean = float(prior_mean)
        self.prior_strength = float(prior_strength)
        self.min_samples = int(min_samples)
        self.sample_validator = sample_validator
        self._samples: dict[str, JITPolicySampleV149] = {}

    @property
    def estimator_digest(self) -> str:
        return digest({"schema": "mini-agi-v14.1-alpha9-jit-estimator-v1", "prior_mean": self.prior_mean,
                       "prior_strength": self.prior_strength, "min_samples": self.min_samples})

    def add(self, samples: Iterable[JITPolicySampleV149]):
        for sample in samples:
            if self.sample_validator is not None and not bool(self.sample_validator(sample)):
                raise PermissionError("JIT sample failed external evidence validation")
            existing = self._samples.get(sample.digest)
            if existing is not None and existing != sample:
                raise RuntimeError("JIT sample digest collision")
            self._samples[sample.digest] = sample

    def recommend(self, *, context_digest: str, allowed_actions: Sequence[str]) -> JITPolicyDecisionReceiptV149:
        validate_digest(context_digest)
        actions = tuple(dict.fromkeys(str(a) for a in allowed_actions if str(a)))
        if not actions:
            raise ValueError("allowed_actions required")
        relevant = [s for s in self._samples.values() if s.context_digest == context_digest and s.action in actions]
        if not relevant:
            raise ValueError("no JIT evidence for context/actions")
        global_mean = math.fsum(s.reward for s in relevant) / len(relevant)
        estimates = []
        for action in actions:
            samples = [s for s in relevant if s.action == action]
            if len(samples) < self.min_samples:
                continue
            mean = math.fsum(s.reward for s in samples) / len(samples)
            denom = self.prior_strength + len(samples)
            posterior = ((self.prior_strength * self.prior_mean) + math.fsum(s.reward for s in samples)) / denom if denom else mean
            estimates.append(JITActionEstimateV149(
                action=action, sample_count=len(samples), mean_reward=mean,
                posterior_reward=posterior, advantage=posterior - global_mean,
                sample_digests=tuple(sorted(s.digest for s in samples)),
            ))
        if not estimates:
            raise ValueError("insufficient JIT samples")
        estimates.sort(key=lambda x: (-x.posterior_reward, -x.sample_count, x.action))
        chosen = estimates[0].action
        source = tuple(sorted({d for e in estimates for d in e.sample_digests}))
        return JITPolicyDecisionReceiptV149(
            context_digest=context_digest, chosen_action=chosen, estimates=tuple(estimates),
            estimator_digest=self.estimator_digest, source_sample_digests=source,
        )
