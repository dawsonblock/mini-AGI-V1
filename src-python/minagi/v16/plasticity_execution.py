from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, Mapping, Any

from egai.common.canonical import digest, validate_digest
from minagi.v14.adaptive_v149 import (
    CapabilityClosureAttemptV149,
    EmpiricalPlasticityRouterV149,
    PlasticityExperimentCommitmentV149,
    PlasticityMechanismV149,
)


@dataclass(frozen=True)
class PlasticityTrialMetricsV160:
    """Measurements produced by an evaluator that is separate from the mechanism runner."""

    success_rate: float
    retention: float
    interference: float
    compute_cost: float
    causal_ablated_success_rate: float
    security_regressions: int = 0
    evidence_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.success_rate,
            self.retention,
            self.interference,
            self.causal_ablated_success_rate,
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("rates must be in [0,1]")
        if float(self.compute_cost) < 0:
            raise ValueError("compute_cost must be non-negative")
        if self.security_regressions < 0:
            raise ValueError("security_regressions must be non-negative")
        for d in self.evidence_digests:
            validate_digest(d)


@dataclass(frozen=True)
class MechanismExecutionResultV160:
    """Opaque result emitted by a mechanism runner.

    The runner is not trusted to grade itself.  `artifact` is handed to the
    independent evaluator callback, which produces `PlasticityTrialMetricsV160`.
    """

    mechanism: PlasticityMechanismV149
    artifact: Any
    artifact_digest: str
    predicted_success_rate: float
    predicted_interference: float
    predicted_compute_cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism", PlasticityMechanismV149(self.mechanism))
        validate_digest(self.artifact_digest)
        if not 0.0 <= float(self.predicted_success_rate) <= 1.0:
            raise ValueError("predicted_success_rate must be in [0,1]")
        if not 0.0 <= float(self.predicted_interference) <= 1.0:
            raise ValueError("predicted_interference must be in [0,1]")
        if float(self.predicted_compute_cost) < 0:
            raise ValueError("predicted_compute_cost must be non-negative")


class PlasticityExecutionHarnessV160:
    """Execute the Alpha9 permanence ladder instead of accepting claimed attempts.

    Mechanism code receives only `(mechanism, request_digest)`.  It returns an
    opaque candidate artifact.  A separately supplied evaluator measures that
    artifact and creates the only metrics accepted by the Alpha9 router.

    The harness stops at the first empirically sufficient mechanism.  It never
    promotes or activates the artifact; its output is proposal/qualification
    evidence only.
    """

    def __init__(self, router: EmpiricalPlasticityRouterV149 | None = None):
        self.router = router or EmpiricalPlasticityRouterV149()

    def run(
        self,
        *,
        commitment: PlasticityExperimentCommitmentV149,
        mechanism_runners: Mapping[PlasticityMechanismV149, Callable[[PlasticityMechanismV149, str], MechanismExecutionResultV160]],
        evaluator: Callable[[MechanismExecutionResultV160, str], PlasticityTrialMetricsV160],
    ):
        if commitment.policy_digest != self.router.policy.digest:
            raise ValueError("commitment policy does not match active router policy")

        attempts: list[CapabilityClosureAttemptV149] = []
        execution_digests: list[str] = []
        for mechanism in commitment.ordered_mechanisms:
            runner = mechanism_runners.get(mechanism)
            if runner is None:
                raise KeyError(f"missing mechanism runner for {mechanism.value}")
            started = monotonic()
            result = runner(mechanism, commitment.request_digest)
            elapsed_ms = (monotonic() - started) * 1000.0
            if result.mechanism is not mechanism:
                raise ValueError("mechanism runner returned a different mechanism")
            metrics = evaluator(result, commitment.request_digest)
            trial_digest = digest({
                "schema": "mini-agi-v16-plasticity-trial-v1",
                "commitment_digest": commitment.digest,
                "mechanism": mechanism.value,
                "artifact_digest": result.artifact_digest,
                "metrics": metrics,
            })
            execution_digests.append(trial_digest)
            evidence = tuple(dict.fromkeys((result.artifact_digest, trial_digest, *metrics.evidence_digests)))
            attempt = CapabilityClosureAttemptV149(
                commitment_digest=commitment.digest,
                request_digest=commitment.request_digest,
                run_id=commitment.run_id,
                evaluator_id=commitment.evaluator_id,
                mechanism=mechanism,
                predicted_success_rate=result.predicted_success_rate,
                predicted_interference=result.predicted_interference,
                predicted_compute_cost=result.predicted_compute_cost,
                realized_success_rate=metrics.success_rate,
                realized_retention=metrics.retention,
                realized_interference=metrics.interference,
                realized_compute_cost=metrics.compute_cost,
                causal_ablated_success_rate=metrics.causal_ablated_success_rate,
                security_regressions=metrics.security_regressions,
                latency_ms=elapsed_ms,
                evidence_digests=evidence,
            )
            attempts.append(attempt)
            sufficient, _ = self.router.is_sufficient(attempt)
            if sufficient:
                break

        report, calibration = self.router.evaluate(commitment=commitment, attempts=tuple(attempts))
        return report, calibration, tuple(attempts), tuple(execution_digests)
