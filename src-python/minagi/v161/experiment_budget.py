"""v16.4.5 WP11 — enforceable experiment budgets.

The learning controller decides WHICH intervention is justified; the
experiment budget decides WHETHER experiments may run at all. The
distinction matters for governance: the research model can propose any
experiment it likes, but resource consumption is metered by the
execution environment — not checked by the model under experiment.

    ExperimentBudgetPolicy  — frozen ceiling: attempts, evaluation
                              calls, wall-clock, GPU-seconds, memory,
                              spend. Digest-bound like every other
                              campaign-frozen object.
    ExperimentBudgetLedger  — the enforcement point. Every charge is
                              checked BEFORE it is consumed (atomic:
                              no partial consumption on breach) and
                              cumulative — a budget that is spent is
                              spent, retrying does not refund it.
    ExhaustionRecord        — the signed durable evidence of a breach:
                              "a failed experiment produces a signed
                              failure or exhaustion record instead of
                              silently retrying".
    AttemptExecutor         — the environment-side path: charge the
                              budgeted attempt, run it, and on breach
                              emit the signed record rather than
                              letting the attempt start.

Determinism: the ledger is a pure accounting function of the charges
presented to it — identical charge sequences produce identical
verdicts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, SignedEnvelope

EXPERIMENT_BUDGET_SCHEMA = "mini-agi-v16.4.5-experiment-budget-v1"
EXHAUSTION_SCHEMA = "mini-agi-v16.4.5-budget-exhaustion-v1"

# Dimensions the ledger meters. All are cumulative non-negative
# quantities; a limit of 0 means that resource may not be consumed.
_DIMENSIONS = ("attempts", "evaluation_calls", "wall_clock_seconds",
               "gpu_seconds", "memory_bytes", "spend")


@dataclass(frozen=True)
class ExperimentBudgetPolicy:
    """The frozen experiment ceiling. Every field is a hard maximum —
    the ledger refuses the charge that would cross it."""
    max_attempts: int
    max_evaluation_calls: int
    wall_clock_seconds: float
    gpu_seconds: float
    memory_bytes: int
    spend_ceiling: float
    schema: str = EXPERIMENT_BUDGET_SCHEMA

    def __post_init__(self):
        if int(self.max_attempts) < 0 or int(self.max_evaluation_calls) < 0:
            raise ValueError("attempt and evaluation-call limits must "
                             "be >= 0")
        for name in ("wall_clock_seconds", "gpu_seconds",
                     "memory_bytes", "spend_ceiling"):
            if not math.isfinite(float(getattr(self, name))) \
                    or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be a finite value >= 0 — "
                                 "an unbounded budget is not a budget")

    @property
    def digest(self) -> str:
        return digest(self)

    def limit(self, dimension: str) -> float:
        return float({"attempts": float(self.max_attempts),
                      "evaluation_calls":
                          float(self.max_evaluation_calls),
                      "wall_clock_seconds": float(self.wall_clock_seconds),
                      "gpu_seconds": float(self.gpu_seconds),
                      "memory_bytes": float(self.memory_bytes),
                      "spend": float(self.spend_ceiling)}[dimension])


class BudgetExhausted(RuntimeError):
    """A charge would exceed the frozen policy. Carries the dimension
    and the numbers so the exhaustion record can quote them without
    re-measuring."""

    def __init__(self, dimension: str, limit: float, consumed: float,
                 requested: float):
        self.dimension = str(dimension)
        self.limit = float(limit)
        self.consumed = float(consumed)
        self.requested = float(requested)
        super().__init__(
            f"experiment budget exhausted: {self.dimension} "
            f"{self.consumed:.6g}+{self.requested:.6g} would exceed "
            f"{self.limit:.6g}")


@dataclass(frozen=True)
class ExhaustionRecord:
    """Signed durable evidence that an experiment hit its budget —
    the record a failed or refused attempt leaves behind instead of a
    silent retry (WP11)."""
    policy_digest: str
    attempt_id: str
    dimension: str
    limit: float
    consumed: float
    requested: float
    observed_at: float
    signer_key_id: str
    signature_b64: str
    schema: str = EXHAUSTION_SCHEMA

    def __post_init__(self):
        validate_digest(self.policy_digest)
        if not self.attempt_id:
            raise ValueError("attempt_id required")
        if self.dimension not in _DIMENSIONS:
            raise ValueError(f"unknown budget dimension {self.dimension}")
        for x in (self.limit, self.consumed, self.requested,
                  self.observed_at):
            if not math.isfinite(float(x)) or float(x) < 0:
                raise ValueError("recorded quantities must be finite "
                                 "and non-negative")
        if not self.signer_key_id or not self.signature_b64:
            raise ValueError("signed exhaustion record required")

    @property
    def body(self) -> dict:
        return {"schema": self.schema,
                "policy_digest": self.policy_digest,
                "attempt_id": self.attempt_id,
                "dimension": self.dimension,
                "limit": float(self.limit),
                "consumed": float(self.consumed),
                "requested": float(self.requested),
                "observed_at": float(self.observed_at),
                "signer_key_id": self.signer_key_id}

    @property
    def digest(self) -> str:
        return digest(self)

    def verify(self, verifier) -> bool:
        return verifier.verify(
            self.body, SignedEnvelope(self.signer_key_id,
                                      self.signature_b64))


class ExperimentBudgetLedger:
    """The enforcement point the execution environment owns.

    `charge()` is atomic: every dimension is checked against the
    frozen policy BEFORE anything is consumed, so a refused charge
    leaves the ledger exactly as it was — no partial spend a retry
    could build on. `exhausted` reports whether ANY dimension is at
    its ceiling, which is the signal the controller gate uses to
    refuse new experiments."""

    def __init__(self, policy: ExperimentBudgetPolicy):
        self.policy = policy
        self._spent = {d: 0.0 for d in _DIMENSIONS}

    def spent(self, dimension: str) -> float:
        return float(self._spent[dimension])

    @property
    def exhausted(self) -> bool:
        """Whether another ATTEMPT may begin. The decision gate asks
        this question — dimension-specific ceilings (GPU, memory,
        spend) still refuse at charge() time, so a deliberate
        gpu_seconds=0 CPU-only policy does not deadlock the gate."""
        return self._spent["attempts"] >= \
            self.policy.limit("attempts")

    def charge(self, *, attempts: int = 0, evaluation_calls: int = 0,
               wall_clock_seconds: float = 0.0,
               gpu_seconds: float = 0.0, memory_bytes: float = 0.0,
               spend: float = 0.0) -> None:
        """Atomically consume budget. Raises BudgetExhausted on the
        FIRST dimension that would breach — before anything is
        consumed."""
        req = {"attempts": float(attempts),
               "evaluation_calls": float(evaluation_calls),
               "wall_clock_seconds": float(wall_clock_seconds),
               "gpu_seconds": float(gpu_seconds),
               "memory_bytes": float(memory_bytes),
               "spend": float(spend)}
        for d, v in req.items():
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{d} charge must be finite and >= 0")
        for d in _DIMENSIONS:
            limit = self.policy.limit(d)
            if self._spent[d] + req[d] > limit:
                raise BudgetExhausted(d, limit, self._spent[d], req[d])
        for d in _DIMENSIONS:
            self._spent[d] += req[d]

    def exhaustion_record(self, exc: BudgetExhausted, *,
                          attempt_id: str, signer: Ed25519Signer,
                          at: float = 0.0) -> ExhaustionRecord:
        """Turn a breach into signed durable evidence."""
        body = {"schema": EXHAUSTION_SCHEMA,
                "policy_digest": self.policy.digest,
                "attempt_id": str(attempt_id),
                "dimension": exc.dimension,
                "limit": float(exc.limit),
                "consumed": float(exc.consumed),
                "requested": float(exc.requested),
                "observed_at": float(at),
                "signer_key_id": signer.key_id}
        env = signer.sign(body)
        return ExhaustionRecord(
            policy_digest=self.policy.digest,
            attempt_id=str(attempt_id), dimension=exc.dimension,
            limit=exc.limit, consumed=exc.consumed,
            requested=exc.requested, observed_at=float(at),
            signer_key_id=env.key_id, signature_b64=env.signature_b64)


class AttemptExecutor:
    """The execution-environment path for mechanism attempts (WP11).

    `execute` enforces the budget BEFORE work starts — the attempt
    count is charged up front so an attempt that is refused never
    consumes runtime — and turns a breach into a signed
    ExhaustionRecord. `work` is a callable receiving nothing and
    returning a charge dict (the actual consumption measured during
    the attempt) plus the caller's result; the actuals are charged
    after the run so measured overruns still breach."""

    def __init__(self, ledger: ExperimentBudgetLedger, *,
                 signer: Ed25519Signer):
        self.ledger = ledger
        self.signer = signer

    def execute(self, attempt_id: str, work, *,
                at: float = 0.0):
        """Run `work()` under the budget. Returns (result, None) on
        success or (None, ExhaustionRecord) on refusal/overrun."""
        try:
            self.ledger.charge(attempts=1)
        except BudgetExhausted as exc:
            return None, self.ledger.exhaustion_record(
                exc, attempt_id=attempt_id, signer=self.signer, at=at)
        result, actuals = work()
        try:
            self.ledger.charge(**dict(actuals or {}))
        except BudgetExhausted as exc:
            return None, self.ledger.exhaustion_record(
                exc, attempt_id=attempt_id, signer=self.signer, at=at)
        return result, None


__all__ = ["AttemptExecutor", "BudgetExhausted",
           "EXPERIMENT_BUDGET_SCHEMA", "EXHAUSTION_SCHEMA",
           "ExhaustionRecord", "ExperimentBudgetLedger",
           "ExperimentBudgetPolicy"]
