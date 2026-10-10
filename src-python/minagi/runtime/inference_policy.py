"""v16.4.4 inference resource budgets (WP-E / SEC-306).

An authenticated client must not obtain unbounded inference by
manipulating generation parameters or abandoning requests.
`InferenceBudgetPolicyV1` is the enforceable contract:

  * ``max_prompt_tokens`` — the backend tokenizes the prompt and
    checks the ACTUAL token count; character length is not a budget;
  * ``max_new_tokens`` — a requested generation above the bound is
    clamped to it and the clamp is reported in the request metrics;
  * ``execution_deadline_seconds`` — passed to the generator's
    cooperative stopping (``max_time``) so a network timeout is never
    mistaken for cancellation of GPU computation;
  * ``max_concurrent_requests`` — enforced by the router BEFORE a
    lease reaches the backend;
  * ``max_queued_per_principal`` — per-principal queue bound enforced
    at lease acquisition;
  * ``max_retries`` — retry policy is governance-controlled, not
    client-controlled.
"""
from __future__ import annotations

from dataclasses import dataclass

INFERENCE_BUDGET_SCHEMA = "mini-agi-v16.4.4-inference-budget-v1"


class BudgetExceeded(PermissionError):
    """The request exceeds the authorized inference budget."""


@dataclass(frozen=True)
class InferenceBudgetPolicyV1:
    """Bounded inference contract. Defaults are the small-model
    development configuration — calibrate against measured hardware
    before production."""
    max_prompt_tokens: int = 4096
    max_new_tokens: int = 512
    execution_deadline_seconds: float = 60.0
    max_concurrent_requests: int = 2
    max_queued_per_principal: int = 8
    max_retries: int = 0
    schema: str = INFERENCE_BUDGET_SCHEMA

    def __post_init__(self):
        for name in ("max_prompt_tokens", "max_new_tokens",
                     "max_concurrent_requests", "max_queued_per_principal",
                     "max_retries"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_prompt_tokens == 0:
            raise ValueError("max_prompt_tokens of 0 serves nothing")
        if self.max_concurrent_requests == 0:
            raise ValueError(
                "max_concurrent_requests of 0 serves nothing")
        if not (0 < float(self.execution_deadline_seconds)):
            raise ValueError(
                "execution_deadline_seconds must be positive")

    def check_prompt_tokens(self, token_count: int) -> None:
        if int(token_count) > self.max_prompt_tokens:
            raise BudgetExceeded(
                f"prompt is {token_count} tokens — the authorized "
                f"prompt budget is {self.max_prompt_tokens}")

    def clamp_max_new_tokens(self, requested: int) -> int:
        return min(int(requested), self.max_new_tokens)

    def to_doc(self) -> dict:
        return {"schema": self.schema,
                "max_prompt_tokens": self.max_prompt_tokens,
                "max_new_tokens": self.max_new_tokens,
                "execution_deadline_seconds":
                    self.execution_deadline_seconds,
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_queued_per_principal": self.max_queued_per_principal,
                "max_retries": self.max_retries}

    @classmethod
    def from_doc(cls, doc: dict) -> "InferenceBudgetPolicyV1":
        if not isinstance(doc, dict):
            raise ValueError("inference budget must be a JSON object")
        schema = str(doc.get("schema", INFERENCE_BUDGET_SCHEMA))
        if schema != INFERENCE_BUDGET_SCHEMA:
            raise ValueError(
                f"inference budget schema {schema!r} unknown")
        return cls(
            max_prompt_tokens=int(doc.get("max_prompt_tokens", 4096)),
            max_new_tokens=int(doc.get("max_new_tokens", 512)),
            execution_deadline_seconds=float(
                doc.get("execution_deadline_seconds", 60.0)),
            max_concurrent_requests=int(
                doc.get("max_concurrent_requests", 2)),
            max_queued_per_principal=int(
                doc.get("max_queued_per_principal", 8)),
            max_retries=int(doc.get("max_retries", 0)))
