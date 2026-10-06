from __future__ import annotations
from dataclasses import dataclass

class BudgetExceeded(RuntimeError): pass

@dataclass(frozen=True, slots=True)
class ResourceBudget:
    max_blocks: int = 100000
    max_tokens: int = 10_000_000
    max_payload_bytes: int = 64 * 1024**3

    def validate(self, *, blocks:int, tokens:int, payload_bytes:int) -> None:
        vals=(blocks,tokens,payload_bytes)
        if any(v < 0 for v in vals): raise ValueError('resource usage cannot be negative')
        if blocks > self.max_blocks: raise BudgetExceeded('block budget exceeded')
        if tokens > self.max_tokens: raise BudgetExceeded('token budget exceeded')
        if payload_bytes > self.max_payload_bytes: raise BudgetExceeded('payload byte budget exceeded')
