from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from kvcontinual.continual.recurrent.affine import AffineSummary
from kvcontinual.continual.types import CacheIdentity


@dataclass
class HybridMemoryBlock:
    block_id: str
    token_start: int
    token_end: int
    tokens: list[int]
    recurrence: dict[int, AffineSummary]
    cache_identity: CacheIdentity
    position_metadata: dict[str, Any] = field(default_factory=dict)
    attention_kv_ref: str | None = None
    entry_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    exit_hidden: dict[int, np.ndarray] = field(default_factory=dict)
    conv_boundary_state: dict[int, np.ndarray] = field(default_factory=dict)
    novelty: float = 0.0
    future_utility: float = 0.0
    confidence: float = 0.0
    importance: float = 0.0
    evidence_id: str | None = None

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start
