from __future__ import annotations

from dataclasses import dataclass
from kvcontinual.execution.types import ExecutionIdentity


@dataclass(frozen=True)
class CacheNamespace:
    execution_identity: ExecutionIdentity
    label: str

    @property
    def key(self) -> str:
        return self.execution_identity.digest


class NamespaceRouter:
    """Pins one execution identity through prefill and decode for a request."""
    def __init__(self):
        self.production: CacheNamespace | None = None

    def activate(self, namespace: CacheNamespace) -> None:
        self.production = namespace

    def require(self) -> CacheNamespace:
        if self.production is None:
            raise RuntimeError("no production cache namespace")
        return self.production
