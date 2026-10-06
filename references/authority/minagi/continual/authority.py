from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningAuthorityPolicy:
    """Authority boundary for persistent learning.

    The default is intentionally asymmetric: memory and reversible adapter
    learning are allowed, while persistent expert/core mutations are denied.
    A research process may opt in to those mutations, but production promotion
    still belongs to the candidate qualification/signing plane.
    """

    allow_memory_writes: bool = True
    allow_adapter_updates: bool = True
    allow_expert_updates: bool = False
    allow_core_updates: bool = False
    require_qualification_for_persistent_neural_state: bool = True

    def assert_memory_write(self) -> None:
        if not self.allow_memory_writes:
            raise PermissionError("persistent memory writes are disabled")

    def assert_adapter_update(self) -> None:
        if not self.allow_adapter_updates:
            raise PermissionError("adapter updates are disabled")

    def assert_expert_update(self) -> None:
        if not self.allow_expert_updates:
            raise PermissionError(
                "persistent expert updates are locked; train a shadow candidate and qualify it"
            )

    def assert_core_update(self) -> None:
        if not self.allow_core_updates:
            raise PermissionError(
                "core-model updates are locked; train a shadow candidate and qualify it"
            )
