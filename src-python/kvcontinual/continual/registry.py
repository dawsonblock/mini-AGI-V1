"""Compatibility facade for the v14 adapter registry.

The legacy continual registry used to contain an unsigned local promotion path.
V14 removes that authority. All candidate registration/qualification/promotion
is delegated to the hardened execution registry, with signed promotion and
qualification-bundle requirements enabled by default.
"""
from __future__ import annotations

from kvcontinual.execution.registry import AdapterRegistry as _ExecutionAdapterRegistry, CandidateManifest


class AdapterRegistry(_ExecutionAdapterRegistry):
    def __init__(self, root: str, *, require_signed_promotions: bool = True,
                 require_qualification_bundle: bool = True):
        if not require_signed_promotions:
            raise PermissionError("v14 forbids unsigned adapter promotion registries")
        if not require_qualification_bundle:
            raise PermissionError("v14 requires independently verified qualification bundles")
        super().__init__(
            root,
            require_signed_promotions=True,
            require_qualification_bundle=True,
        )


__all__ = ["AdapterRegistry", "CandidateManifest"]
