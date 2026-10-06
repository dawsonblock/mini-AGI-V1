"""mini-AGI v16.1 convergence layer.

This package hardens campaign identity/evidence while preserving the v15.8
execution interfaces.  It is intentionally additive: old APIs remain available
for regression compatibility, while new experiments should use v161 APIs.
"""
from .evaluator_registry import EvaluatorArtifact, EvaluatorRegistry
from .dataset_manifest import DatasetMember, DatasetMembershipManifest, DatasetPartitionSet
from .runtime_closure3 import RuntimeClosureV161, RuntimeArtifactSpec

__all__ = [
    "EvaluatorArtifact", "EvaluatorRegistry", "DatasetMember",
    "DatasetMembershipManifest", "DatasetPartitionSet", "RuntimeClosureV161",
    "RuntimeArtifactSpec",
]
