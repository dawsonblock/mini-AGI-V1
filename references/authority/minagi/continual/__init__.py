"""Controlled continual-learning layer for mini-AGI v6.

The v6 policy deliberately separates four learning speeds:

1. transient recurrent/context state;
2. durable episodic/semantic memory;
3. reversible adapter skills;
4. qualified expert/core model generations.

The existing v5.1 virtual expert runtime remains available for research, but this
package is the default authority model for *persistent* learning: memory writes
are low authority, adapters are reversible, and expert/core mutations must be
qualified before production promotion.
"""
from .authority import LearningAuthorityPolicy
from .adapters import LowRankAdapter, AdapterBank, AdapterMetadata, AdapterRoute
from .artifacts import save_adapter_artifact, load_adapter_artifact, adapter_artifact_digest
from .frozen import FrozenBackboneAdapterLM, FastAdapterTrainer, AdapterTrainResult
from .controller import ControlledContinualManager, AdapterCandidate

__all__ = [
    "LearningAuthorityPolicy", "LowRankAdapter", "AdapterBank", "AdapterMetadata",
    "AdapterRoute", "save_adapter_artifact", "load_adapter_artifact",
    "adapter_artifact_digest", "FrozenBackboneAdapterLM", "FastAdapterTrainer",
    "AdapterTrainResult", "ControlledContinualManager", "AdapterCandidate",
]
