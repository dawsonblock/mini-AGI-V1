from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from .generation import EffectiveModelGeneration
from .memory import CanonicalMemoryStore
from .cache_registry import NeuralCacheRegistry
from .adapter_bank import SkillAdapterBank

AUTHORITY_POLICY = {
    "schema": "mini-agi-v4-authority-policy-v1",
    "foundation_model": {"mutable": False, "authority": "highest", "promotion_required": True},
    "stable_adapters": {"mutable": False, "authority": "high", "promotion_required": True},
    "fast_weights": {"mutable": True, "authority": "session", "persistent": False},
    "canonical_memory": {"mutable": "append-only revisions", "authority": "sourceable", "model_independent": True},
    "neural_caches": {"mutable": True, "authority": "none", "disposable": True},
}

@dataclass
class V4Workspace:
    root: Path
    generation: EffectiveModelGeneration
    memory: CanonicalMemoryStore
    cache_registry: NeuralCacheRegistry
    adapter_bank: SkillAdapterBank

    @classmethod
    def initialize(cls, root: str | Path, generation: EffectiveModelGeneration):
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        for name in ("baseline", "candidates", "evidence", "model_generations", "compiled_caches", "adapter_bank"):
            (root / name).mkdir(exist_ok=True)
        (root / "AUTHORITY_POLICY.json").write_text(json.dumps(AUTHORITY_POLICY, indent=2, sort_keys=True) + "\n")
        gpath = root / "model_generations" / f"{generation.digest}.json"
        if not gpath.exists():
            gpath.write_text(json.dumps({
                "digest": generation.digest,
                "base_weights": generation.base_weights,
                "adapters": generation.adapters,
                "tokenizer": generation.tokenizer,
                "chat_template": generation.chat_template,
                "rope_config": generation.rope_config,
                "layer_layout": generation.layer_layout,
                "execution_config": generation.execution_config,
                "cache_abi_version": generation.cache_abi_version,
            }, indent=2, sort_keys=True) + "\n")
        return cls(
            root, generation,
            CanonicalMemoryStore(root / "canonical_memory.sqlite"),
            NeuralCacheRegistry(root / "compiled_caches"),
            SkillAdapterBank(root / "adapter_bank"),
        )
