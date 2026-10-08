"""v16.5 experiment protocol — the signed learning/eval surface.

Prior plan versions bound model identity, dataset partitions, evaluator
digests, seeds, arms, and qualification thresholds, but the actual
experiment also depended on unsigned YAML: LoRA rank/alpha/dropout/target
modules, learning rate, steps, max_length, retrieval/memory/replay k,
practice/retention sample counts, max_new_tokens, dtype, and
quantization. The same signed plan digest could therefore authenticate
materially different experiments.

ExperimentProtocolV1 closes that gap: every experiment-affecting
hyperparameter lives in this one canonical object, and the campaign plan
binds its digest. Changing the experiment is equivalent to changing the
signed plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from egai.common.canonical import digest

RETENTION_SCORERS = ("retention_score", "containment_match")


@dataclass(frozen=True)
class ExperimentProtocolV1:
    """Every hyperparameter that can influence a campaign measurement.

    Fields are flat scalars/tuples so the canonical digest is stable and
    auditable. Operational concerns (storage roots, CUDA requirements,
    campaign ids) are deliberately excluded — they cannot change a
    measured outcome.
    """
    # neural adaptation (LoRA)
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    learning_rate: float
    train_steps: int
    train_max_length: int
    optimizer: str = "adamw-torch"
    # memory / replay arms
    retrieval_k: int = 3
    memory_k: int = 2
    replay_k: int = 2
    practice_samples: int = 8
    retention_samples: int = 12
    # generation / runtime
    max_new_tokens: int = 32
    decoding: str = "greedy"
    dtype: str = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False
    retention_scorer: str = "retention_score"
    require_native_servable_adapter: bool = True
    schema: str = "mini-agi-v16.5-experiment-protocol-v1"

    def __post_init__(self):
        object.__setattr__(self, "lora_target_modules",
                           tuple(str(t) for t in self.lora_target_modules))
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_rank/lora_alpha must be positive")
        if not 0 <= float(self.lora_dropout) < 1:
            raise ValueError("lora_dropout must be in [0,1)")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules required")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.train_steps <= 0 or self.train_max_length <= 0:
            raise ValueError("train_steps/train_max_length must be positive")
        for name in ("retrieval_k", "memory_k", "replay_k",
                     "practice_samples", "retention_samples",
                     "max_new_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.decoding != "greedy":
            raise ValueError("only greedy decoding is supported")
        if self.retention_scorer not in RETENTION_SCORERS:
            raise ValueError(f"retention_scorer must be one of {RETENTION_SCORERS}")
        if not self.dtype or not self.quantization:
            raise ValueError("dtype/quantization required")

    @property
    def digest(self) -> str:
        return digest(self)

    def lora_train_spec_kwargs(self) -> dict:
        """kwargs for LoraTrainSpec — the only training-config source."""
        return {"rank": int(self.lora_rank),
                "alpha": self.lora_alpha,
                "dropout": float(self.lora_dropout),
                "target_modules": self.lora_target_modules,
                "learning_rate": float(self.learning_rate),
                "steps": int(self.train_steps),
                "max_length": int(self.train_max_length)}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ExperimentProtocolV1":
        """Extract the signed protocol from a campaign config mapping.

        Every lora key is required — a v165 campaign cannot silently
        inherit defaults for the learning surface.
        """
        lora = dict(cfg.get("lora") or {})
        missing = [k for k in ("rank", "alpha", "dropout", "target_modules",
                               "learning_rate", "steps", "max_length")
                   if k not in lora]
        if missing:
            raise ValueError(f"v165 lora config missing required keys: {missing}")
        model = dict(cfg.get("model") or {})
        return cls(
            lora_rank=int(lora["rank"]),
            lora_alpha=float(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            lora_target_modules=tuple(str(t) for t in lora["target_modules"]),
            learning_rate=float(lora["learning_rate"]),
            train_steps=int(lora["steps"]),
            train_max_length=int(lora["max_length"]),
            retrieval_k=int(cfg.get("retrieval_k", 3)),
            memory_k=int(cfg.get("memory_k", 2)),
            replay_k=int(cfg.get("replay_k", 2)),
            practice_samples=int(cfg.get("practice_samples", 8)),
            retention_samples=int(cfg.get("retention_samples", 12)),
            max_new_tokens=int(cfg.get("max_new_tokens", 32)),
            dtype=str(model.get("dtype", "auto")),
            quantization=str(model.get("quantization", "none")),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            retention_scorer=str(cfg.get("retention_scorer", "retention_score")),
            require_native_servable_adapter=bool(
                cfg.get("require_native_servable_adapter", True)))
