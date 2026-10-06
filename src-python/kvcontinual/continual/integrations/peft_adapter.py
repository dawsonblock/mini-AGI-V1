from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PeftTrainerPlan:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 2e-5
    target_modules: str = "all-linear"

    def build_lora_config(self):
        try:
            from peft import LoraConfig
        except ImportError as e:
            raise RuntimeError("Install kvcontinual[ml] to use PEFT integration") from e
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            use_rslora=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
