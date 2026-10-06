from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class MlxLoraPlan:
    model: str
    data_dir: str
    adapter_path: str
    iters: int = 200
    batch_size: int = 1
    num_layers: int = 4
    learning_rate: float = 2e-5
    grad_accumulation_steps: int = 1
    mask_prompt: bool = True
    grad_checkpoint: bool = True

    def command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--model",
            self.model,
            "--train",
            "--data",
            self.data_dir,
            "--adapter-path",
            self.adapter_path,
            "--iters",
            str(self.iters),
            "--batch-size",
            str(self.batch_size),
            "--num-layers",
            str(self.num_layers),
            "--learning-rate",
            str(self.learning_rate),
            "--grad-accumulation-steps",
            str(self.grad_accumulation_steps),
        ]
        if self.mask_prompt:
            cmd.append("--mask-prompt")
        if self.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        return cmd

    def validate(self) -> None:
        data = Path(self.data_dir)
        if not (data / "train.jsonl").is_file():
            raise FileNotFoundError(f"missing {data / 'train.jsonl'}")
        Path(self.adapter_path).parent.mkdir(parents=True, exist_ok=True)

    def run(self, check: bool = True) -> subprocess.CompletedProcess[str]:
        self.validate()
        return subprocess.run(self.command(), text=True, check=check)
