from __future__ import annotations
from dataclasses import dataclass
import copy, hashlib
from typing import Mapping, MutableMapping, Any
import torch

@dataclass
class FastWeightSession:
    """Session-scoped reversible parameter state.

    This object intentionally knows nothing about a base model. It may only own
    explicitly supplied fast/adaptor tensors. Stable promotion is handled elsewhere.
    """
    clean_state: dict[str, torch.Tensor]
    state: dict[str, torch.Tensor]
    committed: bool = False

    @classmethod
    def from_state(cls, state: Mapping[str, torch.Tensor]):
        clean={k:v.detach().clone() for k,v in state.items()}
        return cls(clean, {k:v.detach().clone() for k,v in state.items()})

    def apply_delta(self, deltas: Mapping[str, torch.Tensor], scale: float=1.0) -> None:
        if self.committed:
            raise RuntimeError('session already committed')
        for k,d in deltas.items():
            if k not in self.state: raise KeyError(k)
            if self.state[k].shape != d.shape: raise ValueError(f'shape mismatch for {k}')
            self.state[k] = self.state[k] + d.detach().to(self.state[k]) * float(scale)

    def rollback(self) -> None:
        self.state={k:v.detach().clone() for k,v in self.clean_state.items()}
        self.committed=False

    def snapshot(self) -> dict[str,torch.Tensor]:
        return {k:v.detach().clone() for k,v in self.state.items()}

    def digest(self) -> str:
        h=hashlib.sha256()
        for k in sorted(self.state):
            t=self.state[k].detach().cpu().contiguous()
            h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        return h.hexdigest()
