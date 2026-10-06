"""Durable logical expert storage for v5 virtual paging.

Expert identity is a stable UID and never a VRAM slot.  Files retain the v4
``eNNNNN.npz`` format for backward compatibility, but every access is pinned by
SHA-256 inside a graph epoch.  An optimiser step is allowed only after backward
has completed, and updates are atomic, so backward can always reload exactly
the weights used by forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .integrity import fsync_dir, write_sidecar
from .precision import pack_bf16, unpack_bf16


@dataclass(frozen=True)
class ExpertVersion:
    uid: int
    sha256: str


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ExpertStore:
    """One atomic file per logical expert, including Adam state."""

    def __init__(self, path: str | os.PathLike, d_model: int, d_ff: int,
                 read_only: bool = False):
        self.path = str(path)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.read_only = bool(read_only)
        os.makedirs(self.path, exist_ok=True)

    def file(self, uid: int) -> str:
        return os.path.join(self.path, f"e{int(uid):05d}.npz")

    def uids(self) -> list[int]:
        out = []
        for name in os.listdir(self.path):
            if name.startswith("e") and name.endswith(".npz") and ".meta." not in name:
                try:
                    out.append(int(name[1:-4]))
                except ValueError:
                    pass
        return sorted(set(out))

    def version(self, uid: int) -> ExpertVersion:
        p = self.file(uid)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return ExpertVersion(int(uid), _sha256_file(p))

    def load(self, uid: int, expected_sha256: str | None = None) -> Dict[str, torch.Tensor]:
        p = self.file(uid)
        if expected_sha256 is not None:
            got = _sha256_file(p)
            if got != expected_sha256:
                raise RuntimeError(
                    f"expert {uid} changed during graph epoch: expected {expected_sha256}, got {got}")
        with np.load(p, allow_pickle=False) as z:
            out: Dict[str, torch.Tensor] = {}
            for key in ("w1", "w3", "w2"):
                if key not in z.files:
                    raise RuntimeError(f"expert {uid} missing {key}")
                out[key] = torch.from_numpy(np.asarray(z[key])).clone().float()
            for key in ("w1_m", "w3_m", "w2_m", "w1_v", "w3_v", "w2_v"):
                if key in z.files:
                    out[key] = unpack_bf16(z[key]).clone().float()
            step = float(np.asarray(z["adam_step"]).reshape(())) if "adam_step" in z.files else 0.0
            out["adam_step"] = torch.tensor(step, dtype=torch.float32)
            return out

    def write(self, uid: int, state: Dict[str, torch.Tensor]) -> ExpertVersion:
        if self.read_only:
            raise RuntimeError("read-only ExpertStore cannot write")
        arrays = {}
        for key, value in state.items():
            if key == "adam_step":
                arrays[key] = np.asarray(float(value.item() if torch.is_tensor(value) else value),
                                         dtype=np.float32)
            elif key.endswith(("_m", "_v")):
                arrays[key] = pack_bf16(value.detach().cpu().float())
            elif key in ("w1", "w3", "w2"):
                arrays[key] = value.detach().cpu().float().numpy()
        target = self.file(uid)
        tmp = target + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, target)
        fsync_dir(self.path)
        write_sidecar(target)
        return self.version(uid)

    def create(self, uid: int, *, seed: int | None = None,
               from_state: Dict[str, torch.Tensor] | None = None) -> ExpertVersion:
        if from_state is None:
            g = torch.Generator(device="cpu")
            if seed is not None:
                g.manual_seed(int(seed))
            # nn.Linear default-equivalent fan-in scale, without constructing a module.
            lim1 = self.d_model ** -0.5
            lim2 = self.d_ff ** -0.5
            w1 = torch.empty(self.d_ff, self.d_model).uniform_(-lim1, lim1, generator=g)
            w3 = torch.empty(self.d_ff, self.d_model).uniform_(-lim1, lim1, generator=g)
            w2 = torch.empty(self.d_model, self.d_ff).uniform_(-lim2, lim2, generator=g)
            from_state = {"w1": w1, "w3": w3, "w2": w2}
        state = {k: v.detach().cpu().float().clone() for k, v in from_state.items()
                 if k in ("w1", "w3", "w2")}
        for nm in ("w1", "w3", "w2"):
            state[nm + "_m"] = torch.zeros_like(state[nm])
            state[nm + "_v"] = torch.zeros_like(state[nm])
        state["adam_step"] = torch.tensor(0.0)
        return self.write(uid, state)
