"""Non-executable adapter artifact format.

Persistent learned adapters are stored as canonical JSON metadata + NPZ tensor
arrays.  Loading never invokes pickle or arbitrary Python object construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .adapters import AdapterMetadata, LowRankAdapter
from ..candidate_pipeline import artifact_digest

_FORMAT = "minagi-adapter-npz-v1"


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def _write_json(path: Path, obj: dict) -> None:
    data = (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_bytes(path, data)


def save_adapter_artifact(directory: str | os.PathLike, adapter: LowRankAdapter,
                          metadata: AdapterMetadata, *, centroid: torch.Tensor | None = None,
                          extra: dict | None = None) -> dict:
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    if p.is_symlink():
        raise ValueError("adapter artifact directory may not be a symlink")
    arrays = {}
    for name, value in adapter.state_dict().items():
        arrays[name] = value.detach().cpu().numpy()
    # np.savez itself does not pickle numeric ndarrays.  Write through an open
    # file descriptor so numpy cannot silently append an extension.
    fd, tmp = tempfile.mkstemp(prefix="weights.npz.tmp-", dir=str(p))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p / "weights.npz")
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
    if centroid is not None:
        c = centroid.detach().cpu().float().reshape(-1).numpy()
        fd, tmp = tempfile.mkstemp(prefix="centroid.npy.tmp-", dir=str(p))
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                np.save(f, c, allow_pickle=False); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, p / "centroid.npy")
        finally:
            try: os.unlink(tmp)
            except FileNotFoundError: pass
    manifest = {
        "format": _FORMAT,
        "metadata": metadata.as_dict(),
        "extra": extra or {},
    }
    _write_json(p / "adapter.json", manifest)
    digest, tree = artifact_digest(p)
    return {"sha256": digest, "manifest": tree, "format": _FORMAT}


def load_adapter_artifact(directory: str | os.PathLike, *, device="cpu") -> tuple[LowRankAdapter, AdapterMetadata, torch.Tensor | None, dict]:
    p = Path(directory)
    doc = json.loads((p / "adapter.json").read_text(encoding="utf-8"))
    if doc.get("format") != _FORMAT:
        raise ValueError(f"unsupported adapter artifact format: {doc.get('format')!r}")
    meta = AdapterMetadata(**doc["metadata"])
    adapter = LowRankAdapter(meta.d_model, meta.rank, meta.alpha)
    with np.load(p / "weights.npz", allow_pickle=False) as z:
        expected = set(adapter.state_dict())
        got = set(z.files)
        if got != expected:
            raise ValueError(f"adapter tensor set mismatch: missing={sorted(expected-got)}, extra={sorted(got-expected)}")
        state = {}
        for name, ref in adapter.state_dict().items():
            a = np.asarray(z[name])
            if tuple(a.shape) != tuple(ref.shape):
                raise ValueError(f"shape mismatch for {name}: {a.shape} != {tuple(ref.shape)}")
            if a.dtype.kind not in "fiu b":
                raise ValueError(f"unsupported dtype for {name}: {a.dtype}")
            state[name] = torch.from_numpy(a.copy()).to(dtype=ref.dtype)
    adapter.load_state_dict(state, strict=True)
    adapter.to(device)
    centroid = None
    cp = p / "centroid.npy"
    if cp.exists():
        c = np.load(cp, allow_pickle=False)
        if c.ndim != 1 or c.shape[0] != meta.d_model:
            raise ValueError("invalid adapter centroid shape")
        centroid = torch.from_numpy(np.asarray(c, dtype=np.float32).copy())
    return adapter, meta, centroid, doc.get("extra") or {}


def adapter_artifact_digest(directory: str | os.PathLike) -> str:
    return artifact_digest(directory)[0]
