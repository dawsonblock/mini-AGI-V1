"""Canonical optimiser parameter ownership for every learning path.

The model has two plasticity rates: a slow shared trunk and a fast adaptive
pool.  Routers, depth embeddings, gates, resident expert tensors and normal
resident experts all belong to the pool.  Keeping this classification in one
module prevents corpus training and live learning from silently assigning the
same parameter to different rates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParameterSplit:
    trunk: tuple[torch.nn.Parameter, ...]
    pool: tuple[torch.nn.Parameter, ...]
    trunk_names: tuple[str, ...]
    pool_names: tuple[str, ...]


def _pool_parameter_ids(model) -> set[int]:
    # Import lazily to avoid a module cycle at import time.
    from .pool import PooledMLP

    ids: set[int] = set()
    pool = getattr(model, "pool", None)
    if pool is not None:
        gate = getattr(pool, "gate", None)
        if gate is not None:
            ids.add(id(gate))
        experts = getattr(pool, "experts", None)
        if experts is not None:
            for expert in experts:
                ids.update(id(p) for p in expert.parameters())
        # PagedPool stores resident experts as stacked parameters instead of a
        # ModuleList.  These names deliberately mirror PagedPool.
        for name in ("w1", "w3", "w2"):
            obj = getattr(pool, name, None)
            if obj is None:
                continue
            if isinstance(obj, torch.nn.Parameter):
                ids.add(id(obj))
            elif torch.is_tensor(obj):
                ids.add(id(obj))
            elif hasattr(obj, "parameters"):
                ids.update(id(p) for p in obj.parameters())

    # Router rows and depth embeddings are logically expert-selection state,
    # even though their module names live under recurrent blocks rather than
    # under model.pool.
    for site in model.modules():
        if isinstance(site, PooledMLP):
            ids.add(id(site.router.weight))
            ids.add(id(site.depth_emb))
    return ids


def split_parameters(model) -> ParameterSplit:
    pool_ids = _pool_parameter_ids(model)
    trunk, pool, trunk_names, pool_names = [], [], [], []
    for name, param in model.named_parameters():
        if id(param) in pool_ids:
            pool.append(param)
            pool_names.append(name)
        else:
            trunk.append(param)
            trunk_names.append(name)
    # Tied parameters can appear under multiple names; named_parameters
    # de-duplicates them by default, so the two sets must be a partition.
    if {id(p) for p in trunk} & {id(p) for p in pool}:
        raise RuntimeError("optimizer parameter ownership overlaps")
    return ParameterSplit(tuple(trunk), tuple(pool),
                          tuple(trunk_names), tuple(pool_names))


def adamw_groups(model, lr: float, trunk_lr_mult: float, weight_decay: float,
                 split_decay: bool = True) -> list[dict]:
    """Return canonical AdamW groups with stable ``base_lr`` anchors.

    Matrix parameters receive weight decay; vectors (norm scales, gates,
    depth embeddings and biases) do not.  The returned group names start with
    ``trunk`` or ``pool`` so existing schedulers can preserve the two rates.
    """
    s = split_parameters(model)
    out: list[dict] = []
    for owner, params, owner_lr in (
        ("trunk", s.trunk, lr * trunk_lr_mult),
        ("pool", s.pool, lr),
    ):
        if split_decay:
            matrices = [p for p in params if p.dim() >= 2]
            vectors = [p for p in params if p.dim() < 2]
            if matrices:
                out.append({"params": matrices, "name": owner,
                            "weight_decay": weight_decay,
                            "lr": owner_lr, "base_lr": owner_lr})
            if vectors:
                out.append({"params": vectors, "name": owner,
                            "weight_decay": 0.0,
                            "lr": owner_lr, "base_lr": owner_lr})
        elif params:
            out.append({"params": list(params), "name": owner,
                        "weight_decay": weight_decay,
                        "lr": owner_lr, "base_lr": owner_lr})
    return out


def ownership_by_id(model) -> dict[int, str]:
    s = split_parameters(model)
    out = {id(p): "trunk" for p in s.trunk}
    out.update({id(p): "pool" for p in s.pool})
    return out
