"""Verified expert-to-backbone consolidation primitives for v4 models.

Only new v4 models with ``pool_dense_residual`` have a shared dense branch in
recurrent sparse blocks.  A candidate consolidation run distills the current
full sparse model into that dense/shared path, while keeping the sparse experts
as the teacher.  Production promotion is intentionally external: the resulting
candidate must still pass held-out and functional regression gates.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch
import torch.nn.functional as F

from .optim_groups import split_parameters
from .recur import HybridSparseMLP


@contextmanager
def sparse_enabled(model, enabled: bool):
    mods=[m for m in model.modules() if isinstance(m, HybridSparseMLP)]
    if not mods:
        raise RuntimeError("consolidation requires a v4 model with pool_dense_residual")
    old=[m.sparse_enabled for m in mods]
    for m in mods: m.sparse_enabled=bool(enabled)
    try: yield
    finally:
        for m,v in zip(mods,old): m.sparse_enabled=v


class Consolidator:
    """Online self-distillation into slow shared parameters.

    The teacher target is recomputed from the current full model immediately
    before every student update.  Only canonical trunk parameters are optimized;
    expert/router/gate parameters never enter this optimizer.
    """
    def __init__(self, model, lr=1e-5, weight_decay=0.01, temperature=1.0,
                 ce_weight=0.25, clip=1.0):
        self.model=model; self.temperature=float(temperature)
        self.ce_weight=float(ce_weight); self.clip=float(clip); self.steps=0
        split=split_parameters(model)
        if not any(isinstance(m,HybridSparseMLP) for m in model.modules()):
            raise RuntimeError("model has no dense+sparse v4 recurrent branch")
        self.opt=torch.optim.AdamW(list(split.trunk), lr=float(lr),
                                   weight_decay=float(weight_decay), betas=(0.9,0.95))

    def step(self, x, targets=None):
        self.model.train()
        with torch.no_grad(), sparse_enabled(self.model, True):
            teacher,_ = self.model(x, targets=None)
            teacher_p=F.softmax(teacher.float()/self.temperature, dim=-1)
        self.opt.zero_grad(set_to_none=True)
        with sparse_enabled(self.model, False):
            student,_ = self.model(x, targets=None)
            logp=F.log_softmax(student.float()/self.temperature, dim=-1)
            kd=F.kl_div(logp, teacher_p, reduction='batchmean')*(self.temperature**2)
            loss=kd
            ce=None
            if targets is not None:
                ce=F.cross_entropy(student.reshape(-1,student.shape[-1]).float(),
                                   targets.reshape(-1))
                loss=loss+self.ce_weight*ce
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step(); self.steps+=1
        return {"loss":float(loss.detach()),"distill":float(kd.detach()),
                "ce":None if ce is None else float(ce.detach()),
                "grad_norm":float(gn),"step":self.steps}


class ConsolidationCandidate:
    """Reversible in-process guard around consolidation experiments.

    ``Consolidator`` necessarily mutates the model it is given.  Wrapping an
    experiment in this guard snapshots only canonical trunk parameters and
    automatically restores them unless ``commit()`` is called.  The safer
    production pattern is still to run consolidation in a separate candidate
    workspace/process and promote the resulting artifact through the candidate
    authority; this guard prevents accidental mutation during local experiments
    and tests.
    """
    def __init__(self, model):
        self.model = model
        self._params = tuple(split_parameters(model).trunk)
        self._saved = None
        self._committed = False

    def __enter__(self):
        if self._saved is not None:
            raise RuntimeError("consolidation candidate session already active")
        self._saved = [p.detach().cpu().clone() for p in self._params]
        self._committed = False
        return self

    def commit(self):
        if self._saved is None:
            raise RuntimeError("consolidation candidate session is not active")
        self._committed = True

    @torch.no_grad()
    def rollback(self):
        if self._saved is None:
            return
        for p, old in zip(self._params, self._saved):
            p.copy_(old.to(device=p.device, dtype=p.dtype))
        self._committed = False

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None or not self._committed:
                self.rollback()
        finally:
            self._saved = None
        return False
