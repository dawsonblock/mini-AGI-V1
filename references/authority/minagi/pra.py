"""Trainable Process Reward Agent utilities for frozen-policy search.

This module supplies the missing *trainable* verifier substrate behind
``minagi.verifier``.  It deliberately does not fabricate a trained reward
model.  Instead it trains a compact two-head controller on top of a caller-
provided frozen text encoder:

* reward head: is the current reasoning step locally supported/correct?
* search head: is external evidence likely to change that judgment?

The resulting object can be bound directly into ``ProcessVerifier`` so online
beam/tree search receives step-wise rewards while the reasoner remains frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PRAExample:
    question: str
    partial_trace: str
    step: str
    evidence: tuple[str, ...] = ()
    reward_label: int = 1
    search_label: int = 1

    def validate(self):
        if self.reward_label not in (0, 1) or self.search_label not in (0, 1):
            raise ValueError("PRA labels must be binary")


def format_pra_input(question: str, partial_trace: str, step: str,
                     evidence: Sequence[str] | None = None) -> str:
    ev = list(evidence or [])
    lines = ["<question>", question, "</question>",
             "<partial_trace>", partial_trace, "</partial_trace>",
             "<candidate_step>", step, "</candidate_step>"]
    if ev:
        lines += ["<evidence>"] + [str(x) for x in ev] + ["</evidence>"]
    return "\n".join(lines)


class ProcessRewardAgent(nn.Module):
    """Small shared controller over frozen text representations."""

    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_in = int(d_in)
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.d_in),
            nn.Linear(self.d_in, self.hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
        )
        self.reward_head = nn.Linear(self.hidden, 2)
        self.search_head = nn.Linear(self.hidden, 2)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x.float())
        return self.reward_head(h), self.search_head(h)

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor):
        self.eval()
        r, s = self(x)
        return F.softmax(r, -1)[..., 1], F.softmax(s, -1)[..., 1]


@dataclass
class PRATrainResult:
    losses: list[float]
    reward_accuracy: float
    search_accuracy: float


def _encode(encoder: Callable[[list[str]], object], texts: list[str]) -> torch.Tensor:
    out = encoder(texts)
    if hasattr(out, "detach"):
        out = out.detach().cpu()
    a = np.asarray(out, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != len(texts):
        raise ValueError("encoder must return [batch, dim] representations")
    return torch.from_numpy(a.copy())


def train_process_reward_agent(agent: ProcessRewardAgent,
                               encoder: Callable[[list[str]], object],
                               examples: Sequence[PRAExample], *,
                               epochs: int = 5, batch_size: int = 32,
                               lr: float = 2e-3, weight_decay: float = 1e-4,
                               seed: int = 0, device: str | torch.device = "cpu") -> PRATrainResult:
    if not examples:
        raise ValueError("PRA training requires examples")
    for e in examples: e.validate()
    texts = [format_pra_input(e.question, e.partial_trace, e.step, e.evidence)
             for e in examples]
    X = _encode(encoder, texts)
    if X.shape[1] != agent.d_in:
        raise ValueError(f"encoder dim {X.shape[1]} != agent d_in {agent.d_in}")
    yr = torch.tensor([e.reward_label for e in examples], dtype=torch.long)
    ys = torch.tensor([e.search_label for e in examples], dtype=torch.long)
    dev = torch.device(device)
    agent.to(dev)
    X, yr, ys = X.to(dev), yr.to(dev), ys.to(dev)
    opt = torch.optim.AdamW(agent.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    losses = []
    agent.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(len(examples), generator=gen)
        for a in range(0, len(examples), max(1, int(batch_size))):
            ii = order[a:a+batch_size].to(dev)
            rr, ss = agent(X[ii])
            loss = F.cross_entropy(rr, yr[ii]) + F.cross_entropy(ss, ys[ii])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))
    agent.eval()
    with torch.no_grad():
        rr, ss = agent(X)
        racc = float((rr.argmax(-1) == yr).float().mean())
        sacc = float((ss.argmax(-1) == ys).float().mean())
    return PRATrainResult(losses, racc, sacc)


class BoundProcessRewardAgent:
    """Bind a trained PRA to text encoding for ``ProcessVerifier`` callbacks."""

    def __init__(self, agent: ProcessRewardAgent,
                 encoder: Callable[[list[str]], object], *,
                 search_threshold: float = 0.5, device="cpu"):
        self.agent = agent.to(device).eval()
        self.encoder = encoder
        self.search_threshold = float(search_threshold)
        self.device = torch.device(device)

    def _prob(self, text: str):
        x = _encode(self.encoder, [text]).to(self.device)
        return self.agent.probabilities(x)

    def scorer(self, question: str, partial_trace: str, step: str,
               evidence: Sequence[str]) -> float:
        rp, _ = self._prob(format_pra_input(question, partial_trace, step, evidence))
        return float(rp[0])

    def search_policy(self, question: str, partial_trace: str, step: str) -> bool:
        # Search decision is made without retrieved evidence, matching the
        # causal decision point at which the controller has not searched yet.
        _, sp = self._prob(format_pra_input(question, partial_trace, step, ()))
        return bool(float(sp[0]) >= self.search_threshold)


def save_pra(path: str | os.PathLike, agent: ProcessRewardAgent, *, extra: dict | None = None):
    """Safe NPZ+JSON serialization; no pickle/object deserialization."""
    p = Path(path); p.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in agent.state_dict().items()}
    with open(p / "weights.npz.tmp", "wb") as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(p / "weights.npz.tmp", p / "weights.npz")
    meta = {"format":"minagi-pra-npz-v1", "d_in":agent.d_in, "hidden":agent.hidden,
            "extra":extra or {}}
    tmp = p / "pra.json.tmp"
    tmp.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, p / "pra.json")


def load_pra(path: str | os.PathLike, *, device="cpu") -> tuple[ProcessRewardAgent, dict]:
    p=Path(path); meta=json.loads((p/"pra.json").read_text(encoding="utf-8"))
    if meta.get("format") != "minagi-pra-npz-v1":
        raise ValueError("unsupported PRA artifact")
    agent=ProcessRewardAgent(int(meta["d_in"]), int(meta["hidden"]), dropout=0.0)
    with np.load(p/"weights.npz", allow_pickle=False) as z:
        state={}
        for k, ref in agent.state_dict().items():
            if k not in z.files or tuple(z[k].shape)!=tuple(ref.shape):
                raise ValueError(f"invalid PRA tensor {k}")
            state[k]=torch.from_numpy(np.asarray(z[k]).copy()).to(ref.dtype)
    agent.load_state_dict(state, strict=True); agent.to(device).eval()
    return agent, meta.get("extra") or {}
