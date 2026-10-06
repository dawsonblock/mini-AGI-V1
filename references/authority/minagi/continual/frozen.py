from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import AdapterBank, AdapterRoute, LowRankAdapter
from .authority import LearningAuthorityPolicy


class FrozenBackboneAdapterLM(nn.Module):
    """Generic frozen decoder plus reversible final-hidden adapters.

    Subclasses implement ``frozen_forward`` and return ``(base_logits, hidden)``.
    The output projection is frozen along with the backbone.  This provides a
    conservative learning path; deeper LoRA insertion should remain a separately
    qualified experiment.
    """

    def __init__(self, backbone: nn.Module, d_model: int,
                 output_projection: nn.Module,
                 authority: LearningAuthorityPolicy | None = None):
        super().__init__()
        self.backbone = backbone
        self.d_model = int(d_model)
        self.output_projection = output_projection
        self.authority = authority or LearningAuthorityPolicy()
        self.freeze_base()

    def freeze_base(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.output_projection.parameters():
            p.requires_grad_(False)

    def assert_base_frozen(self) -> None:
        bad = ["backbone." + n for n, p in self.backbone.named_parameters() if p.requires_grad]
        bad += ["output_projection." + n for n, p in self.output_projection.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"base parameters unexpectedly trainable: {bad[:8]}")

    def frozen_forward(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError

    def forward_with_adapter(self, input_ids: torch.Tensor,
                             adapter: LowRankAdapter | None = None, **kwargs):
        with torch.no_grad():
            base_logits, hidden = self.frozen_forward(input_ids, **kwargs)
        if adapter is None:
            return base_logits, hidden
        delta = adapter.to(hidden.device).delta(hidden.detach())
        return base_logits.detach() + self.output_projection(delta), hidden.detach()

    def forward_with_bank(self, input_ids: torch.Tensor, bank: AdapterBank, **kwargs):
        with torch.no_grad():
            base_logits, hidden = self.frozen_forward(input_ids, **kwargs)
        routes = bank.route(hidden)
        if not routes:
            return base_logits, hidden, routes
        adapted = bank.apply(hidden.detach(), routes)
        return base_logits.detach() + self.output_projection(adapted - hidden.detach()), hidden.detach(), routes


@dataclass
class AdapterTrainResult:
    losses: list[float]
    update_norms: list[float]


class FastAdapterTrainer:
    """Session-local gradient learning that can only mutate the adapter."""

    def __init__(self, lm: FrozenBackboneAdapterLM, lr: float = 2e-4,
                 clip: float = 1.0):
        self.lm = lm
        self.lr = float(lr)
        self.clip = float(clip)

    def train_batch(self, adapter: LowRankAdapter, input_ids: torch.Tensor,
                    labels: torch.Tensor, steps: int = 1) -> AdapterTrainResult:
        self.lm.authority.assert_adapter_update()
        self.lm.assert_base_frozen()
        adapter = adapter.to(input_ids.device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=self.lr, weight_decay=0.0)
        losses, norms = [], []
        for _ in range(max(1, int(steps))):
            before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
            opt.zero_grad(set_to_none=True)
            logits, _ = self.lm.forward_with_adapter(input_ids, adapter)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), self.clip)
            opt.step()
            sq = 0.0
            for k, v in adapter.state_dict().items():
                d = v.detach().float() - before[k].to(v.device).float()
                sq += float((d * d).sum())
            losses.append(float(loss.detach()))
            norms.append(sq ** 0.5)
        self.lm.assert_base_frozen()
        return AdapterTrainResult(losses=losses, update_norms=norms)
