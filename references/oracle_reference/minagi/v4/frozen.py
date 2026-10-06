from __future__ import annotations
import hashlib
import torch

class FrozenBaseViolation(RuntimeError):
    pass

class FrozenBaseGuard:
    """Fail-closed guard for the v4 frozen-base milestone."""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._digest = self.parameter_digest()

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    def assert_frozen(self):
        bad = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if bad:
            raise FrozenBaseViolation(f"base parameters require gradients: {bad[:8]}")
        now = self.parameter_digest()
        if now != self._digest:
            raise FrozenBaseViolation("foundation-model parameters changed")
        return True

    def parameter_digest(self) -> str:
        h = hashlib.sha256()
        for name, p in self.model.named_parameters():
            t = p.detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
        return h.hexdigest()
