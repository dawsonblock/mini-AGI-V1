from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

from kvcontinual.execution.recurrent.interfaces import ApproximateExecution, BackendCapabilities
from kvcontinual.execution.source import SourceSegmentStore
from kvcontinual.execution.types import OracleMetrics, RuntimeRiskSignals


@dataclass
class ExactReplayState:
    input_ids: list[int]
    last_logits: np.ndarray
    device: str


class TransformersMPSReplayBackend:
    """Exact-replay backend for macOS using Transformers + PyTorch MPS.

    This backend intentionally advertises no HYPIC materialization capability.
    RC10 therefore routes arbitrary PIC to exact selected-token replay instead
    of manufacturing stale recurrent artifacts. If MPS is unavailable, CPU is
    used. `PYTORCH_ENABLE_MPS_FALLBACK=1` can be set by the launcher so an
    unsupported MPS operation can fall back to CPU inside PyTorch.
    """

    def __init__(self, source_store: SourceSegmentStore, model: Any, *, device: str | None = None):
        import torch
        self.torch = torch
        self.source_store = source_store
        self.model = model
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        source_store: SourceSegmentStore,
        model_name_or_path: str,
        *,
        dtype: str = "float16",
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> "TransformersMPSReplayBackend":
        import torch
        from transformers import AutoModelForCausalLM
        dtype_obj = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(dtype)
        if dtype_obj is None:
            raise ValueError(f"unsupported dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype_obj,
            trust_remote_code=trust_remote_code,
        )
        return cls(source_store, model, device=device)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            hypic_seam8=False,
            single_state_init=False,
            exact_prefix_checkpoint=False,
            exact_selected_replay=True,
            causal_conv_seam_repair=False,
            full_attention_relocation=False,
            requires_artifact_qualification=True,
            platform="macos-transformers",
            device=self.device,
        )

    def _tokens(self, ids: list[str]) -> list[int]:
        out: list[int] = []
        for sid in ids:
            seg = self.source_store.get(sid)
            if seg is None:
                raise KeyError(f"unknown source segment: {sid}")
            out.extend(seg.tokens)
        if not out:
            raise ValueError("selected source segments contain no tokens")
        return out

    def materialize(self, source_segment_ids, identity):
        return []

    def hypic_seam8(self, artifacts):
        raise NotImplementedError("macOS exact backend does not claim HYPIC capture/composition")

    def single_state_init(self, artifacts):
        raise NotImplementedError("macOS exact backend does not claim LinearKV state initialization")

    def exact_prefix_replay(self, source_segment_ids, checkpoint):
        # Without native recurrent checkpoint restore, replay the selected tokens.
        return self.exact_selected_replay(source_segment_ids)

    def exact_selected_replay(self, source_segment_ids):
        torch = self.torch
        ids = self._tokens(source_segment_ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1].detach().float().cpu().numpy()
        return ExactReplayState(ids, logits, self.device)

    def generate_from_segments(self, source_segment_ids: list[str], *, max_new_tokens: int = 64, **kwargs):
        torch = self.torch
        ids = self._tokens(source_segment_ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            generated = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)
        return generated[0].detach().cpu().tolist()

    def compare_to_oracle(self, approximate: Any, exact: Any) -> OracleMetrics:
        a = np.asarray(approximate.last_logits if hasattr(approximate, "last_logits") else approximate, dtype=np.float64)
        b = np.asarray(exact.last_logits if hasattr(exact, "last_logits") else exact, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError("oracle logit shapes differ")
        amax, bmax = np.max(a), np.max(b)
        pa = np.exp(a - amax); pa /= pa.sum()
        pb = np.exp(b - bmax); pb /= pb.sum()
        eps = 1e-12
        kl = float(np.sum(pa * np.log((pa + eps) / (pb + eps))))
        return OracleMetrics(logit_kl=kl, top1_agreement=float(np.argmax(a) == np.argmax(b)))
