from __future__ import annotations

"""Exact-boundary replay backend for RC10 qualification on HF hybrid models.

This backend intentionally favors correctness over speed: to reconstruct a seam
under a changed predecessor it executes the exact model on prefix+seam and then
extracts the recurrent/full-attention planes.  RC10 may subsequently apply the
cached suffix summary.  This is useful for oracle-grounded qualification of the
compiled suffix path before optimized vendor-specific state injection exists.
"""

from typing import Sequence

import torch

from .full_model_oracle import ExactFullModelOracle, ExactModelResult
from .conv_boundary import capture_boundary
from .gdn_reference import scan as gdn_scan, zero_state_for
from .hf_hybrid_capture import HFHybridTraceCollector, discover_hybrid_layout, extract_full_attention_pic
from .qwen35_rc10 import captured_steps
from .rc10 import TransitionOrientation
from .rc10_execution import CanonicalRC10Block, ReplayPlaneResult


class HFExactBoundaryReplayBackend:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str | None = None,
        orientation: TransitionOrientation = TransitionOrientation.LEFT,
    ):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.orientation = orientation
        layout = discover_hybrid_layout(model)
        self._fa_layers = tuple(sorted(
            int(x.layer_idx) for x in layout.layers
            if x.kind.value == "full_attention" and x.layer_idx is not None
        ))
        self._conv_layers = tuple(sorted(
            int(x.layer_idx) for x in layout.layers
            if x.kind.value == "gated_delta" and x.layer_idx is not None
        ))
        self.oracle = ExactFullModelOracle(model, device=self.device)

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return self._fa_layers

    @property
    def convolution_layers(self) -> tuple[int, ...]:
        return self._conv_layers

    @torch.no_grad()
    def _run(self, tokens: Sequence[int], *, fresh_start: int) -> ReplayPlaneResult:
        ids = torch.tensor([list(map(int, tokens))], dtype=torch.long, device=self.device)
        capture = HFHybridTraceCollector(self.model).run(
            input_ids=ids, use_cache=True, output_hidden_states=True,
        )
        recurrent = {}
        conv_states = {}
        for layer, cap in capture.recurrent.items():
            steps = captured_steps(cap)
            if not steps:
                continue
            state = zero_state_for(steps[0], orientation=self.orientation)
            recurrent[int(layer)] = gdn_scan(steps, state, orientation=self.orientation).detach()
            if cap.conv_inputs is not None:
                conv_states[int(layer)] = capture_boundary(cap.conv_inputs, cap.conv_kernel_size)

        fresh = {}
        for layer in self._fa_layers:
            whole = extract_full_attention_pic(
                capture.past_key_values, int(layer),
                positions=tuple(range(len(tokens))),
            )
            # Keep only material freshly replayed by this call.  Values are still
            # conditioned on the entire exact prefix because they came from the
            # full exact prefill above.
            key = whole.key[..., fresh_start:, :].detach()
            value = whole.value[..., fresh_start:, :].detach()
            positions = tuple(range(fresh_start, len(tokens)))
            fresh[int(layer)] = type(whole)(key, value, positions, whole.token_digest)
        return ReplayPlaneResult(recurrent, fresh, conv_states)

    def replay_boundary(
        self,
        prefix_tokens: Sequence[int],
        block: CanonicalRC10Block,
        seam_width: int,
    ) -> ReplayPlaneResult:
        width = int(seam_width)
        if width <= 0 or width >= len(block.tokens):
            raise ValueError("boundary replay requires a non-empty seam smaller than the block")
        prefix = tuple(map(int, prefix_tokens))
        return self._run(prefix + block.tokens[:width], fresh_start=len(prefix))

    def replay_segment(
        self,
        prefix_tokens: Sequence[int],
        block: CanonicalRC10Block,
    ) -> ReplayPlaneResult:
        prefix = tuple(map(int, prefix_tokens))
        return self._run(prefix + block.tokens, fresh_start=len(prefix))


    def replay_selected(self, tokens: Sequence[int]) -> ReplayPlaneResult:
        return self._run(tuple(map(int, tokens)), fresh_start=0)

    def exact(self, tokens: Sequence[int]) -> ExactModelResult:
        return self.oracle.run(tokens)


__all__ = ["HFExactBoundaryReplayBackend"]
