from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.checkpoints import PrefixCheckpointStore
from kvcontinual.execution.recurrent.coherence import RuntimeRiskGate
from kvcontinual.execution.recurrent.interfaces import ReconstructionBackend, backend_capabilities
from kvcontinual.execution.source import SourceSegmentStore
from kvcontinual.execution.types import AssemblyTopology, ExecutionIdentity, OracleMetrics, ReconstructionAction, ReconstructionMode, RuntimeRiskSignals


@dataclass
class ReconstructionResult:
    action: ReconstructionAction
    state: Any
    topology: AssemblyTopology
    risk_signals: RuntimeRiskSignals | None = None
    shadow_oracle_metrics: OracleMetrics | None = None
    cache_misses: list[str] | None = None
    reason: str = ""


class ShadowAuditSampler:
    def __init__(self, rate: float = 0.01, salt: str = "rc10"):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0,1]")
        self.rate = rate
        self.salt = salt

    def selected(self, request_key: str) -> bool:
        if self.rate <= 0:
            return False
        h = hashlib.sha256((self.salt + request_key).encode()).digest()
        u = int.from_bytes(h[:8], "big") / float(2**64)
        return u < self.rate


class ReconstructionRuntime:
    """Topology-aware RC10 reconstruction policy.

    Arbitrary PIC: fixed seam-8 HYPIC first, then optional LinearKV baseline,
    then exact selected replay. Prefix-only reconstruction is never used as a
    generic arbitrary-segment fallback.
    """
    def __init__(
        self,
        source_store: SourceSegmentStore,
        artifact_store: ExecutionArtifactStore,
        backend: ReconstructionBackend,
        checkpoints: PrefixCheckpointStore | None = None,
        gate: RuntimeRiskGate | None = None,
        shadow_sampler: ShadowAuditSampler | None = None,
        allow_single_state_experimental: bool = False,
    ):
        self.source_store = source_store
        self.artifact_store = artifact_store
        self.backend = backend
        self.checkpoints = checkpoints or PrefixCheckpointStore()
        self.gate = gate or RuntimeRiskGate()
        self.shadow_sampler = shadow_sampler or ShadowAuditSampler(0.0)
        self.allow_single_state_experimental = allow_single_state_experimental

    def reconstruct(
        self,
        source_segment_ids: list[str],
        identity: ExecutionIdentity,
        mode: ReconstructionMode = ReconstructionMode.BALANCED,
        *,
        topology: AssemblyTopology | None = None,
        conversation_id: str | None = None,
        target_token_position: int | None = None,
        request_key: str = "",
    ) -> ReconstructionResult:
        if not source_segment_ids:
            raise ValueError("source_segment_ids cannot be empty")
        topology = topology or self.source_store.classify_topology(source_segment_ids)
        caps = backend_capabilities(self.backend)

        if mode == ReconstructionMode.EXACT:
            return self._exact(source_segment_ids, topology, "EXACT mode requested")

        if topology in (AssemblyTopology.CANONICAL, AssemblyTopology.EXACT_PREFIX):
            if not caps.exact_prefix_checkpoint:
                return self._exact(source_segment_ids, topology, "backend has no exact prefix-checkpoint restore; replayed selected tokens")
            checkpoint = None
            if conversation_id is not None and target_token_position is not None:
                checkpoint = self.checkpoints.deepest(conversation_id, target_token_position, identity)
            state = self.backend.exact_prefix_replay(source_segment_ids, checkpoint)
            return ReconstructionResult(ReconstructionAction.EXACT_PREFIX_REPLAY, state, topology, reason="shared/canonical prefix path")

        if not caps.hypic_seam8:
            return self._exact(source_segment_ids, topology, "backend does not provide qualified HYPIC seam-8 artifacts")
        if not caps.causal_conv_seam_repair:
            return self._exact(source_segment_ids, topology, "backend lacks causal-convolution seam repair")
        if not caps.full_attention_relocation:
            return self._exact(source_segment_ids, topology, "backend lacks full-attention position relocation")

        if hasattr(self.artifact_store, 'admission'):
            try:
                self.artifact_store.admission()
            except (RuntimeError, ValueError, OSError) as exc:
                return self._exact(source_segment_ids, topology, f"persistent data plane not admitted: {exc}")

        source_digests = {sid: self.source_store.content_digest(sid) for sid in source_segment_ids}
        hits, misses = self.artifact_store.resolve(source_segment_ids, identity, source_digests)
        if misses:
            # Regenerate only from authoritative source tokens. A backend may cache
            # the resulting artifacts under this exact execution namespace.
            built = self.backend.materialize(misses, identity)
            for art in built:
                expected_source = source_digests.get(art.source_segment_id)
                if expected_source is None:
                    raise RuntimeError(f"backend materialized unknown source segment {art.source_segment_id!r}")
                if art.source_content_digest and art.source_content_digest != expected_source:
                    raise RuntimeError("backend materialized artifact for the wrong source content")
                art.source_content_digest = expected_source
                if hasattr(self.artifact_store, 'admission'):
                    self.artifact_store.put(art, source_token_count=len(self.source_store.get(art.source_segment_id).tokens))
                else:
                    self.artifact_store.put(art)
            hits, still_missing = self.artifact_store.resolve(source_segment_ids, identity, source_digests)
            if still_missing:
                return self._exact(source_segment_ids, topology, "cache materialization incomplete", still_missing)

        artifacts = [self.artifact_store.get(sid, identity, source_digests[sid]) for sid in source_segment_ids]
        if any(a is None for a in artifacts):
            return self._exact(source_segment_ids, topology, "exact identity cache miss")
        artifacts = [a for a in artifacts if a is not None]

        # Arbitrary assembly requires a seam contract that matches the cached tail.
        if any(a.fixed_seam_width != 8 for a in artifacts):
            return self._exact(source_segment_ids, topology, "arbitrary PIC requires fixed seam-8 artifact contract")
        if caps.requires_artifact_qualification and any(not a.hypic_qualified for a in artifacts):
            return self._exact(source_segment_ids, topology, "execution artifact lacks a complete HYPIC qualification receipt")

        approx = self.backend.hypic_seam8(artifacts)
        if mode == ReconstructionMode.FAST or self.gate.acceptable(approx.risk_signals):
            result = ReconstructionResult(
                ReconstructionAction.HYPIC_SEAM8,
                approx.state,
                topology,
                risk_signals=approx.risk_signals,
                reason="fixed seam-8 HYPIC path accepted",
            )
            self._maybe_shadow(result, source_segment_ids, request_key)
            return result

        if self.allow_single_state_experimental and caps.single_state_init:
            alt = self.backend.single_state_init(artifacts)
            if self.gate.acceptable(alt.risk_signals):
                result = ReconstructionResult(
                    ReconstructionAction.SINGLE_STATE_INIT,
                    alt.state,
                    topology,
                    risk_signals=alt.risk_signals,
                    reason="HYPIC seam risk high; LinearKV-style baseline accepted",
                )
                self._maybe_shadow(result, source_segment_ids, request_key)
                return result

        return self._exact(source_segment_ids, topology, "accelerated arbitrary-PIC path failed risk gate")

    def _exact(self, ids: list[str], topology: AssemblyTopology, reason: str, misses: list[str] | None = None):
        return ReconstructionResult(
            ReconstructionAction.EXACT_SELECTED_REPLAY,
            self.backend.exact_selected_replay(ids),
            topology,
            cache_misses=misses,
            reason=reason,
        )

    def _maybe_shadow(self, result: ReconstructionResult, ids: list[str], request_key: str) -> None:
        if self.shadow_sampler.selected(request_key or "|".join(ids)):
            exact = self.backend.exact_selected_replay(ids)
            result.shadow_oracle_metrics = self.backend.compare_to_oracle(result.state, exact)
