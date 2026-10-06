from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Mapping, Any
import time
from .util import canonical_json, sha256_bytes, tree_digest

class CandidateState(str, Enum):
    CANDIDATE = "candidate"
    OFFLINE = "offline_qualified"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"

@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    artifact_path: str
    artifact_digest: str
    base_generation: str
    training_manifest_digest: str
    created_at: float

    @classmethod
    def create(cls, *, candidate_id: str, artifact_path: str,
               base_generation: str, training_manifest: Mapping[str, Any]):
        return cls(
            candidate_id, str(Path(artifact_path).resolve()), tree_digest(artifact_path),
            base_generation,
            sha256_bytes(canonical_json(dict(training_manifest)).encode()),
            time.time(),
        )

@dataclass(frozen=True)
class QualificationResult:
    new_capability_gain: float
    worst_old_capability_regression: float
    hallucination_delta: float
    latency_ratio: float
    memory_ratio: float
    integrity_ok: bool
    replay_pass: bool
    behavioral_pass: bool
    notes: str = ""

    @property
    def digest(self):
        return sha256_bytes(canonical_json(asdict(self)).encode())

class QualificationPlane:
    def __init__(self, *, min_gain: float = 0.01, max_regression: float = 0.01,
                 max_hallucination_delta: float = 0.01,
                 max_latency_ratio: float = 1.5, max_memory_ratio: float = 1.5):
        self.min_gain = min_gain
        self.max_regression = max_regression
        self.max_hallucination_delta = max_hallucination_delta
        self.max_latency_ratio = max_latency_ratio
        self.max_memory_ratio = max_memory_ratio

    def decide(self, candidate: CandidateArtifact,
               result: QualificationResult) -> CandidateState:
        if tree_digest(candidate.artifact_path) != candidate.artifact_digest or not result.integrity_ok:
            return CandidateState.REJECTED
        if not result.replay_pass or not result.behavioral_pass:
            return CandidateState.QUARANTINED
        if result.new_capability_gain < self.min_gain:
            return CandidateState.REJECTED
        if result.worst_old_capability_regression > self.max_regression:
            return CandidateState.REJECTED
        if result.hallucination_delta > self.max_hallucination_delta:
            return CandidateState.REJECTED
        if result.latency_ratio > self.max_latency_ratio or result.memory_ratio > self.max_memory_ratio:
            return CandidateState.QUARANTINED
        return CandidateState.OFFLINE

    def advance(self, state: CandidateState, *, shadow_pass: bool = False,
                canary_pass: bool = False) -> CandidateState:
        if state == CandidateState.OFFLINE:
            return CandidateState.SHADOW if shadow_pass else CandidateState.QUARANTINED
        if state == CandidateState.SHADOW:
            return CandidateState.CANARY if canary_pass else CandidateState.QUARANTINED
        if state == CandidateState.CANARY:
            return CandidateState.PROMOTED if canary_pass else CandidateState.QUARANTINED
        return state
