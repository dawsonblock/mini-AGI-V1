from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True)
class SignedEvidenceRef:
    kind: str
    object_digest: str
    signer_key_id: str
    decision: str
    schema: str = "mini-agi-v16.1-signed-evidence-ref-v1"

    def __post_init__(self):
        if not self.kind or not self.signer_key_id:
            raise ValueError("kind and signer_key_id required")
        validate_digest(self.object_digest)
        if self.decision not in {"PASS", "BLOCK", "QUALIFIED", "REJECTED"}:
            raise ValueError("unsupported evidence decision")


@dataclass(frozen=True)
class ConvergedEvidenceBundleV161:
    candidate_digest: str
    experiment: SignedEvidenceRef
    reproduction: SignedEvidenceRef
    plasticity: SignedEvidenceRef
    runtime_closure: SignedEvidenceRef
    dream: SignedEvidenceRef | None = None
    schema: str = "mini-agi-v16.1-converged-evidence-bundle-v1"

    def __post_init__(self):
        validate_digest(self.candidate_digest)
        refs = [self.experiment, self.reproduction, self.plasticity, self.runtime_closure]
        if self.dream is not None:
            refs.append(self.dream)
        signer_roles = [x.signer_key_id for x in refs]
        if len(set(signer_roles)) < 3:
            raise PermissionError("converged qualification requires at least three independent signer identities")
        if any(x.decision not in {"PASS", "QUALIFIED"} for x in refs):
            raise PermissionError("all converged evidence must be positive")

    @property
    def digest(self) -> str:
        return digest(self)
