from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Iterable

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .models import EvidenceStrength


@dataclass(frozen=True)
class EvidenceStrengthProofV143:
    evidence_digests: tuple[str, ...]
    verification_receipt_digests: tuple[str, ...]
    replication_receipt_digests: tuple[str, ...]
    fresh_ood_receipt_digests: tuple[str, ...]
    reproduction_receipt_digests: tuple[str, ...]
    derived_strength: int
    authority_id: str
    authority_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha3-evidence-strength-proof-v1"

    def __post_init__(self):
        for seq in (self.evidence_digests, self.verification_receipt_digests,
                    self.replication_receipt_digests, self.fresh_ood_receipt_digests,
                    self.reproduction_receipt_digests):
            for d in seq: validate_digest(d)
        EvidenceStrength(int(self.derived_strength))
        if not self.authority_id or self.authority_generation < 0:
            raise ValueError("invalid evidence authority identity/generation")

    def unsigned(self):
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self):
        return digest(self)


def derive_strength(*, evidence_count: int, verification_count: int, replication_count: int,
                    fresh_ood_count: int, reproduction_count: int) -> EvidenceStrength:
    if evidence_count <= 0:
        return EvidenceStrength.E0_UNVERIFIED
    strength = EvidenceStrength.E1_INTERNALLY_CONSISTENT
    if verification_count > 0:
        strength = EvidenceStrength.E2_INDEPENDENTLY_VERIFIED
    if verification_count > 0 and replication_count > 0:
        strength = EvidenceStrength.E3_REPLICATED
    if verification_count > 0 and replication_count > 0 and fresh_ood_count > 0:
        strength = EvidenceStrength.E4_FRESH_OOD_VALIDATED
    if verification_count > 0 and replication_count > 0 and fresh_ood_count > 0 and reproduction_count > 0:
        strength = EvidenceStrength.E5_INDEPENDENTLY_REPRODUCED
    return strength


class EvidenceStrengthAuthorityV143:
    """Independent authority that derives evidence strength from existing artifacts.

    The proposer cannot choose E0-E5. The authority checks that every referenced
    artifact exists in CAS and signs the derived level.
    """
    def __init__(self, *, authority_id: str, authority_generation: int, signer, cas):
        self.authority_id=str(authority_id); self.authority_generation=int(authority_generation)
        self.signer=signer; self.cas=cas

    def _require_all(self, values: Iterable[str]) -> tuple[str, ...]:
        out=tuple(str(x) for x in values)
        for d in out:
            validate_digest(d)
            self.cas.get_bytes(d)
        return out

    def issue(self, *, evidence_digests, verification_receipt_digests=(), replication_receipt_digests=(),
              fresh_ood_receipt_digests=(), reproduction_receipt_digests=()) -> EvidenceStrengthProofV143:
        evidence=self._require_all(evidence_digests)
        verification=self._require_all(verification_receipt_digests)
        replication=self._require_all(replication_receipt_digests)
        fresh=self._require_all(fresh_ood_receipt_digests)
        reproduction=self._require_all(reproduction_receipt_digests)
        strength=derive_strength(evidence_count=len(evidence), verification_count=len(verification),
                                 replication_count=len(replication), fresh_ood_count=len(fresh),
                                 reproduction_count=len(reproduction))
        proof=EvidenceStrengthProofV143(evidence,verification,replication,fresh,reproduction,int(strength),
                                       self.authority_id,self.authority_generation,time.time())
        env=self.signer.sign(asdict(proof.unsigned()))
        return replace(proof, signer_key_id=env.key_id, signature_b64=env.signature_b64)


class EvidenceStrengthValidatorV143:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation: int, cas):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids)
        self.generation=int(authority_generation); self.cas=cas

    def validate(self, proof: EvidenceStrengthProofV143, *, expected_evidence_digests: tuple[str,...]) -> EvidenceStrength:
        if proof.signer_key_id not in self.trusted:
            raise PermissionError("untrusted evidence-strength signer")
        if proof.authority_generation != self.generation:
            raise PermissionError("evidence-strength generation mismatch")
        if tuple(proof.evidence_digests) != tuple(expected_evidence_digests):
            raise PermissionError("evidence-strength proof does not bind proposal evidence")
        for seq in (proof.evidence_digests, proof.verification_receipt_digests, proof.replication_receipt_digests,
                    proof.fresh_ood_receipt_digests, proof.reproduction_receipt_digests):
            for d in seq:
                try: self.cas.get_bytes(d)
                except Exception as exc: raise PermissionError("evidence-strength proof references missing artifact") from exc
        derived=derive_strength(evidence_count=len(proof.evidence_digests),
                                verification_count=len(proof.verification_receipt_digests),
                                replication_count=len(proof.replication_receipt_digests),
                                fresh_ood_count=len(proof.fresh_ood_receipt_digests),
                                reproduction_count=len(proof.reproduction_receipt_digests))
        if int(derived) != int(proof.derived_strength):
            raise PermissionError("evidence-strength proof overclaims derived level")
        if not self.verifier.verify(asdict(proof.unsigned()), SignedEnvelope(proof.signer_key_id, proof.signature_b64)):
            raise PermissionError("invalid evidence-strength signature")
        return derived
