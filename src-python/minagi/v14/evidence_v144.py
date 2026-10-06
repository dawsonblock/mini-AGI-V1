from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json, time
from typing import Mapping

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .models import EvidenceStrength
from .verification import EpisodeVerificationReceipt


def _load_body(cas, value: str) -> dict:
    raw = json.loads(cas.get_bytes(validate_digest(value)).decode("utf-8"))
    return raw.get("value", raw)


@dataclass(frozen=True)
class EvidenceStageReceiptV144:
    """Signed evidence-stage receipt.

    Stage 2 must be rooted in signed EpisodeVerificationReceipt objects.
    Stages 3-5 must extend a valid stage-(N-1) receipt over the exact same
    evidence set.  This prevents alpha3-style evidence-strength inflation by
    counting arbitrary CAS objects with suggestive names.
    """

    stage_strength: int
    evidence_digests: tuple[str, ...]
    parent_receipt_digests: tuple[str, ...]
    authority_id: str
    authority_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha4-evidence-stage-receipt-v1"

    def __post_init__(self):
        strength = EvidenceStrength(int(self.stage_strength))
        if strength < EvidenceStrength.E2_INDEPENDENTLY_VERIFIED:
            raise ValueError("evidence stage receipts start at E2")
        if not self.evidence_digests or not self.parent_receipt_digests:
            raise ValueError("evidence and parent receipts are required")
        for d in self.evidence_digests + self.parent_receipt_digests:
            validate_digest(d)
        if not self.authority_id or self.authority_generation < 0:
            raise ValueError("invalid evidence stage authority")

    def unsigned(self):
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self):
        return digest(self)


class EvidenceStageValidatorV144:
    def __init__(self, *, verifier, episode_trusted_key_ids, stage_key_policy: Mapping[int, set[str] | tuple[str, ...]],
                 authority_generation: int, cas):
        self.verifier = verifier
        self.episode_keys = set(str(x) for x in episode_trusted_key_ids)
        self.stage_keys = {int(k): set(str(x) for x in v) for k, v in dict(stage_key_policy).items()}
        self.generation = int(authority_generation)
        self.cas = cas

    def _verify_episode_parent(self, value: str, expected_evidence: tuple[str, ...]) -> None:
        body = _load_body(self.cas, value)
        receipt = EpisodeVerificationReceipt(**body)
        if receipt.verifier_key_id not in self.episode_keys:
            raise PermissionError("untrusted episode-verification signer")
        if not self.verifier.verify(asdict(receipt.unsigned()), SignedEnvelope(receipt.verifier_key_id, receipt.signature_b64)):
            raise PermissionError("invalid episode-verification signature")
        if not receipt.passed or receipt.score < receipt.pass_threshold:
            raise PermissionError("episode verification did not pass")
        if receipt.evidence_root_digest not in expected_evidence:
            raise PermissionError("episode verification does not bind evidence root")

    def validate(self, receipt: EvidenceStageReceiptV144, *, expected_evidence_digests: tuple[str, ...] | None = None,
                 _seen: set[str] | None = None) -> EvidenceStrength:
        strength = EvidenceStrength(int(receipt.stage_strength))
        trusted = self.stage_keys.get(int(strength), set())
        if receipt.signer_key_id not in trusted:
            raise PermissionError(f"untrusted E{int(strength)} evidence-stage signer")
        if receipt.authority_generation != self.generation:
            raise PermissionError("evidence-stage generation mismatch")
        if expected_evidence_digests is not None and tuple(receipt.evidence_digests) != tuple(expected_evidence_digests):
            raise PermissionError("evidence-stage receipt evidence mismatch")
        if not self.verifier.verify(asdict(receipt.unsigned()), SignedEnvelope(receipt.signer_key_id, receipt.signature_b64)):
            raise PermissionError("invalid evidence-stage signature")

        seen = set() if _seen is None else _seen
        if receipt.digest in seen:
            raise PermissionError("evidence-stage cycle detected")
        seen.add(receipt.digest)

        if strength == EvidenceStrength.E2_INDEPENDENTLY_VERIFIED:
            covered = set()
            for parent in receipt.parent_receipt_digests:
                self._verify_episode_parent(parent, tuple(receipt.evidence_digests))
                body = _load_body(self.cas, parent)
                covered.add(body["evidence_root_digest"])
            if set(receipt.evidence_digests) - covered:
                raise PermissionError("not every evidence object has an independent verification receipt")
        else:
            parent_strength = int(strength) - 1
            valid_parent = False
            for parent in receipt.parent_receipt_digests:
                body = _load_body(self.cas, parent)
                if body.get("schema") != "mini-agi-v14.1-alpha4-evidence-stage-receipt-v1":
                    continue
                pr = EvidenceStageReceiptV144(**body)
                if int(pr.stage_strength) != parent_strength:
                    continue
                self.validate(pr, expected_evidence_digests=tuple(receipt.evidence_digests), _seen=seen)
                valid_parent = True
            if not valid_parent:
                raise PermissionError(f"E{int(strength)} requires valid E{parent_strength} parent receipt")
        seen.remove(receipt.digest)
        return strength


class EvidenceStageAuthorityV144:
    def __init__(self, *, authority_id: str, authority_generation: int, signer,
                 validator: EvidenceStageValidatorV144, cas):
        self.authority_id = str(authority_id)
        self.generation = int(authority_generation)
        self.signer = signer
        self.validator = validator
        self.cas = cas

    def issue(self, *, stage_strength: int, evidence_digests, parent_receipt_digests) -> EvidenceStageReceiptV144:
        strength = EvidenceStrength(int(stage_strength))
        evidence = tuple(str(x) for x in evidence_digests)
        parents = tuple(str(x) for x in parent_receipt_digests)
        for d in evidence + parents:
            self.cas.get_bytes(validate_digest(d))
        # Before extending the chain, prove all parents are appropriate.
        if strength == EvidenceStrength.E2_INDEPENDENTLY_VERIFIED:
            # Temporary unsigned receipt lets validator enforce parent coverage.
            probe = EvidenceStageReceiptV144(int(strength), evidence, parents, self.authority_id, self.generation, time.time(),
                                            signer_key_id=self.signer.key_id, signature_b64="x")
            # Validate the parents directly; signature of the new receipt does not yet exist.
            covered = set()
            for parent in parents:
                self.validator._verify_episode_parent(parent, evidence)
                covered.add(_load_body(self.cas, parent)["evidence_root_digest"])
            if set(evidence) - covered:
                raise PermissionError("E2 requires independent verification for every evidence object")
        else:
            parent_strength = int(strength) - 1
            ok = False
            for parent in parents:
                body = _load_body(self.cas, parent)
                if body.get("schema") != "mini-agi-v14.1-alpha4-evidence-stage-receipt-v1":
                    continue
                pr = EvidenceStageReceiptV144(**body)
                if int(pr.stage_strength) == parent_strength:
                    self.validator.validate(pr, expected_evidence_digests=evidence)
                    ok = True
            if not ok:
                raise PermissionError(f"E{int(strength)} requires a valid E{parent_strength} parent")
        receipt = EvidenceStageReceiptV144(int(strength), evidence, parents, self.authority_id, self.generation, time.time())
        env = self.signer.sign(asdict(receipt.unsigned()))
        return replace(receipt, signer_key_id=env.key_id, signature_b64=env.signature_b64)


@dataclass(frozen=True)
class EvidenceStrengthProofV144:
    evidence_digests: tuple[str, ...]
    highest_stage_receipt_digest: str
    derived_strength: int
    authority_id: str
    authority_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha4-evidence-strength-proof-v1"

    def __post_init__(self):
        for d in self.evidence_digests:
            validate_digest(d)
        validate_digest(self.highest_stage_receipt_digest)
        EvidenceStrength(int(self.derived_strength))
        if not self.authority_id or self.authority_generation < 0:
            raise ValueError("invalid evidence-strength authority")

    def unsigned(self):
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self):
        return digest(self)


class EvidenceStrengthAuthorityV144:
    def __init__(self, *, authority_id: str, authority_generation: int, signer,
                 stage_validator: EvidenceStageValidatorV144, cas):
        self.authority_id = str(authority_id)
        self.generation = int(authority_generation)
        self.signer = signer
        self.stage_validator = stage_validator
        self.cas = cas

    def issue(self, *, evidence_digests, highest_stage_receipt_digest: str) -> EvidenceStrengthProofV144:
        evidence = tuple(str(x) for x in evidence_digests)
        body = _load_body(self.cas, highest_stage_receipt_digest)
        stage = EvidenceStageReceiptV144(**body)
        strength = self.stage_validator.validate(stage, expected_evidence_digests=evidence)
        proof = EvidenceStrengthProofV144(evidence, validate_digest(highest_stage_receipt_digest), int(strength),
                                          self.authority_id, self.generation, time.time())
        env = self.signer.sign(asdict(proof.unsigned()))
        return replace(proof, signer_key_id=env.key_id, signature_b64=env.signature_b64)


class EvidenceStrengthValidatorV144:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation: int,
                 stage_validator: EvidenceStageValidatorV144, cas):
        self.verifier = verifier
        self.trusted = set(str(x) for x in trusted_key_ids)
        self.generation = int(authority_generation)
        self.stage_validator = stage_validator
        self.cas = cas

    def validate(self, proof: EvidenceStrengthProofV144, *, expected_evidence_digests: tuple[str, ...]) -> EvidenceStrength:
        if proof.signer_key_id not in self.trusted:
            raise PermissionError("untrusted evidence-strength signer")
        if proof.authority_generation != self.generation:
            raise PermissionError("evidence-strength generation mismatch")
        if tuple(proof.evidence_digests) != tuple(expected_evidence_digests):
            raise PermissionError("evidence-strength proof does not bind proposal evidence")
        if not self.verifier.verify(asdict(proof.unsigned()), SignedEnvelope(proof.signer_key_id, proof.signature_b64)):
            raise PermissionError("invalid evidence-strength signature")
        body = _load_body(self.cas, proof.highest_stage_receipt_digest)
        stage = EvidenceStageReceiptV144(**body)
        derived = self.stage_validator.validate(stage, expected_evidence_digests=tuple(proof.evidence_digests))
        if int(derived) != int(proof.derived_strength):
            raise PermissionError("evidence-strength proof overclaims receipt chain")
        return derived
