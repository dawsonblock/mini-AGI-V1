from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
import uuid

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class PromotionAuthorizationV141:
    authorization_id: str
    candidate_digest: str
    qualification_digest: str
    authority_generation: int
    policy_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-promotion-authorization-v1"

    def __post_init__(self) -> None:
        validate_digest(self.candidate_digest)
        validate_digest(self.qualification_digest)
        if not self.authorization_id:
            raise ValueError("authorization_id required")
        if self.authority_generation < 0 or self.policy_generation < 0:
            raise ValueError("generations must be non-negative")

    def unsigned(self) -> "PromotionAuthorizationV141":
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self) -> str:
        return digest(self)


class PromotionAuthorityV141:
    """Mints promotion authorization only from authoritative PROMOTE records."""

    def __init__(self, *, governance_db, signer, authority_generation: int, policy_generation: int):
        self.db = governance_db
        self.signer = signer
        self.authority_generation = int(authority_generation)
        self.policy_generation = int(policy_generation)

    def authorize(self, *, candidate_digest: str, qualification_digest: str) -> PromotionAuthorizationV141:
        row = self.db.qualification(qualification_digest)
        if row is None:
            raise PermissionError("qualification not found in authoritative store")
        if row["candidate_digest"] != candidate_digest:
            raise PermissionError("qualification/candidate mismatch")
        if row["decision"] != "PROMOTE":
            raise PermissionError("qualification decision does not permit promotion")
        if int(row["authority_generation"]) != self.authority_generation:
            raise PermissionError("stale authority generation")
        if int(row["policy_generation"]) != self.policy_generation:
            raise PermissionError("stale policy generation")
        auth = PromotionAuthorizationV141(
            authorization_id="PA14-" + uuid.uuid4().hex,
            candidate_digest=candidate_digest,
            qualification_digest=qualification_digest,
            authority_generation=self.authority_generation,
            policy_generation=self.policy_generation,
            issued_at=time.time(),
        )
        env = self.signer.sign(asdict(auth.unsigned()))
        auth = replace(auth, signer_key_id=env.key_id, signature_b64=env.signature_b64)
        self.db.record_promotion(
            candidate_digest=candidate_digest,
            qualification_digest=qualification_digest,
            authorization_digest=auth.digest,
            authority_generation=self.authority_generation,
            policy_generation=self.policy_generation,
            signer_key_id=env.key_id,
        )
        return auth


class PromotionAuthorizationValidatorV141:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation: int, policy_generation: int):
        self.verifier = verifier
        self.trusted = set(str(x) for x in trusted_key_ids)
        self.authority_generation = int(authority_generation)
        self.policy_generation = int(policy_generation)

    def validate(self, auth: PromotionAuthorizationV141, *, candidate_digest: str, qualification_digest: str) -> bool:
        if auth.signer_key_id not in self.trusted:
            raise PermissionError("untrusted promotion key")
        if auth.candidate_digest != candidate_digest or auth.qualification_digest != qualification_digest:
            raise PermissionError("promotion authorization binding mismatch")
        if auth.authority_generation != self.authority_generation or auth.policy_generation != self.policy_generation:
            raise PermissionError("promotion authorization generation mismatch")
        env = SignedEnvelope(auth.signer_key_id, auth.signature_b64)
        if not self.verifier.verify(asdict(auth.unsigned()), env):
            raise PermissionError("invalid promotion authorization signature")
        return True
