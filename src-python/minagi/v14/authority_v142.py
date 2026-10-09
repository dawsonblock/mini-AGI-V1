from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
import uuid

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class PromotionAuthorizationV142:
    authorization_id: str
    candidate_digest: str
    qualification_digest: str
    mutation_scopes: tuple[str, ...]
    authority_generation: int
    policy_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha2-promotion-authorization-v1"

    def __post_init__(self):
        validate_digest(self.candidate_digest); validate_digest(self.qualification_digest)
        object.__setattr__(self, "mutation_scopes", tuple(sorted(set(self.mutation_scopes))))
        if not self.mutation_scopes: raise ValueError("mutation scope required")
    def unsigned(self): return replace(self, signer_key_id="", signature_b64="")
    @property
    def digest(self): return digest(self)


class PromotionAuthorityV142:
    def __init__(self, *, governance_db, signer, qualification_validator, authority_generation: int, policy_generation: int):
        self.db=governance_db; self.signer=signer; self.qvalidator=qualification_validator
        self.agen=int(authority_generation); self.pgen=int(policy_generation)

    def authorize(self, *, qualification, mutation_scopes: tuple[str, ...]) -> PromotionAuthorizationV142:
        row=self.db.qualification(qualification.digest)
        if row is None: raise PermissionError("qualification not found in authoritative store")
        self.qvalidator.validate(qualification, candidate_digest=row["candidate_digest"],
                                 evaluation_digest=row["evaluation_digest"], metrics_digest=row["metrics_digest"])
        if qualification.decision != "PROMOTE" or row["decision"] != "PROMOTE":
            raise PermissionError("qualification decision does not permit promotion")
        if int(row["authority_generation"]) != self.agen or int(row["policy_generation"]) != self.pgen:
            raise PermissionError("stale promotion generation")
        auth=PromotionAuthorizationV142("PA142-"+uuid.uuid4().hex, qualification.candidate_digest,
                                       qualification.digest, tuple(mutation_scopes), self.agen, self.pgen, time.time())
        env=self.signer.sign(asdict(auth.unsigned())); auth=replace(auth, signer_key_id=env.key_id, signature_b64=env.signature_b64)
        self.db.record_promotion(candidate_digest=auth.candidate_digest, qualification_digest=auth.qualification_digest,
                                 authorization_digest=auth.digest, authority_generation=self.agen, policy_generation=self.pgen,
                                 signer_key_id=env.key_id, mutation_scopes=auth.mutation_scopes)
        return auth


class PromotionAuthorizationValidatorV142:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation: int, policy_generation: int):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids); self.agen=int(authority_generation); self.pgen=int(policy_generation)
    def validate(self, auth: PromotionAuthorizationV142, *, candidate_digest: str, qualification_digest: str, required_scope: str | None = None) -> bool:
        if auth.signer_key_id not in self.trusted: raise PermissionError("untrusted promotion key")
        if auth.candidate_digest != candidate_digest or auth.qualification_digest != qualification_digest: raise PermissionError("promotion binding mismatch")
        if auth.authority_generation != self.agen or auth.policy_generation != self.pgen: raise PermissionError("promotion generation mismatch")
        if required_scope and required_scope not in set(auth.mutation_scopes): raise PermissionError(f"promotion scope does not permit {required_scope}")
        if not self.verifier.verify(asdict(auth.unsigned()), SignedEnvelope(auth.signer_key_id, auth.signature_b64)):
            raise PermissionError("invalid promotion authorization signature")
        return True


class AuthorizationGateV142:
    """Resolves the actual signed authorization body from CAS and DB before mutation."""
    def __init__(self, *, cas, governance_db, validator: PromotionAuthorizationValidatorV142):
        self.cas=cas; self.db=governance_db; self.validator=validator
    def require(self, authorization_digest: str, *, mutation_scope: str) -> PromotionAuthorizationV142:
        validate_digest(authorization_digest)
        row=self.db.require_authorization(authorization_digest, mutation_scope=mutation_scope)
        try:
            raw=__import__('json').loads(self.cas.get_bytes(authorization_digest).decode('utf-8')); body=raw.get('value', raw)
        except Exception as exc: raise PermissionError("authorization body missing from CAS") from exc
        body["mutation_scopes"]=tuple(body.get("mutation_scopes",()))
        auth=PromotionAuthorizationV142(**body)
        if auth.digest != authorization_digest: raise PermissionError("authorization digest/body mismatch")
        self.validator.validate(auth, candidate_digest=row["candidate_digest"], qualification_digest=row["qualification_digest"], required_scope=mutation_scope)
        return auth
