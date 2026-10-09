from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
import uuid

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .qualification_v143 import QualificationRecordV143, QualificationValidatorV143


@dataclass(frozen=True)
class MutationCommitmentV143:
    scope: str
    target_digest: str
    def __post_init__(self):
        if not self.scope: raise ValueError("mutation scope required")
        validate_digest(self.target_digest)


@dataclass(frozen=True)
class PromotionAuthorizationV143:
    authorization_id: str
    candidate_digest: str
    build_digest: str
    evaluation_digest: str
    qualification_digest: str
    runtime_manifest_digest: str
    artifact_root_digest: str
    mutation_commitments: tuple[MutationCommitmentV143,...]
    authority_generation: int
    policy_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha3-promotion-authorization-v1"
    def __post_init__(self):
        for d in (self.candidate_digest,self.build_digest,self.evaluation_digest,self.qualification_digest,
                  self.runtime_manifest_digest,self.artifact_root_digest): validate_digest(d)
        if not self.mutation_commitments: raise ValueError("content-bound mutation commitments required")
        object.__setattr__(self,"mutation_commitments",tuple(sorted(self.mutation_commitments,key=lambda x:(x.scope,x.target_digest))))
    def unsigned(self): return replace(self,signer_key_id="",signature_b64="")
    @property
    def digest(self): return digest(self)


class PromotionAuthorityV143:
    def __init__(self, *, governance_db, signer, qualification_validator:QualificationValidatorV143,
                 authority_generation:int, policy_generation:int):
        self.db=governance_db; self.signer=signer; self.qvalidator=qualification_validator
        self.agen=int(authority_generation); self.pgen=int(policy_generation)
    def authorize(self, *, qualification:QualificationRecordV143, runtime_manifest_digest:str, artifact_root_digest:str,
                  mutation_commitments:tuple[MutationCommitmentV143,...]):
        qrow=self.db.qualification(qualification.digest)
        if qrow is None: raise PermissionError("qualification not authoritative")
        self.qvalidator.validate(qualification,candidate_digest=qrow["candidate_digest"],build_digest=qualification.build_digest,
                                 evaluation_digest=qrow["evaluation_digest"])
        if qualification.decision != "PROMOTE" or qrow["decision"] != "PROMOTE": raise PermissionError("qualification rejects promotion")
        auth=PromotionAuthorizationV143("PA143-"+uuid.uuid4().hex,qualification.candidate_digest,qualification.build_digest,
            qualification.evaluation_digest,qualification.digest,validate_digest(runtime_manifest_digest),validate_digest(artifact_root_digest),
            tuple(mutation_commitments),self.agen,self.pgen,time.time())
        env=self.signer.sign(asdict(auth.unsigned())); auth=replace(auth,signer_key_id=env.key_id,signature_b64=env.signature_b64)
        self.db.record_promotion_v143(candidate_digest=auth.candidate_digest,build_digest=auth.build_digest,evaluation_digest=auth.evaluation_digest,
            qualification_digest=auth.qualification_digest,authorization_digest=auth.digest,runtime_manifest_digest=auth.runtime_manifest_digest,
            artifact_root_digest=auth.artifact_root_digest,authority_generation=self.agen,policy_generation=self.pgen,
            signer_key_id=env.key_id,mutation_commitments=tuple((m.scope,m.target_digest) for m in auth.mutation_commitments))
        return auth


class PromotionAuthorizationValidatorV143:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation:int, policy_generation:int):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids); self.agen=int(authority_generation); self.pgen=int(policy_generation)
    def validate(self,auth:PromotionAuthorizationV143,*,candidate_digest:str,build_digest:str,evaluation_digest:str,
                 qualification_digest:str,runtime_manifest_digest:str,artifact_root_digest:str,required_scope:str|None=None,target_digest:str|None=None):
        if auth.signer_key_id not in self.trusted: raise PermissionError("untrusted promotion key")
        expected=(candidate_digest,build_digest,evaluation_digest,qualification_digest,runtime_manifest_digest,artifact_root_digest)
        actual=(auth.candidate_digest,auth.build_digest,auth.evaluation_digest,auth.qualification_digest,auth.runtime_manifest_digest,auth.artifact_root_digest)
        if actual != expected: raise PermissionError("promotion exact-content binding mismatch")
        if (auth.authority_generation,auth.policy_generation)!=(self.agen,self.pgen): raise PermissionError("promotion generation mismatch")
        if required_scope is not None:
            td=validate_digest(target_digest or runtime_manifest_digest)
            if not any(m.scope==required_scope and m.target_digest==td for m in auth.mutation_commitments):
                raise PermissionError("promotion does not authorize exact requested mutation")
        if not self.verifier.verify(asdict(auth.unsigned()),SignedEnvelope(auth.signer_key_id,auth.signature_b64)):
            raise PermissionError("invalid promotion authorization signature")
        return True


class AuthorizationGateV143:
    def __init__(self, *, cas, governance_db, validator:PromotionAuthorizationValidatorV143):
        self.cas=cas; self.db=governance_db; self.validator=validator
    def _load(self,authorization_digest:str):
        import json
        row=self.db.activation(authorization_digest)
        if row is None: raise PermissionError("promotion authorization not found")
        raw=json.loads(self.cas.get_bytes(validate_digest(authorization_digest)).decode("utf-8")); body=raw.get("value",raw)
        body["mutation_commitments"]=tuple(MutationCommitmentV143(**m) for m in body.get("mutation_commitments",()))
        auth=PromotionAuthorizationV143(**body)
        if auth.digest != authorization_digest: raise PermissionError("authorization digest/body mismatch")
        return auth,row
    def require(self,authorization_digest:str,*,mutation_scope:str,target_digest:str,consume:bool=True):
        auth,row=self._load(authorization_digest); target_digest=validate_digest(target_digest)
        self.db.require_mutation(authorization_digest,mutation_scope=mutation_scope,target_digest=target_digest)
        self.validator.validate(auth,candidate_digest=row["candidate_digest"],build_digest=row["build_digest"],evaluation_digest=row["evaluation_digest"],
            qualification_digest=row["qualification_digest"],runtime_manifest_digest=row["runtime_manifest_digest"],artifact_root_digest=row["artifact_root_digest"],
            required_scope=mutation_scope,target_digest=target_digest)
        if consume:self.db.consume_mutation(authorization_digest,mutation_scope=mutation_scope,target_digest=target_digest)
        return auth
