from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .evaluation_v143 import SignedEvaluationBundleV143, EvaluationValidatorV143, derive_metrics
from .qualification_v142 import QualificationPolicyV142


@dataclass(frozen=True)
class QualificationRecordV143:
    candidate_digest: str
    build_digest: str
    evaluation_digest: str
    raw_results_digest: str
    derived_metrics_digest: str
    decision: str
    reasons: tuple[str,...]
    qualifier_id: str
    qualifier_generation: int
    authority_generation: int
    policy_generation: int
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha3-qualification-record-v1"
    def __post_init__(self):
        for d in (self.candidate_digest,self.build_digest,self.evaluation_digest,self.raw_results_digest,self.derived_metrics_digest): validate_digest(d)
        if self.decision not in {"PROMOTE","REJECT"}: raise ValueError("invalid qualification decision")
    def unsigned(self): return replace(self,signer_key_id="",signature_b64="")
    @property
    def digest(self): return digest(self)


class QualificationAuthorityV143:
    def __init__(self, *, qualifier_id:str, qualifier_generation:int, authority_generation:int, policy_generation:int,
                 signer, evaluation_validator:EvaluationValidatorV143, cas, policy:QualificationPolicyV142|None=None):
        self.qualifier_id=str(qualifier_id); self.qualifier_generation=int(qualifier_generation)
        self.authority_generation=int(authority_generation); self.policy_generation=int(policy_generation)
        self.signer=signer; self.evaluation_validator=evaluation_validator; self.cas=cas; self.policy=policy or QualificationPolicyV142()
    def qualify(self,bundle:SignedEvaluationBundleV143,*,candidate_digest:str,build_digest:str)->QualificationRecordV143:
        self.evaluation_validator.validate(bundle,candidate_digest=candidate_digest,build_digest=build_digest)
        metrics=derive_metrics(bundle.case_results)
        md=self.cas.put_json({"schema":"mini-agi-v14.1-alpha3-derived-evaluation-metrics-v1","metrics":metrics})
        decision,reasons=self.policy.decide(metrics)
        rec=QualificationRecordV143(candidate_digest,build_digest,bundle.digest,bundle.raw_results_digest,md,decision,reasons,
                                    self.qualifier_id,self.qualifier_generation,self.authority_generation,self.policy_generation,time.time())
        env=self.signer.sign(asdict(rec.unsigned()))
        return replace(rec,signer_key_id=env.key_id,signature_b64=env.signature_b64)


class QualificationValidatorV143:
    def __init__(self, *, verifier, trusted_key_ids, qualifier_generation:int, authority_generation:int, policy_generation:int):
        self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids); self.qgen=int(qualifier_generation); self.agen=int(authority_generation); self.pgen=int(policy_generation)
    def validate(self,rec:QualificationRecordV143,*,candidate_digest:str,build_digest:str,evaluation_digest:str)->bool:
        if rec.signer_key_id not in self.trusted: raise PermissionError("untrusted qualifier key")
        if (rec.qualifier_generation,rec.authority_generation,rec.policy_generation)!=(self.qgen,self.agen,self.pgen): raise PermissionError("qualification generation mismatch")
        if (rec.candidate_digest,rec.build_digest,rec.evaluation_digest)!=(candidate_digest,build_digest,evaluation_digest): raise PermissionError("qualification binding mismatch")
        if not self.verifier.verify(asdict(rec.unsigned()),SignedEnvelope(rec.signer_key_id,rec.signature_b64)): raise PermissionError("invalid qualification signature")
        return True
