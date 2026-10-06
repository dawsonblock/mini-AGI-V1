from dataclasses import dataclass
from .model import EvidenceRecord,Origin,Verification

@dataclass(frozen=True)
class EpisodeEvidence:
    task_hash:str; attempt_hash:str; verification_hash:str; repair_hash:str; outcome_hash:str
    @property
    def root(self): return self.outcome_hash

class EvidenceSession:
    """Authoritative episode ingestion. Model statements remain inference; verifier/repair outcomes are separate records."""
    def __init__(self,ledger,model_digest='',environment_digest=''):
        self.ledger=ledger;self.model_digest=model_digest;self.environment_digest=environment_digest
    def _add(self,rid,payload,origin,verified=True,parents=(),source='',tool_digest=''):
        return self.ledger.append(EvidenceRecord.now(rid,payload,origin,source_identity=source,environment_digest=self.environment_digest,model_digest=self.model_digest,tool_digest=tool_digest,parent_evidence=tuple(parents),verification_state=Verification.VERIFIED if verified else Verification.UNVERIFIED))
    def record(self,case,attempted_output,attempt_score,repair,receipt,repaired_score):
        task=self._add('task:'+case.case_id,{'kind':'task','case_id':case.case_id,'task_kind':case.task_kind,'input':str(case.input)},Origin.ENVIRONMENT,True,source='benchmark-environment')
        attempt=self._add('attempt:'+case.case_id,{'kind':'model_attempt','output':str(attempted_output),'score':float(attempt_score)},Origin.MODEL_INFERENCE,False,(task.record_hash,),source='frozen-model')
        ver=self._add('verification:'+case.case_id,{'kind':'verification_receipt','receipt_digest':receipt.digest,'verifier_id':receipt.verifier_id,'passed':receipt.passed},Origin.DETERMINISTIC_TOOL,True,(attempt.record_hash,),source=receipt.verifier_id)
        rep=self._add('repair:'+case.case_id,{'kind':'verified_repair','output':repair.output_text,'provider_id':repair.provider_id,'provider_provenance':repair.provenance_digest},Origin.DETERMINISTIC_TOOL,True,(ver.record_hash,),source=repair.provider_id)
        out=self._add('outcome:'+case.case_id,{'kind':'episode_outcome','attempt_score':float(attempt_score),'repair_score':float(repaired_score),'claim':f'verified repair succeeds for task kind {case.task_kind}','stance':'support'},Origin.DETERMINISTIC_TOOL,True,(rep.record_hash,),source=receipt.verifier_id)
        return EpisodeEvidence(task.record_hash,attempt.record_hash,ver.record_hash,rep.record_hash,out.record_hash)
