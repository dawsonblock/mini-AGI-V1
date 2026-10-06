from dataclasses import dataclass,replace
from egai.common.canonical import digest,validate_digest
@dataclass(frozen=True)
class ExperimentResult:
    result_id:str;experiment_digest:str;model_digest:str;final_skill_snapshot_digest:str;journal_head_digest:str
    leakage_clean:bool;model_digest_constant:bool;final_future_gain:float;final_future_ci_low:float;final_future_ci_high:float
    retention_regression:float;security_regression:float;learned_procedures:int;runner_id:str
    closure_digest:str='';closure_artifact_digest:str='';environment_digest:str='';backend_digest:str='';verified_repairs:int=0;skill_failures:int=0
    negative_transfer_rate:float=0.;replay_audit_digest:str='';replay_audit_artifact_digest:str='';replay_grounded_coverage:float=0.
    replay_prefix_violations:int=0;future_eval_n:int=0;degraded_skills:int=0;retired_skills:int=0
    evidence_head_digest:str='';belief_snapshot_digest:str='';learning_package_digests:tuple[str,...]=()
    runner_key_id:str='';signature_b64:str=''
    def __post_init__(self):
        for d in (self.experiment_digest,self.model_digest,self.final_skill_snapshot_digest,self.journal_head_digest):validate_digest(d)
        for d in (self.closure_digest,self.closure_artifact_digest,self.environment_digest,self.backend_digest,self.replay_audit_digest,self.replay_audit_artifact_digest,self.evidence_head_digest,self.belief_snapshot_digest):
            if d:validate_digest(d)
        for d in self.learning_package_digests:validate_digest(d)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,runner_key_id='',signature_b64='')
