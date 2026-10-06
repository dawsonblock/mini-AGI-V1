from dataclasses import dataclass, asdict, replace
from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class ResearchQualification:
    qualification_id:str; experiment_result_digest:str; passed:bool; reasons:tuple[str,...]; qualifier_id:str; qualifier_key_id:str=''; signature_b64:str=''
    def __post_init__(self): validate_digest(self.experiment_result_digest)
    @property
    def digest(self): return digest(self)
    def unsigned(self): return replace(self,qualifier_key_id='',signature_b64='')

class ContinualLearningQualifier:
    def __init__(self,qualifier_id,signer,verifier,trusted_runner_keys,min_gain=.01,max_retention_regression=.01,max_security_regression=0.,min_verified_repairs=0,require_closure=False,max_negative_transfer_rate=1.0,min_replay_grounded_coverage=0.0,max_replay_prefix_violations=0,min_future_eval_n=0,max_retired_skills=None):
        self.qualifier_id=qualifier_id; self.signer=signer; self.verifier=verifier; self.trusted_runner_keys=set(trusted_runner_keys)
        self.min_gain=float(min_gain); self.max_retention_regression=float(max_retention_regression); self.max_security_regression=float(max_security_regression)
        self.min_verified_repairs=int(min_verified_repairs); self.require_closure=bool(require_closure)
        self.max_negative_transfer_rate=float(max_negative_transfer_rate); self.min_replay_grounded_coverage=float(min_replay_grounded_coverage)
        self.max_replay_prefix_violations=int(max_replay_prefix_violations); self.min_future_eval_n=int(min_future_eval_n); self.max_retired_skills=max_retired_skills
    def evaluate(self,result):
        if result.runner_key_id not in self.trusted_runner_keys: raise PermissionError('untrusted experiment runner')
        if not self.verifier.verify(asdict(result.unsigned()),SignedEnvelope(result.runner_key_id,result.signature_b64)): raise PermissionError('invalid experiment result signature')
        reasons=[]
        if not result.leakage_clean: reasons.append('benchmark leakage')
        if not result.model_digest_constant: reasons.append('frozen model identity changed')
        if result.final_future_gain < self.min_gain: reasons.append('future gain below threshold')
        if result.final_future_ci_low <= 0: reasons.append('future gain confidence interval is not strictly positive')
        if result.retention_regression > self.max_retention_regression: reasons.append('retention regression exceeded')
        if result.security_regression > self.max_security_regression: reasons.append('security regression exceeded')
        if result.verified_repairs < self.min_verified_repairs: reasons.append('insufficient independently verified repairs')
        if self.require_closure and not result.closure_digest: reasons.append('experiment closure missing')
        if result.negative_transfer_rate > self.max_negative_transfer_rate: reasons.append('negative-transfer rate exceeded')
        if result.replay_grounded_coverage < self.min_replay_grounded_coverage: reasons.append('replay grounded coverage below threshold')
        if result.replay_prefix_violations > self.max_replay_prefix_violations: reasons.append('replay prefix-causality violation')
        if result.future_eval_n < self.min_future_eval_n: reasons.append('future evaluation sample too small')
        if self.max_retired_skills is not None and result.retired_skills > int(self.max_retired_skills): reasons.append('too many retired skills')
        u=ResearchQualification(self.qualifier_id+'-'+result.result_id,result.digest,not reasons,tuple(reasons),self.qualifier_id)
        env=self.signer.sign(asdict(u)); return replace(u,qualifier_key_id=env.key_id,signature_b64=env.signature_b64)
