from dataclasses import dataclass,replace
from typing import Any
import math
from egai.common.canonical import digest,validate_digest

@dataclass(frozen=True)
class CandidateManifest:
    candidate_id:str; proposal_digest:str; artifact_digest:str
    def __post_init__(self):validate_digest(self.proposal_digest);validate_digest(self.artifact_digest)
    @property
    def digest(self):return digest(self)

@dataclass(frozen=True)
class BuildManifest:
    build_id:str; candidate_digest:str; environment_digest:str; output_digest:str
    builder_id:str=''; builder_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.candidate_digest);validate_digest(self.environment_digest);validate_digest(self.output_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,builder_key_id='',signature_b64='')

@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id:str; version:str; task_set_digest:str; retention_set_digest:str; security_set_digest:str
    scorer_id:str='exact_match'; harness_version:str='egai-benchmark-v1'
    preregistered:bool=True; registrar_id:str=''; benchmark_key_id:str=''; signature_b64:str=''
    def __post_init__(self):
        for d in (self.task_set_digest,self.retention_set_digest,self.security_set_digest):validate_digest(d)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,benchmark_key_id='',signature_b64='')

@dataclass(frozen=True)
class RawResult:
    case_id:str; split:str; baseline_score:float; candidate_score:float
    baseline_calibration:float=0.; candidate_calibration:float=0.; security_regression:bool=False
    authorized_change:bool=True; baseline_resource:float=1.; candidate_resource:float=1.; evidence_digest:str=''
    def __post_init__(self):
        if self.split not in ('future','retention','security'): raise ValueError('invalid result split')
        for name in ('baseline_score','candidate_score','baseline_calibration','candidate_calibration','baseline_resource','candidate_resource'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError('finite nonnegative result metrics required')
        if self.split=='security' and self.candidate_score<self.baseline_score and not self.security_regression:
            raise ValueError('security regression cannot be suppressed')


@dataclass(frozen=True)
class ResultBundle:
    run_id:str; benchmark_digest:str; build_digest:str; results:tuple[RawResult,...]; runner_id:str
    runner_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.benchmark_digest);validate_digest(self.build_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,runner_key_id='',signature_b64='')

@dataclass(frozen=True)
class EvaluationBundle:
    evaluation_id:str; build_digest:str; benchmark_digest:str; result_bundle_digest:str
    metrics:dict[str,Any]; provenance_closure:float; evaluator_id:str
    evaluator_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.build_digest);validate_digest(self.benchmark_digest);validate_digest(self.result_bundle_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,evaluator_key_id='',signature_b64='')

@dataclass(frozen=True)
class QualificationRecord:
    qualification_id:str; evaluation_digest:str; passed:bool; reasons:tuple[str,...]; qualifier_id:str
    qualifier_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.evaluation_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,qualifier_key_id='',signature_b64='')

@dataclass(frozen=True)
class PromotionDecision:
    decision_id:str; qualification_digest:str; approved:bool; authority_id:str; decision_nonce:str
    authority_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.qualification_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,authority_key_id='',signature_b64='')

@dataclass(frozen=True)
class RuntimeManifest:
    runtime_id:str; decision_digest:str; base_model_digest:str; component_digests:tuple[str,...]
    previous_runtime_digest:str=''
    def __post_init__(self):
        validate_digest(self.decision_digest);validate_digest(self.base_model_digest)
        for d in self.component_digests:validate_digest(d)
        if self.previous_runtime_digest:validate_digest(self.previous_runtime_digest)
    @property
    def digest(self):return digest(self)

@dataclass(frozen=True)
class RollbackAuthorization:
    rollback_id:str; from_runtime_digest:str; to_runtime_digest:str; reason:str; authority_id:str
    authority_key_id:str=''; signature_b64:str=''
    def __post_init__(self):validate_digest(self.from_runtime_digest);validate_digest(self.to_runtime_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,authority_key_id='',signature_b64='')
