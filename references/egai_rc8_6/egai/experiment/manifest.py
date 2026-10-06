from dataclasses import dataclass, asdict, replace
from typing import Any
from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from egai.bench.datasets import cases_digest

@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id:str
    version:str
    model_digest:str
    experience_digest:str
    evaluation_digest:str
    checkpoints:tuple[int,...]
    scorer_id:str
    learner_config:dict[str,Any]
    near_leakage_threshold:float|None
    seed:int
    registrar_id:str
    registrar_key_id:str=''
    signature_b64:str=''
    def __post_init__(self):
        for d in (self.model_digest,self.experience_digest,self.evaluation_digest): validate_digest(d)
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError('checkpoints must be sorted unique and non-empty')
        if self.checkpoints[0] != 0: raise ValueError('checkpoint 0 required')
    @property
    def digest(self): return digest(self)
    def unsigned(self): return replace(self,registrar_key_id='',signature_b64='')

class ExperimentRegistrar:
    def __init__(self,registrar_id,signer): self.registrar_id=registrar_id; self.signer=signer
    def register(self,experiment_id,version,model_digest,experience_cases,evaluation_cases,
                 checkpoints,scorer_id='exact_match',learner_config=None,near_leakage_threshold=.95,seed=0):
        u=ExperimentManifest(experiment_id,version,model_digest,cases_digest(experience_cases),
             cases_digest(evaluation_cases),tuple(sorted(set(int(x) for x in checkpoints))),scorer_id,
             dict(learner_config or {}),near_leakage_threshold,int(seed),self.registrar_id)
        env=self.signer.sign(asdict(u)); return replace(u,registrar_key_id=env.key_id,signature_b64=env.signature_b64)

def verify_manifest(manifest,verifier,trusted_key_ids,model_digest,experience_cases,evaluation_cases):
    if manifest.registrar_key_id not in set(trusted_key_ids): raise PermissionError('untrusted experiment registrar')
    if not verifier.verify(asdict(manifest.unsigned()),SignedEnvelope(manifest.registrar_key_id,manifest.signature_b64)):
        raise PermissionError('invalid experiment manifest signature')
    if manifest.model_digest != model_digest: raise ValueError('model digest differs from preregistration')
    if manifest.experience_digest != cases_digest(experience_cases): raise ValueError('experience stream differs from preregistration')
    if manifest.evaluation_digest != cases_digest(evaluation_cases): raise ValueError('evaluation set differs from preregistration')
    return True
