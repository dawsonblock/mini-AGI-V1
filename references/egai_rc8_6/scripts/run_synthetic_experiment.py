from dataclasses import asdict
import json
from egai.bench.engine import TaskCase
from egai.bench.sequential import SequentialExperiment
from egai.cognition.model import FrozenModel, FrozenModelIdentity
from egai.cognition.agent import SandboxAdaptiveAgent

class RuleModel(FrozenModel):
    def __init__(self):
        self.identity=FrozenModelIdentity('synthetic-rule',(('weights','sha256:'+'7'*64),),'','','test')
    @property
    def model_digest(self): return self.identity.digest
    def generate(self,prompt):
        task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
        if 'Reverse the input characters.' in prompt: return task[::-1]
        return 'UNKNOWN'

model=RuleModel(); agent=SandboxAdaptiveAgent(model)
meta={'feedback_verified':True,'verifier_id':'synthetic-oracle'}
experience=[
    TaskCase('e1','experience','abc','cba','reverse','Reverse the input characters.',('reverse',),meta),
    TaskCase('e2','experience','cat','tac','reverse','Reverse the input characters.',('reverse',),meta),
]
evaluation=[
    TaskCase('f1','future','dog','god','reverse'),
    TaskCase('r1','retention','plain','UNKNOWN','other'),
]
score=lambda a,b: 1.0 if str(a).strip()==str(b).strip() else 0.0
report=SequentialExperiment(experience,evaluation,score,checkpoints=(0,1,2),bootstrap_samples=500,near_leakage_threshold=None).run(model,agent)
print(json.dumps({'model_digest_constant':report.model_digest_constant,'learned_procedures':report.learned_procedures,'points':[asdict(x) for x in report.points]},indent=2))
