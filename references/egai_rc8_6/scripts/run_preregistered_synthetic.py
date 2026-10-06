import json,tempfile
from dataclasses import asdict
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.artifacts import ArtifactStore
from egai.experiment.manifest import ExperimentRegistrar
from egai.experiment.journal import ExperimentJournal
from egai.experiment.orchestrator import ExperimentOrchestrator
from egai.experiment.qualify import ContinualLearningQualifier
from egai.bench.engine import TaskCase
from egai.cognition.model import FrozenModel,FrozenModelIdentity

class RuleModel(FrozenModel):
    def __init__(self): self.ident=FrozenModelIdentity('rule-model',(('weights','sha256:'+'9'*64),),'','','synthetic')
    @property
    def model_digest(self): return self.ident.digest
    def generate(self,prompt):
        task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
        return task[::-1] if 'Reverse characters.' in prompt else 'UNKNOWN'

with tempfile.TemporaryDirectory() as d:
    registrar=Ed25519Signer.generate('experiment-registrar'); runner=Ed25519Signer.generate('experiment-runner'); qualifier=Ed25519Signer.generate('research-qualifier')
    verifier=Ed25519Verifier()
    for s in (registrar,runner,qualifier): verifier.register(s.key_id,s.public_bytes())
    model=RuleModel()
    experience=[
      TaskCase('e1','experience','abc','cba','reverse','Reverse characters.',('reverse',)),
      TaskCase('e2','experience','cat','tac','reverse','Reverse characters.',('reverse',)),
    ]
    evaluation=[TaskCase('f1','future','dog','god','reverse'),TaskCase('f2','future','sun','nus','reverse'),TaskCase('r1','retention','plain','UNKNOWN','other'),TaskCase('s1','security','safe','UNKNOWN','other')]
    manifest=ExperimentRegistrar('registrar',registrar).register('EXP-SYNTH','1',model.model_digest,experience,evaluation,(0,1,2),learner_config={'bootstrap_samples':250},near_leakage_threshold=None)
    journal=ExperimentJournal(Path(d)/'experiment.jsonl',runner,verifier)
    orchestrator=ExperimentOrchestrator('runner',runner,verifier,(registrar.key_id,),journal,ArtifactStore(Path(d)/'cas'),feedback_signer=runner,trusted_feedback_keys=(runner.key_id,),source_root=Path(__file__).parents[1]/'egai')
    result,report=orchestrator.run('RESULT-SYNTH',manifest,model,experience,evaluation,lambda a,b:1.0 if str(a).strip()==str(b).strip() else 0.0)
    q=ContinualLearningQualifier('research-qualifier',qualifier,verifier,(runner.key_id,),min_gain=.1,min_verified_repairs=2,require_closure=True).evaluate(result)
    print(json.dumps({'manifest_digest':manifest.digest,'model_digest':model.model_digest,'result':asdict(result),'qualification':asdict(q),'points':[asdict(p) for p in report.points]},indent=2))
