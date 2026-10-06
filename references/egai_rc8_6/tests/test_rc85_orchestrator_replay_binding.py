import tempfile,unittest
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.artifacts import ArtifactStore
from egai.experiment.manifest import ExperimentRegistrar
from egai.experiment.journal import ExperimentJournal
from egai.experiment.orchestrator import ExperimentOrchestrator
from egai.bench.engine import TaskCase
from egai.cognition.model import FrozenModel,FrozenModelIdentity
class Rule(FrozenModel):
 def __init__(self):self.i=FrozenModelIdentity('r',(('w','sha256:'+'7'*64),),'','','test')
 @property
 def model_digest(self):return self.i.digest
 @property
 def backend_digest(self):return 'sha256:'+'8'*64
 def generate(self,prompt):
  task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
  return task[::-1] if 'Reverse characters.' in prompt else 'UNKNOWN'
class T(unittest.TestCase):
 def test_replay_audit_is_bound_into_result_and_closure(self):
  with tempfile.TemporaryDirectory() as d:
   reg=Ed25519Signer.generate('reg');run=Ed25519Signer.generate('run');v=Ed25519Verifier();v.register(reg.key_id,reg.public_bytes());v.register(run.key_id,run.public_bytes())
   m=Rule();exp=[TaskCase('e1','experience','abc','cba','reverse','Reverse characters.'),TaskCase('e2','experience','cat','tac','reverse','Reverse characters.')]
   ev=[TaskCase('f','future','dog','god','reverse'),TaskCase('r','retention','plain','UNKNOWN','other')]
   mf=ExperimentRegistrar('reg',reg).register('x','1',m.model_digest,exp,ev,(0,1,2),learner_config={'bootstrap_samples':100,'synthetic_reference_repairs':True},near_leakage_threshold=None)
   store=ArtifactStore(Path(d)/'cas');j=ExperimentJournal(Path(d)/'j.jsonl',run,v)
   result,report=ExperimentOrchestrator('runner',run,v,(reg.key_id,),j,store,source_root=Path(__file__).parents[1]/'egai').run('res',mf,m,exp,ev,lambda a,b:1. if a==b else 0.)
   self.assertTrue(result.replay_audit_digest); self.assertTrue(store.exists(result.replay_audit_artifact_digest))
   self.assertEqual(result.replay_grounded_coverage,1.0); self.assertEqual(result.replay_prefix_violations,0)
   self.assertIn('replay_audit',[e.event_type for e in j.entries()])
