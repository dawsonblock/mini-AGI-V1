import tempfile,unittest
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.artifacts import ArtifactStore
from egai.experiment.manifest import ExperimentRegistrar
from egai.experiment.journal import ExperimentJournal
from egai.experiment.orchestrator import ExperimentOrchestrator
from egai.experiment.qualify import ContinualLearningQualifier
from egai.bench.engine import TaskCase
from egai.cognition.model import FrozenModel,FrozenModelIdentity

class Rule(FrozenModel):
 def __init__(self):self.i=FrozenModelIdentity('r',(('w','sha256:'+'7'*64),),'','','test')
 @property
 def model_digest(self):return self.i.digest
 @property
 def backend_digest(self): return 'sha256:'+'8'*64
 def generate(self,prompt):
  task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
  return task[::-1] if 'Reverse characters.' in prompt else 'UNKNOWN'

class T(unittest.TestCase):
 def test_closure_and_signed_feedback_are_required_capabilities(self):
  with tempfile.TemporaryDirectory() as d:
   reg=Ed25519Signer.generate('reg'); run=Ed25519Signer.generate('run'); qsign=Ed25519Signer.generate('q')
   v=Ed25519Verifier()
   for s in (reg,run,qsign): v.register(s.key_id,s.public_bytes())
   m=Rule()
   exp=[TaskCase('e1','experience','abc','cba','reverse','Reverse characters.'),TaskCase('e2','experience','cat','tac','reverse','Reverse characters.')]
   ev=[TaskCase('f','future','dog','god','reverse'),TaskCase('r','retention','plain','UNKNOWN','other')]
   mf=ExperimentRegistrar('reg',reg).register('x','1',m.model_digest,exp,ev,(0,1,2),learner_config={'bootstrap_samples':100,'synthetic_reference_repairs':True},near_leakage_threshold=None)
   j=ExperimentJournal(Path(d)/'j.jsonl',run,v); store=ArtifactStore(Path(d)/'cas')
   result,report=ExperimentOrchestrator('runner',run,v,(reg.key_id,),j,store,feedback_signer=run,trusted_feedback_keys=(run.key_id,),source_root=Path(__file__).parents[1]/'egai').run('res',mf,m,exp,ev,lambda a,b:1. if a==b else 0.)
   self.assertTrue(result.closure_digest); self.assertTrue(store.exists(result.closure_artifact_digest)); self.assertGreaterEqual(result.verified_repairs,2)
   events=[x.event_type for x in j.entries()]
   self.assertEqual(events.count('verification_receipt'),2)
   q=ContinualLearningQualifier('q',qsign,v,(run.key_id,),min_gain=.01,min_verified_repairs=2,require_closure=True).evaluate(result)
   self.assertTrue(q.passed,q.reasons)
