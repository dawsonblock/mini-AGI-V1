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
  task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0];return task[::-1] if 'Reverse characters.' in prompt else 'UNKNOWN'
class T(unittest.TestCase):
 def test_evidence_beliefs_and_learning_packages_are_bound(self):
  with tempfile.TemporaryDirectory() as d:
   reg=Ed25519Signer.generate('reg');run=Ed25519Signer.generate('run');v=Ed25519Verifier();v.register(reg.key_id,reg.public_bytes());v.register(run.key_id,run.public_bytes());m=Rule()
   exp=[TaskCase('e1','experience','abc','cba','reverse'),TaskCase('e2','experience','cat','tac','reverse')];ev=[TaskCase('f','future','dog','god','reverse'),TaskCase('r','retention','plain','UNKNOWN','other')]
   mf=ExperimentRegistrar('reg',reg).register('x','1',m.model_digest,exp,ev,(0,1,2),learner_config={'synthetic_reference_repairs':True},near_leakage_threshold=None)
   store=ArtifactStore(Path(d)/'cas');j=ExperimentJournal(Path(d)/'j.jsonl',run,v);result,report=ExperimentOrchestrator('runner',run,v,(reg.key_id,),j,store).run('res',mf,m,exp,ev,lambda a,b:1. if a==b else 0.)
   self.assertTrue(result.evidence_head_digest);self.assertTrue(result.belief_snapshot_digest);self.assertEqual(len(result.learning_package_digests),1);self.assertTrue(store.exists(result.learning_package_digests[0]))
   events=[x.event_type for x in j.entries()];self.assertIn('evidence_closure',events);self.assertIn('learning_proposal',events)
   self.assertTrue(report.traces[0].cognitive_actions);self.assertIn('verify',report.traces[0].cognitive_actions)
