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
 def generate(self,prompt):
  task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
  return task[::-1] if 'Reverse characters.' in prompt else 'UNKNOWN'
class T(unittest.TestCase):
 def test_preregistered_run(self):
  with tempfile.TemporaryDirectory() as d:
   reg=Ed25519Signer.generate('reg');run=Ed25519Signer.generate('run');v=Ed25519Verifier();v.register(reg.key_id,reg.public_bytes());v.register(run.key_id,run.public_bytes())
   m=Rule();meta={'feedback_verified':True,'verifier_id':'oracle'}
   exp=[TaskCase('e1','experience','abc','cba','reverse','Reverse characters.',('reverse',),meta,True),TaskCase('e2','experience','cat','tac','reverse','Reverse characters.',('reverse',),meta,True)]
   ev=[TaskCase('f','future','dog','god','reverse'),TaskCase('r','retention','plain','UNKNOWN','other')]
   mf=ExperimentRegistrar('reg',reg).register('x','1',m.model_digest,exp,ev,(0,1,2),learner_config={'bootstrap_samples':100,'synthetic_reference_repairs':True},near_leakage_threshold=None)
   j=ExperimentJournal(Path(d)/'j.jsonl',run,v);o=ExperimentOrchestrator('runner',run,v,(reg.key_id,),j,ArtifactStore(Path(d)/'cas'))
   result,report=o.run('res',mf,m,exp,ev,lambda a,b:1. if a==b else 0.)
   self.assertTrue(result.model_digest_constant);self.assertEqual(result.final_future_gain,1.0);self.assertTrue(j.verify(mf.digest));self.assertEqual(report.learned_procedures,1)
