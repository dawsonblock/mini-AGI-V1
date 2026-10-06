import tempfile,unittest
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.artifacts import ArtifactStore
from egai.experiment.manifest import ExperimentRegistrar
from egai.experiment.journal import ExperimentJournal
from egai.experiment.orchestrator import ExperimentOrchestrator
from egai.bench.engine import TaskCase
from egai.cognition.model import FrozenEchoModel
class T(unittest.TestCase):
 def test_no_implicit_answer_key_repair(self):
  with tempfile.TemporaryDirectory() as d:
   reg=Ed25519Signer.generate('reg');run=Ed25519Signer.generate('run');v=Ed25519Verifier();v.register(reg.key_id,reg.public_bytes());v.register(run.key_id,run.public_bytes());m=FrozenEchoModel()
   exp=[TaskCase('e','experience','x','y','k')];ev=[TaskCase('f','future','z','z','k'),TaskCase('r','retention','q','q','k')]
   mf=ExperimentRegistrar('reg',reg).register('x','1',m.model_digest,exp,ev,(0,1),learner_config={},near_leakage_threshold=None)
   o=ExperimentOrchestrator('runner',run,v,(reg.key_id,),ExperimentJournal(Path(d)/'j',run,v),ArtifactStore(Path(d)/'cas'))
   with self.assertRaises(RuntimeError):o.run('res',mf,m,exp,ev,lambda a,b:1. if a==b else 0.)
