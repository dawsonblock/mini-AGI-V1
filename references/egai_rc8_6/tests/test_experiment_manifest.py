import tempfile,unittest
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.experiment.manifest import ExperimentRegistrar,verify_manifest
from egai.bench.engine import TaskCase
from egai.cognition.model import FrozenEchoModel
class T(unittest.TestCase):
 def test_manifest_binds_model_and_datasets(self):
  s=Ed25519Signer.generate('exp');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes());m=FrozenEchoModel()
  e=[TaskCase('e','experience','a','b')];q=[TaskCase('f','future','x','y'),TaskCase('r','retention','z','z')]
  mf=ExperimentRegistrar('r',s).register('x','1',m.model_digest,e,q,(0,1))
  self.assertTrue(verify_manifest(mf,v,(s.key_id,),m.model_digest,e,q))
  with self.assertRaises(ValueError):verify_manifest(mf,v,(s.key_id,),FrozenEchoModel('other').model_digest,e,q)
