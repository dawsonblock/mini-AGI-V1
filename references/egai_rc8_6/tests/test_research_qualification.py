import unittest
from dataclasses import asdict,replace
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.experiment.result import ExperimentResult
from egai.experiment.qualify import ContinualLearningQualifier
class T(unittest.TestCase):
 def _r(self,s,v,gain=.1,lo=.02,ret=0,sec=0):
  u=ExperimentResult('r','sha256:'+'1'*64,'sha256:'+'2'*64,'sha256:'+'3'*64,'sha256:'+'4'*64,True,True,gain,lo,.2,ret,sec,2,'runner')
  e=s.sign(asdict(u));return replace(u,runner_key_id=e.key_id,signature_b64=e.signature_b64)
 def test_pass_and_fail(self):
  r=Ed25519Signer.generate('runner');q=Ed25519Signer.generate('q');v=Ed25519Verifier();v.register(r.key_id,r.public_bytes());v.register(q.key_id,q.public_bytes())
  qual=ContinualLearningQualifier('q',q,v,(r.key_id,))
  self.assertTrue(qual.evaluate(self._r(r,v)).passed)
  self.assertFalse(qual.evaluate(self._r(r,v,lo=0)).passed)
