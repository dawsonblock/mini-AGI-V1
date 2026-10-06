import unittest
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.trust import AuthorityTrust
class T(unittest.TestCase):
 def test_role_confusion_rejected(self):
  runner=Ed25519Signer.generate('runner');qual=Ed25519Signer.generate('qual')
  trust=AuthorityTrust({'runner':(runner.key_id,),'qualifier':(qual.key_id,)})
  trust.require('runner',runner.key_id)
  with self.assertRaises(PermissionError):trust.require('qualifier',runner.key_id)
