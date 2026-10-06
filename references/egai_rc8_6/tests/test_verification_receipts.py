import unittest
from dataclasses import replace
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from egai.skills.verification import VerificationAuthority, VerificationReceiptValidator
from egai.bench.engine import TaskCase

class T(unittest.TestCase):
 def test_signed_receipt_and_tamper_rejection(self):
  s=Ed25519Signer.generate('feedback'); v=Ed25519Verifier(); v.register(s.key_id,s.public_bytes())
  c=TaskCase('e1','experience','abc','cba','reverse','Reverse characters.')
  a=VerificationAuthority('oracle',s,lambda a,b:1. if a==b else 0.)
  r=a.verify_repair(c,'UNKNOWN','cba')
  validator=VerificationReceiptValidator(v,(s.key_id,))
  self.assertTrue(validator.validate(r,c,'UNKNOWN','cba'))
  with self.assertRaises((PermissionError,ValueError)):
   validator.validate(replace(r,score=.5),c,'UNKNOWN','cba')

 def test_receipt_binds_attempt(self):
  s=Ed25519Signer.generate('feedback'); v=Ed25519Verifier(); v.register(s.key_id,s.public_bytes())
  c=TaskCase('e1','experience','abc','cba','reverse','Reverse characters.')
  r=VerificationAuthority('oracle',s,lambda a,b:1. if a==b else 0.).verify_repair(c,'BAD','cba')
  with self.assertRaises(ValueError):
   VerificationReceiptValidator(v,(s.key_id,)).validate(r,c,'DIFFERENT','cba')
