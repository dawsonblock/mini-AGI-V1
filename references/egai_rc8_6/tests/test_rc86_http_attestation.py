import unittest
from dataclasses import replace
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.cognition.attestation import issue_attestation,verify_attestation
class T(unittest.TestCase):
 def test_signed_http_runtime_attestation(self):
  s=Ed25519Signer.generate('server');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes());md='sha256:'+'1'*64;rd='sha256:'+'2'*64
  a=issue_attestation('qwen',md,rd,'llama-server/1','server',s);self.assertTrue(verify_attestation(a,v,(s.key_id,),md,'qwen'))
  with self.assertRaises((PermissionError,ValueError)):verify_attestation(replace(a,served_model='other'),v,(s.key_id,),md,'qwen')
