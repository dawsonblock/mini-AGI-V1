import unittest
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.bench.sequential import EpisodeTrace
from egai.replay.audit import ReplayAuditor
class T(unittest.TestCase):
 def test_grounded_prefix_causal_audit(self):
  s=Ed25519Signer.generate('r');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes())
  a=ReplayAuditor(s,v).audit_traces('a',(EpisodeTrace('c','reverse','abc','BAD','cba',0.,1.,()),))
  self.assertEqual(a.prefix_violations,0); self.assertEqual(a.grounded_coverage,1.0); self.assertEqual(a.grounded_transitions,2)
