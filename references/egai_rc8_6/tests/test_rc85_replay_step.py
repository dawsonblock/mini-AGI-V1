import unittest
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.replay.world import ReplayWorld,ReplayNode
class T(unittest.TestCase):
 def test_step_reveals_recorded_next_without_exposing_future_to_policy(self):
  s=Ed25519Signer.generate('r');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes())
  root=ReplayNode('r',0,{'x':1},{},{})
  child=ReplayNode('c',1,{}, {'a':'go'}, {'reward':1}, 'r')
  w=ReplayWorld('w',[root,child],s,v); cap=w.issue_capability(0)
  self.assertEqual([n.node_id for n in w.view(cap)],['r'])
  out=w.step('r',{'a':'go'},cap); self.assertTrue(out.grounded); self.assertEqual(out.node.node_id,'c')
