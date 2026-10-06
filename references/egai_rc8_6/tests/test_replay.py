import unittest
from egai.replay.world import ReplayWorld,ReplayNode
from helpers import trust
class T(unittest.TestCase):
 def test_prefix_isolation_and_signature(self):
  s,v=trust();w=ReplayWorld('w',[ReplayNode('a',0,{'x':0},{},{'r':0}),ReplayNode('b',1,{'x':1},{'op':'x'},{'r':1},'a')],s,v)
  cap=w.issue_capability(0);self.assertEqual(len(w.view(cap)),1);self.assertFalse(w.grounded_transition('a',{'op':'x'},cap).grounded)
