import unittest
from egai.learning.proposals import PlasticityProposer, LEVEL, RING_CAP
class T(unittest.TestCase):
 def test_ring_limits(self):
  p=PlasticityProposer().propose("x","root","R1",repeated_gap=True,skill_possible=False)
  self.assertLessEqual(LEVEL[p.mechanism],RING_CAP["R1"])
 def test_no_promotion_surface(self):
  p=PlasticityProposer()
  self.assertFalse(hasattr(p,"promote"))
  self.assertFalse(hasattr(p,"deploy"))
