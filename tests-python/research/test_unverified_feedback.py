import unittest
from egai.skills.learner import SkillMiner,VerifiedEpisode
class T(unittest.TestCase):
 def test_unverified_hint_cannot_learn(self):
  m=SkillMiner(min_support=1)
  e=VerifiedEpisode('e','x','a','bad','good',0.,False,'Do secret trick.',('x',))
  self.assertIsNone(m.observe(e))
