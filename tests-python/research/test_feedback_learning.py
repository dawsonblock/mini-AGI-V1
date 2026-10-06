import unittest
from egai.skills.learner import SkillMiner,VerifiedEpisode
class T(unittest.TestCase):
 def test_failed_attempt_can_learn_from_verified_repair(self):
  m=SkillMiner(min_support=2)
  a=VerifiedEpisode('1','reverse','abc','cba','cba',1.,True,'Reverse the input characters.',('reverse',),'UNKNOWN',0.,'oracle')
  b=VerifiedEpisode('2','reverse','cat','tac','tac',1.,True,'Reverse the input characters.',('reverse',),'UNKNOWN',0.,'oracle')
  self.assertIsNone(m.observe(a));self.assertIsNotNone(m.observe(b))
