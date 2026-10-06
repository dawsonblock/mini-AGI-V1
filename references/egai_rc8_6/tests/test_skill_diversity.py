import unittest
from egai.skills.learner import SkillMiner, VerifiedEpisode
class T(unittest.TestCase):
 def test_distinct_input_requirement(self):
  m=SkillMiner(min_support=2,min_distinct_inputs=2)
  def e(i,input_text): return VerifiedEpisode(str(i),'reverse',input_text,input_text[::-1],input_text[::-1],1.,True,'Reverse characters.',('reverse',),'UNKNOWN',0.,'oracle','sha256:'+'1'*64)
  self.assertIsNone(m.observe(e(1,'abc')))
  self.assertIsNone(m.observe(e(2,'abc')))
  self.assertIsNotNone(m.observe(e(3,'cat')))
