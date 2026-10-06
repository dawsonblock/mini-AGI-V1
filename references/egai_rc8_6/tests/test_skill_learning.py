import unittest
from egai.skills.learner import SkillMiner,VerifiedEpisode
from egai.skills.memory import SandboxSkillMemory
class T(unittest.TestCase):
 def test_requires_repeated_verified_support(self):
  m=SkillMiner(min_support=2);e=lambda i:VerifiedEpisode(str(i),'reverse','abc','cba','cba',1.,True,'Reverse the input characters.','reverse')
  self.assertIsNone(m.observe(e(1)));p=m.observe(e(2));self.assertIsNotNone(p)
  mem=SandboxSkillMemory();mem.upsert(p);self.assertEqual(mem.retrieve('reverse xyz','reverse')[0].procedure_id,p.procedure_id)
