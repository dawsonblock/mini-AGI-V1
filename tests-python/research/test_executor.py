import unittest
from egai.skills.executor import SkillExecutor
from egai.skills.model import SkillManifest
class T(unittest.TestCase):
 def test_verified_handler(self):
  x=SkillExecutor();x.register_handler('inc',lambda i:i['x']+1);x.register_verifier('is2',lambda i,o:o==2)
  m=SkillManifest('s','1','p',(),(),(),('x',),('y',),{'handler':'inc'},verifier={'handler':'is2'})
  self.assertTrue(x.execute(m,{'x':1}).ok)
