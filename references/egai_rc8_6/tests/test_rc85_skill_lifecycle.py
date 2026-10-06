import unittest
from egai.skills.memory import SandboxSkillMemory,LearnedProcedure
class T(unittest.TestCase):
 def test_bad_skill_degrades_then_retires_and_is_not_retrieved(self):
  m=SandboxSkillMemory();p=LearnedProcedure('p','x','alpha','do alpha',('e1','e2'),2)
  m.upsert(p); self.assertEqual(m.state('p'),'active')
  m.mark_failure('p');m.mark_failure('p');self.assertEqual(m.state('p'),'degraded')
  m.mark_failure('p');m.mark_failure('p');self.assertEqual(m.state('p'),'retired')
  self.assertEqual(m.retrieve('alpha','x'),[])
  self.assertEqual(m.health()['retired'],1)
