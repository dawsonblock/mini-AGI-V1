import tempfile,unittest
from egai.skills.registry import SkillRegistry
class T(unittest.TestCase):
 def test_no_direct_promotion_surface(self):
  with tempfile.TemporaryDirectory() as d:
   r=SkillRegistry(d+'/s.db');self.assertFalse(hasattr(r,'promote'));self.assertFalse(hasattr(r,'deploy'))
