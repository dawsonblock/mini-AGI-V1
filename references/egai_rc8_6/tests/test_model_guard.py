import unittest
from egai.cognition.model import FrozenEchoModel,FrozenModelGuard
class T(unittest.TestCase):
 def test_guard(self):
  a=FrozenEchoModel('a');g=FrozenModelGuard(a);self.assertTrue(g.assert_unchanged(a))
  with self.assertRaises(RuntimeError):g.assert_unchanged(FrozenEchoModel('b'))
