import unittest
from egai.bench.engine import TaskCase
from egai.bench.leakage import detect_leakage
class T(unittest.TestCase):
 def test_near_duplicate_detected(self):
  a=[TaskCase('a','experience','fix package dependency error in project alpha','x')]
  b=[TaskCase('b','future','fix package dependency error in project beta','y')]
  self.assertFalse(detect_leakage(a,b,.7).clean)
