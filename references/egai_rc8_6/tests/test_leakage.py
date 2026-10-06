import unittest
from egai.bench.engine import TaskCase
from egai.bench.leakage import detect_leakage
class T(unittest.TestCase):
 def test_exact_overlap(self):
  a=[TaskCase('a','experience','same','x')];b=[TaskCase('b','future','same','y')]
  self.assertFalse(detect_leakage(a,b).clean)
