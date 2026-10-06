import unittest
from egai.bench.engine import TaskCase
from egai.bench.leakage import detect_leakage
class T(unittest.TestCase):
 def test_case_punctuation_normalization_detected(self):
  a=[TaskCase('a','experience','Fix: FOO_bar!','x')];b=[TaskCase('b','future','fix foo_bar','y')]
  self.assertFalse(detect_leakage(a,b,.99).clean)
