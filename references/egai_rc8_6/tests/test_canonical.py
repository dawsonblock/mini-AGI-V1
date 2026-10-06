import unittest,math
from egai.common.canonical import digest
class T(unittest.TestCase):
 def test_nonfinite_rejected(self):
  for x in (math.nan,math.inf,-math.inf):
   with self.assertRaises(ValueError):digest({'x':x})
