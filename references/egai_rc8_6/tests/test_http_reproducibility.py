import unittest
from egai.cognition.backends import FrozenHTTPChatModel
from egai.cognition.model import FrozenModelIdentity
class T(unittest.TestCase):
 def test_research_mode_requires_zero_temperature(self):
  i=FrozenModelIdentity('x',(('w','sha256:'+'1'*64),),'','','http')
  with self.assertRaises(ValueError): FrozenHTTPChatModel(i,'http://localhost:1','x',temperature=.2)
