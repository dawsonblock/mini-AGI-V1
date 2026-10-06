import tempfile,unittest
from egai.authority.ledger import ImprovementLedger
from helpers import trust
class T(unittest.TestCase):
 def test_signed_chain(self):
  s,v=trust()
  with tempfile.TemporaryDirectory() as d:
   l=ImprovementLedger(d+'/i.db',s,v);l.record('i','p',{}, {},'q','d');self.assertTrue(l.verify_chain())
