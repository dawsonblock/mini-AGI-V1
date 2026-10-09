import tempfile
import unittest
from egai.authority.ledger import ImprovementLedger
from helpers import trust
class T(unittest.TestCase):
 def test_signed_chain(self):
  s,v=trust()
  with tempfile.TemporaryDirectory() as d:
   ledger=ImprovementLedger(d+'/i.db',s,v);ledger.record('i','p',{}, {},'q','d');self.assertTrue(ledger.verify_chain())
