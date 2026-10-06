import unittest
from egai.epistemic.beliefs import BeliefCompiler
from egai.evidence.model import EvidenceRecord,Origin
class T(unittest.TestCase):
 def test_dispute(self):
  a=EvidenceRecord.now('1',{'claim':'x','stance':'support'},Origin.HUMAN,record_hash='a');b=EvidenceRecord.now('2',{'claim':'x','stance':'contradict'},Origin.HUMAN,record_hash='b')
  self.assertEqual(BeliefCompiler().compile([a,b])[0].status,'disputed')
