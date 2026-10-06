import tempfile,unittest
from egai.runtime import Runtime
from egai.evidence.model import EvidenceRecord,Origin,Verification
class T(unittest.TestCase):
 def test_runtime(self):
  with tempfile.TemporaryDirectory() as d:
   r=Runtime(d);r.evidence.append(EvidenceRecord.now('e',{'claim':'x'},Origin.DETERMINISTIC_TOOL,verification_state=Verification.VERIFIED));s=r.status();self.assertTrue(s['evidence_chain_valid']);self.assertEqual(s['beliefs'],1)
