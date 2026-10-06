import tempfile,unittest
from egai.runtime import Runtime
from egai.evidence.model import EvidenceRecord,Origin,Verification
class T(unittest.TestCase):
 def test_cross_process_style_reload(self):
  with tempfile.TemporaryDirectory() as d:
   r=Runtime(d);r.evidence.append(EvidenceRecord.now('e',{'claim':'x'},Origin.DETERMINISTIC_TOOL,verification_state=Verification.VERIFIED));r.evidence.checkpoint();r.evidence.close();r.improvements.close();r.skills.close()
   r2=Runtime(d);self.assertTrue(r2.evidence.verify_chain())
