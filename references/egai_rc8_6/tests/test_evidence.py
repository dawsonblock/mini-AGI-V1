import tempfile,unittest
from egai.evidence.ledger import EvidenceLedger
from egai.evidence.model import EvidenceRecord,Origin,Verification
from helpers import trust
class T(unittest.TestCase):
 def test_chain_and_strict_eligibility(self):
  s,v=trust()
  with tempfile.TemporaryDirectory() as d:
   l=EvidenceLedger(d+'/e.db',s,v);l.append(EvidenceRecord.now('1',{'claim':'x'},Origin.DETERMINISTIC_TOOL,verification_state=Verification.VERIFIED));l.append(EvidenceRecord.now('2',{'claim':'hallucination'},Origin.MODEL_INFERENCE,verification_state=Verification.UNVERIFIED));l.append(EvidenceRecord.now('3',{'claim':'dream'},Origin.SIMULATION,verification_state=Verification.VERIFIED));l.checkpoint();self.assertTrue(l.verify_chain());self.assertEqual([x.record_id for x in l.eligible('belief')],['1'])
 def test_metadata_tamper_detected(self):
  s,v=trust()
  with tempfile.TemporaryDirectory() as d:
   l=EvidenceLedger(d+'/e.db',s,v);l.append(EvidenceRecord.now('1',{'claim':'x'},Origin.HUMAN,verification_state=Verification.VERIFIED));l.execute("UPDATE evidence SET previous_hash='evil' WHERE seq=1");self.assertFalse(l.verify_chain())
 def test_body_tamper_detected(self):
  s,v=trust()
  with tempfile.TemporaryDirectory() as d:
   l=EvidenceLedger(d+'/e.db',s,v);l.append(EvidenceRecord.now('1',{'claim':'x'},Origin.HUMAN,verification_state=Verification.VERIFIED));l.execute("UPDATE evidence SET body=replace(body,'x','y') WHERE seq=1");self.assertFalse(l.verify_chain())
