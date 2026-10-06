import tempfile,unittest,json
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.experiment.journal import ExperimentJournal
class T(unittest.TestCase):
 def test_chain_and_tamper(self):
  with tempfile.TemporaryDirectory() as d:
   s=Ed25519Signer.generate('j');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes());p=Path(d)/'j.jsonl';j=ExperimentJournal(p,s,v)
   ed='sha256:'+'1'*64;j.append(ed,'start',{'x':1});j.append(ed,'checkpoint',{'x':2});self.assertTrue(j.verify(ed))
   rows=p.read_text().splitlines();o=json.loads(rows[1]);o['payload']['x']=9;rows[1]=json.dumps(o);p.write_text('\n'.join(rows)+'\n');self.assertFalse(j.verify(ed))
