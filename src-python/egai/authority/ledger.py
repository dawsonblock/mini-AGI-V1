import json
from egai.common.store import SQLiteStore
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope

class ImprovementLedger(SQLiteStore):
    def __init__(self,path,signer,verifier):
        super().__init__(path);self.signer=signer;self.verifier=verifier
        self.execute("""CREATE TABLE IF NOT EXISTS improvements(seq INTEGER PRIMARY KEY,improvement_id TEXT UNIQUE,body TEXT,entry_hash TEXT UNIQUE,previous_hash TEXT,signer_key_id TEXT,signature_b64 TEXT)""")
    def record(self,improvement_id,proposal_digest,predicted,actual,qualification_digest,decision_digest,future=None):
        row=self.db.execute('SELECT seq,entry_hash FROM improvements ORDER BY seq DESC LIMIT 1').fetchone();seq=(row['seq']+1 if row else 1);prev=(row['entry_hash'] if row else 'GENESIS')
        body={'improvement_id':improvement_id,'proposal_digest':proposal_digest,'predicted':predicted,'actual':actual,'qualification_digest':qualification_digest,'decision_digest':decision_digest,'future':future or {},'seq':seq,'previous_hash':prev}
        h=digest(body);env=self.signer.sign({'entry_hash':h,'seq':seq,'previous':prev})
        self.execute('INSERT INTO improvements VALUES(?,?,?,?,?,?,?)',(seq,improvement_id,json.dumps(body),h,prev,env.key_id,env.signature_b64));return h
    def verify_chain(self):
        prev='GENESIS';seq=0
        for r in self.db.execute('SELECT * FROM improvements ORDER BY seq'):
            seq+=1;body=json.loads(r['body'])
            if r['seq']!=seq or r['previous_hash']!=prev or digest(body)!=r['entry_hash']:return False
            if not self.verifier.verify({'entry_hash':r['entry_hash'],'seq':seq,'previous':prev},SignedEnvelope(r['signer_key_id'],r['signature_b64'])):return False
            prev=r['entry_hash']
        return True
