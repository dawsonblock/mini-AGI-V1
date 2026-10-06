import json
from dataclasses import asdict,replace
from .model import EvidenceRecord,Origin,Verification,EvidencePolicy
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope
from egai.common.store import SQLiteStore

class EvidenceLedger(SQLiteStore):
    def __init__(self,path,signer,verifier,policy=None):
        super().__init__(path); self.signer=signer; self.verifier=verifier; self.policy=policy or EvidencePolicy()
        self.execute("""CREATE TABLE IF NOT EXISTS evidence(seq INTEGER PRIMARY KEY,record_id TEXT UNIQUE NOT NULL,
          body TEXT NOT NULL,record_hash TEXT UNIQUE NOT NULL,previous_hash TEXT NOT NULL,signer_key_id TEXT NOT NULL,signature_b64 TEXT NOT NULL)""")
        self.execute("""CREATE TABLE IF NOT EXISTS checkpoints(seq INTEGER PRIMARY KEY,head_hash TEXT NOT NULL,signer_key_id TEXT NOT NULL,signature_b64 TEXT NOT NULL)""")
    def _head(self):
        r=self.db.execute("SELECT seq,record_hash FROM evidence ORDER BY seq DESC LIMIT 1").fetchone();return (r['seq'],r['record_hash']) if r else (0,'GENESIS')
    def append(self,record):
        if record.record_hash or record.signature_b64 or record.sequence: raise ValueError('caller may not pre-authorize evidence')
        with self.immediate():
            seq,prev=self._head();seq+=1
            unsigned=replace(record,sequence=seq,previous_record_hash=prev,record_hash='',signer_key_id='',signature_b64='')
            h=digest(asdict(unsigned));env=self.signer.sign({'record_hash':h,'sequence':seq,'previous':prev})
            signed=replace(unsigned,record_hash=h,signer_key_id=env.key_id,signature_b64=env.signature_b64)
            self.db.execute('INSERT INTO evidence VALUES(?,?,?,?,?,?,?)',(seq,signed.record_id,json.dumps(asdict(signed),default=lambda x:x.value),h,prev,env.key_id,env.signature_b64))
        return signed
    def checkpoint(self):
        seq,head=self._head();env=self.signer.sign({'checkpoint_seq':seq,'head_hash':head})
        self.execute('INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?)',(seq,head,env.key_id,env.signature_b64));return seq,head
    def all(self):
        out=[]
        for row in self.db.execute('SELECT body FROM evidence ORDER BY seq'):
            d=json.loads(row['body']);d['origin']=Origin(d['origin']);d['verification_state']=Verification(d['verification_state']);d['parent_evidence']=tuple(d.get('parent_evidence',[]));out.append(EvidenceRecord(**d))
        return out
    def verify_chain(self):
        prev='GENESIS';seq=0
        for row in self.db.execute('SELECT * FROM evidence ORDER BY seq'):
            seq+=1;d=json.loads(row['body']);d['origin']=Origin(d['origin']);d['verification_state']=Verification(d['verification_state']);d['parent_evidence']=tuple(d.get('parent_evidence',[]));r=EvidenceRecord(**d)
            if r.sequence!=seq or row['seq']!=seq or r.previous_record_hash!=prev or row['previous_hash']!=prev:return False
            if row['record_hash']!=r.record_hash or row['signer_key_id']!=r.signer_key_id or row['signature_b64']!=r.signature_b64:return False
            unsigned=replace(r,record_hash='',signer_key_id='',signature_b64='')
            if digest(asdict(unsigned))!=r.record_hash:return False
            if not self.verifier.verify({'record_hash':r.record_hash,'sequence':r.sequence,'previous':prev},SignedEnvelope(r.signer_key_id,r.signature_b64)):return False
            prev=r.record_hash
        cp=self.db.execute('SELECT * FROM checkpoints ORDER BY seq DESC LIMIT 1').fetchone()
        if cp:
            if cp['seq']>seq:return False
            h=self.db.execute('SELECT record_hash FROM evidence WHERE seq=?',(cp['seq'],)).fetchone();expected=(h['record_hash'] if h else 'GENESIS')
            if cp['head_hash']!=expected:return False
            if not self.verifier.verify({'checkpoint_seq':cp['seq'],'head_hash':cp['head_hash']},SignedEnvelope(cp['signer_key_id'],cp['signature_b64'])):return False
        return True
    def eligible(self,use='belief'):
        attr=use+'_eligible';return [r for r in self.all() if getattr(self.policy.classify(r),attr)]
