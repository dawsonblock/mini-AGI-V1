import json
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class JournalEntry:
    seq:int
    experiment_digest:str
    event_type:str
    payload:dict
    previous_hash:str
    entry_hash:str=''
    signer_key_id:str=''
    signature_b64:str=''
    def unsigned_for_hash(self): return replace(self,entry_hash='',signer_key_id='',signature_b64='')
    def unsigned_for_sig(self): return replace(self,signer_key_id='',signature_b64='')

class ExperimentJournal:
    """Append-only JSONL journal with hash chaining and signatures."""
    def __init__(self,path,signer,verifier):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.signer=signer; self.verifier=verifier
    def entries(self):
        if not self.path.exists(): return []
        return [JournalEntry(**json.loads(x)) for x in self.path.read_text(encoding='utf-8').splitlines() if x.strip()]
    def append(self,experiment_digest,event_type,payload):
        xs=self.entries(); seq=len(xs)+1; prev=xs[-1].entry_hash if xs else 'GENESIS'
        base=JournalEntry(seq,experiment_digest,event_type,dict(payload),prev)
        h=digest(asdict(base.unsigned_for_hash()))
        hashed=replace(base,entry_hash=h)
        env=self.signer.sign(asdict(hashed.unsigned_for_sig()))
        signed=replace(hashed,signer_key_id=env.key_id,signature_b64=env.signature_b64)
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(asdict(signed),sort_keys=True,separators=(',',':'))+'\n')
        return signed
    def verify(self,expected_experiment_digest=None):
        prev='GENESIS'; seq=1
        for e in self.entries():
            if e.seq!=seq or e.previous_hash!=prev: return False
            if expected_experiment_digest and e.experiment_digest!=expected_experiment_digest:return False
            if digest(asdict(e.unsigned_for_hash()))!=e.entry_hash:return False
            if not self.verifier.verify(asdict(e.unsigned_for_sig()),SignedEnvelope(e.signer_key_id,e.signature_b64)):return False
            prev=e.entry_hash; seq+=1
        return True
