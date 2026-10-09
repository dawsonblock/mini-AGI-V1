from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
import os
import tempfile
import time
from pathlib import Path
from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class ExternalAuditAnchorV143:
    event_seq:int
    event_digest:str
    checkpoint_digest:str
    authority_generation:int
    created_at:float
    signer_key_id:str=""
    signature_b64:str=""
    schema:str="mini-agi-v14.1-alpha3-external-audit-anchor-v1"
    def __post_init__(self): validate_digest(self.event_digest); validate_digest(self.checkpoint_digest)
    def unsigned(self): return replace(self,signer_key_id="",signature_b64="")
    @property
    def digest(self): return digest(self)

class FileAuditAnchorV143:
    """Separate, mandatory signed anchor file for the governance audit DB."""
    def __init__(self,path,*,signer=None,verifier=None,trusted_key_ids=()):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.signer=signer; self.verifier=verifier; self.trusted=set(str(x) for x in trusted_key_ids)
    def write(self,*,event_seq:int,event_digest:str,checkpoint_digest:str,authority_generation:int):
        if self.signer is None: raise RuntimeError("anchor signer unavailable")
        a=ExternalAuditAnchorV143(int(event_seq),event_digest,checkpoint_digest,int(authority_generation),time.time())
        env=self.signer.sign(asdict(a.unsigned())); a=replace(a,signer_key_id=env.key_id,signature_b64=env.signature_b64)
        fd,tmp=tempfile.mkstemp(prefix="audit-anchor-",dir=str(self.path.parent)); os.close(fd)
        try:
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(asdict(a),f,sort_keys=True,separators=(",",":")); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,self.path)
            dfd=os.open(self.path.parent,os.O_RDONLY)
            try:os.fsync(dfd)
            finally:os.close(dfd)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return a
    def read_and_verify(self):
        if not self.path.exists(): raise RuntimeError("mandatory external audit anchor missing")
        body=json.loads(self.path.read_text()); a=ExternalAuditAnchorV143(**body)
        if a.signer_key_id not in self.trusted: raise PermissionError("untrusted external audit anchor signer")
        if self.verifier is None or not self.verifier.verify(asdict(a.unsigned()),SignedEnvelope(a.signer_key_id,a.signature_b64)):
            raise PermissionError("invalid external audit anchor signature")
        return a
