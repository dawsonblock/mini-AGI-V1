from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import re
import tempfile
import time
import uuid

_DIGEST_RE=re.compile(r"^[0-9a-f]{64}$")

@dataclass(frozen=True, slots=True)
class CASObject:
    digest: str
    size_bytes: int
    path: Path
    created: bool

class ContentAddressedStore:
    """Crash-safe SHA-256 content-addressed store with corruption quarantine."""
    def __init__(self, root: str|Path):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.quarantine_root=self.root/'_quarantine'; self.quarantine_root.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def digest_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

    def path_for(self,digest:str)->Path:
        if not _DIGEST_RE.fullmatch(digest): raise ValueError("digest must be lowercase sha256 hex")
        return self.root/digest[:2]/digest[2:]

    def put(self,data:bytes)->CASObject:
        digest=self.digest_bytes(data); path=self.path_for(digest); path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise RuntimeError("existing CAS object digest mismatch")
            return CASObject(digest,len(data),path,False)
        fd,tmp=tempfile.mkstemp(prefix=".cas-",dir=path.parent)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            try: os.link(tmp,path)
            except FileExistsError: pass
            if not path.exists(): os.replace(tmp,path)
            try:
                dfd=os.open(path.parent,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
            except OSError: pass
        finally:
            try: os.unlink(tmp)
            except FileNotFoundError: pass
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise RuntimeError("CAS write verification failed")
        return CASObject(digest,len(data),path,True)

    def get(self,digest:str)->bytes:
        p=self.path_for(digest); data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest: raise RuntimeError("CAS object digest mismatch")
        return data

    def verify(self,digest:str)->bool:
        try: self.get(digest); return True
        except (OSError,RuntimeError,ValueError): return False

    def quarantine(self,digest:str)->Path|None:
        p=self.path_for(digest)
        if not p.exists(): return None
        q=self.quarantine_root/f"{digest}.{int(time.time())}.{uuid.uuid4().hex}.corrupt"
        os.replace(p,q)
        try:
            dfd=os.open(self.quarantine_root,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
        except OSError: pass
        return q

    def verify_or_quarantine(self,digest:str)->tuple[bool,Path|None]:
        p=self.path_for(digest)
        if not p.exists(): return False,None
        if self.verify(digest): return True,None
        return False,self.quarantine(digest)

    def iter_digests(self):
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir() or shard.name.startswith('_') or len(shard.name)!=2: continue
            for p in sorted(shard.iterdir()):
                if p.is_file():
                    d=shard.name+p.name
                    if _DIGEST_RE.fullmatch(d): yield d

    def delete(self,digest:str)->bool:
        p=self.path_for(digest)
        try:
            p.unlink(); return True
        except FileNotFoundError: return False
