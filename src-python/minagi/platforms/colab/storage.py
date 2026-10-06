from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, os, tempfile

@dataclass
class ColabStorage:
    root: Path
    def __init__(self, root: str | Path):
        self.root=Path(root)
        for d in ("cas","campaigns","datasets","adapters","evidence","epochs","ledgers","logs","exports"):
            (self.root/d).mkdir(parents=True,exist_ok=True)

    def put_bytes(self, data: bytes) -> str:
        h=hashlib.sha256(data).hexdigest(); target=self.root/"cas"/"sha256"/h[:2]/h
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=data: raise RuntimeError("CAS digest collision")
            return "sha256:"+h
        fd,tmp=tempfile.mkstemp(prefix=".tmp-",dir=target.parent)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=h: raise IOError("CAS read-back verification failed")
        return "sha256:"+h

    def get_bytes(self, d: str) -> bytes:
        alg,h=d.split(":",1)
        if alg!="sha256" or len(h)!=64: raise ValueError("invalid digest")
        p=self.root/"cas"/"sha256"/h[:2]/h
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=h: raise IOError("CAS object corruption")
        return data
