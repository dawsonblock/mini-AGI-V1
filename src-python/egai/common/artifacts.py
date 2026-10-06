from pathlib import Path
import os, hashlib
from .canonical import sha256_bytes, validate_digest

class ArtifactStore:
    def __init__(self,root,max_bytes=2_000_000_000):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.max_bytes=max_bytes
    def _path(self,digest_value):
        validate_digest(digest_value); h=digest_value.split(':',1)[1]; return self.root/h[:2]/h[2:]
    def put_bytes(self,b:bytes)->str:
        if len(b)>self.max_bytes: raise ValueError("artifact too large")
        d=sha256_bytes(b); p=self._path(d); p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            tmp=p.with_suffix('.tmp'); tmp.write_bytes(b); os.replace(tmp,p)
        return d
    def put_file(self,path,chunk_size=1024*1024)->str:
        src=Path(path); size=src.stat().st_size
        if size>self.max_bytes: raise ValueError("artifact too large")
        h=hashlib.sha256()
        with src.open('rb') as f:
            while True:
                b=f.read(chunk_size)
                if not b: break
                h.update(b)
        d='sha256:'+h.hexdigest(); dst=self._path(d); dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            tmp=dst.with_suffix('.tmp')
            with src.open('rb') as a,tmp.open('wb') as b:
                while True:
                    chunk=a.read(chunk_size)
                    if not chunk: break
                    b.write(chunk)
            if self._hash_file(tmp)!=d:
                tmp.unlink(missing_ok=True); raise IOError('artifact changed during copy')
            os.replace(tmp,dst)
        return d
    def _hash_file(self,p,chunk_size=1024*1024):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while True:
                b=f.read(chunk_size)
                if not b: break
                h.update(b)
        return 'sha256:'+h.hexdigest()
    def get_bytes(self,digest_value:str)->bytes:
        p=self._path(digest_value)
        if not p.is_file(): raise FileNotFoundError(digest_value)
        b=p.read_bytes()
        if sha256_bytes(b)!=digest_value: raise IOError("artifact digest mismatch")
        return b
    def exists(self,digest_value:str)->bool:
        try: self.get_bytes(digest_value); return True
        except (FileNotFoundError,ValueError): return False
