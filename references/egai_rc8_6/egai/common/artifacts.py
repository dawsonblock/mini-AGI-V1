from pathlib import Path
import os,hashlib,tempfile
from .canonical import sha256_bytes,validate_digest
class ArtifactStore:
    def __init__(self,root,max_bytes=2_000_000_000):self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.max_bytes=max_bytes
    def _path(self,digest_value):validate_digest(digest_value);h=digest_value.split(':',1)[1];return self.root/h[:2]/h[2:]
    def _atomic(self,dst,writer):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():return
        fd,name=tempfile.mkstemp(prefix=dst.name+'.',suffix='.tmp',dir=dst.parent);os.close(fd);tmp=Path(name)
        try:
            writer(tmp)
            try:os.link(tmp,dst)
            except FileExistsError:pass
            except OSError:
                if not dst.exists():os.replace(tmp,dst);return
        finally:tmp.unlink(missing_ok=True)
    def put_bytes(self,b):
        if len(b)>self.max_bytes:raise ValueError('artifact too large')
        d=sha256_bytes(b);p=self._path(d);self._atomic(p,lambda tmp:tmp.write_bytes(b));return d
    def put_file(self,path,chunk_size=1024*1024):
        src=Path(path);size=src.stat().st_size
        if size>self.max_bytes:raise ValueError('artifact too large')
        d=self._hash_file(src,chunk_size);dst=self._path(d)
        def writer(tmp):
            with src.open('rb') as a,tmp.open('wb') as b:
                while True:
                    c=a.read(chunk_size)
                    if not c:break
                    b.write(c)
            if self._hash_file(tmp,chunk_size)!=d:raise IOError('artifact changed during copy')
        self._atomic(dst,writer);return d
    def _hash_file(self,p,chunk_size=1024*1024):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            while True:
                b=f.read(chunk_size)
                if not b:break
                h.update(b)
        return 'sha256:'+h.hexdigest()
    def get_bytes(self,digest_value):
        p=self._path(digest_value)
        if not p.is_file():raise FileNotFoundError(digest_value)
        b=p.read_bytes()
        if sha256_bytes(b)!=digest_value:raise IOError('artifact digest mismatch')
        return b
    def exists(self,digest_value):
        try:self.get_bytes(digest_value);return True
        except (FileNotFoundError,ValueError):return False
