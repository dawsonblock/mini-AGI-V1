from __future__ import annotations
from pathlib import Path
import json, os, time, uuid

class LockTimeout(TimeoutError): pass

class ProcessFileLock:
    """Cross-process lock with owner tokens and conservative stale-lock recovery."""
    def __init__(self,path:str|Path,timeout:float=10.0,poll:float=.05,stale_after:float=300.0):
        self.path=Path(path); self.timeout=timeout; self.poll=poll; self.stale_after=stale_after
        self.fd=None; self.token=uuid.uuid4().hex

    @staticmethod
    def _pid_alive(pid:int)->bool:
        if pid <= 0: return False
        try: os.kill(pid,0); return True
        except ProcessLookupError: return False
        except PermissionError: return True
        except OSError: return True

    def _break_stale(self)->bool:
        try:
            raw=json.loads(self.path.read_text(encoding='utf-8'))
            pid=int(raw.get('pid',-1)); created=float(raw.get('created_at',0))
        except Exception:
            # Unknown/corrupt locks are only breakable after the file itself is old.
            try: created=self.path.stat().st_mtime
            except FileNotFoundError: return True
            pid=-1
        age=max(0.0,time.time()-created)
        if age < self.stale_after or self._pid_alive(pid): return False
        try: self.path.unlink(); return True
        except FileNotFoundError: return True

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True); deadline=time.monotonic()+self.timeout
        payload=json.dumps({'pid':os.getpid(),'token':self.token,'created_at':time.time()},sort_keys=True).encode()
        while True:
            try:
                self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                os.write(self.fd,payload); os.fsync(self.fd); return self
            except FileExistsError:
                if self._break_stale(): continue
                if time.monotonic()>=deadline: raise LockTimeout(str(self.path))
                time.sleep(self.poll)

    def __exit__(self,*_):
        if self.fd is not None: os.close(self.fd); self.fd=None
        try:
            raw=json.loads(self.path.read_text(encoding='utf-8'))
            if raw.get('token') == self.token: self.path.unlink()
        except FileNotFoundError: pass
        except Exception: pass
