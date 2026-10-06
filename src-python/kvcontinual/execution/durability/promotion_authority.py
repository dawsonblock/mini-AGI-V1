from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, hmac, json, os, secrets, sqlite3, tempfile, time
from .file_lock import ProcessFileLock


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if mode is not None:
            try: os.chmod(tmp, mode)
            except OSError: pass
        os.replace(tmp, path)
        try:
            dfd=os.open(path.parent,os.O_RDONLY); os.fsync(dfd); os.close(dfd)
        except OSError: pass
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


@dataclass(frozen=True, slots=True)
class PromotionAuthorization:
    candidate_id: str
    qualification_digest: str
    old_cache_identity: str
    new_cache_identity: str
    key_id: str
    key_epoch: int
    issued_at: float
    expires_at: float
    nonce: str
    signature: str


@dataclass(frozen=True, slots=True)
class RollbackAuthorization:
    current_adapter_id: str
    target_adapter_id: str
    old_cache_identity: str
    new_cache_identity: str
    reason_digest: str
    key_id: str
    key_epoch: int
    issued_at: float
    expires_at: float
    nonce: str
    signature: str


class LocalPromotionAuthority:
    """Rotatable local HMAC-SHA256 control authority with durable replay prevention.

    Keys are retained by key id so pre-rotation approvals remain verifiable until
    expiry. Authorization nonces are consumed exactly once in a SQLite store under
    a cross-process lock. The release-signing RSA key is deliberately separate.
    """
    def __init__(self, key_path: str | Path):
        self.key_path = Path(key_path)
        self.root = self.key_path.parent
        self.keys_dir = self.root / "keys"
        self.active_path = self.root / "active.json"
        self.epochs_path = self.root / "key_epochs.json"
        self.revocations_path = self.root / "revoked_keys.json"
        self.usage_db = self.root / "authorization_usage.sqlite3"
        self.usage_lock = self.root / "authorization_usage.lock"
        self.root.mkdir(parents=True, exist_ok=True); self.keys_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap(); self._init_usage_store()

    @staticmethod
    def _key_id(key: bytes) -> str:
        return hashlib.sha256(key).hexdigest()[:16]

    def _key_file(self, key_id: str) -> Path:
        if len(key_id) != 16 or any(c not in "0123456789abcdef" for c in key_id):
            raise ValueError("invalid key id")
        return self.keys_dir / f"{key_id}.key"

    def _load_epochs(self) -> dict[str,int]:
        if self.epochs_path.exists():
            try:
                raw=json.loads(self.epochs_path.read_text())
                return {str(k):int(v) for k,v in raw.items()}
            except Exception as e:
                raise RuntimeError("invalid control authority epoch map") from e
        ids=[]
        for p in sorted(self.keys_dir.glob("*.key"), key=lambda x:(x.stat().st_mtime_ns,x.name)):
            if len(p.stem)==16: ids.append(p.stem)
        epochs={kid:i+1 for i,kid in enumerate(ids)}
        if epochs: _atomic_write(self.epochs_path,(json.dumps(epochs,indent=2,sort_keys=True)+"\n").encode())
        return epochs

    def _write_epochs(self, epochs: dict[str,int]) -> None:
        _atomic_write(self.epochs_path,(json.dumps(epochs,indent=2,sort_keys=True)+"\n").encode())

    def key_epoch(self, key_id: str) -> int:
        epochs=self._load_epochs()
        if key_id not in epochs: raise RuntimeError("unknown control key epoch")
        return int(epochs[key_id])

    def _activate(self, key: bytes, *, created_at: float | None = None) -> str:
        if len(key) < 32: raise RuntimeError("promotion authority key must be at least 32 bytes")
        key_id=self._key_id(key); kf=self._key_file(key_id)
        if not kf.exists(): _atomic_write(kf,key,0o600)
        epochs=self._load_epochs()
        if key_id not in epochs:
            epochs[key_id]=(max(epochs.values()) if epochs else 0)+1; self._write_epochs(epochs)
        epoch=int(epochs[key_id])
        _atomic_write(self.key_path,key,0o600)
        meta={"format":"rc10-control-keyring-v4","key_purpose":"adapter-control-authority","active_key_id":key_id,
              "active_epoch":epoch,"activated_at":time.time() if created_at is None else float(created_at)}
        _atomic_write(self.active_path,(json.dumps(meta,indent=2,sort_keys=True)+"\n").encode())
        self._active_key_id=key_id; self._active_key=key; self._active_epoch=epoch
        return key_id

    def _bootstrap(self) -> None:
        if self.active_path.exists():
            try:
                meta=json.loads(self.active_path.read_text()); purpose=meta.get("key_purpose")
                if purpose not in (None,"promotion-and-rollback-control","adapter-control-authority"):
                    raise RuntimeError("control authority key purpose mismatch")
                key_id=str(meta["active_key_id"]); key=self._key_file(key_id).read_bytes()
                if self._key_id(key) != key_id: raise RuntimeError("control key id mismatch")
                epochs=self._load_epochs()
                if key_id not in epochs:
                    epochs[key_id]=(max(epochs.values()) if epochs else 0)+1; self._write_epochs(epochs)
                epoch=int(epochs[key_id])
                self._active_key_id=key_id; self._active_key=key; self._active_epoch=epoch; _atomic_write(self.key_path,key,0o600)
                if meta.get('format') != 'rc10-control-keyring-v4' or purpose != 'adapter-control-authority' or int(meta.get('active_epoch',epoch)) != epoch:
                    upgraded={"format":"rc10-control-keyring-v4","key_purpose":"adapter-control-authority",
                              "active_key_id":key_id,"active_epoch":epoch,"activated_at":float(meta.get("activated_at",time.time()))}
                    _atomic_write(self.active_path,(json.dumps(upgraded,indent=2,sort_keys=True)+"\n").encode())
                return
            except Exception as e:
                raise RuntimeError("invalid control authority keyring") from e
        key=self.key_path.read_bytes() if self.key_path.exists() else secrets.token_bytes(32)
        self._activate(key)

    def _init_usage_store(self)->None:
        with sqlite3.connect(self.usage_db) as c:
            c.execute('PRAGMA journal_mode=WAL'); c.execute('PRAGMA synchronous=FULL')
            c.execute('''CREATE TABLE IF NOT EXISTS used_authorizations(
                nonce TEXT PRIMARY KEY, kind TEXT NOT NULL, key_id TEXT NOT NULL,
                binding_digest TEXT NOT NULL, used_at REAL NOT NULL)''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_used_authorizations_kind ON used_authorizations(kind,used_at)')
            c.commit()

    @property
    def active_key_id(self) -> str: return self._active_key_id

    @property
    def active_epoch(self) -> int: return int(self._active_epoch)

    def list_key_ids(self) -> tuple[str, ...]:
        ids=[]
        for p in sorted(self.keys_dir.glob("*.key")):
            if len(p.stem)==16:
                try:
                    key=p.read_bytes()
                    if self._key_id(key)==p.stem: ids.append(p.stem)
                except OSError: pass
        return tuple(ids)

    def _load_revocations(self) -> dict[str,dict]:
        if not self.revocations_path.exists(): return {}
        try:
            raw=json.loads(self.revocations_path.read_text())
            return {str(k):dict(v) for k,v in raw.items()}
        except Exception as e:
            raise RuntimeError('invalid control key revocation map') from e

    def is_key_revoked(self,key_id:str)->bool:
        return key_id in self._load_revocations()

    def revoke_key(self,key_id:str,*,reason:str)->dict:
        if key_id==self.active_key_id: raise RuntimeError('rotate active control key before revoking it')
        self._get_key(key_id,allow_revoked=True)
        rev=self._load_revocations()
        entry={'revoked_at':time.time(),'reason_digest':hashlib.sha256(reason.encode()).hexdigest()}
        rev[key_id]=entry
        _atomic_write(self.revocations_path,(json.dumps(rev,indent=2,sort_keys=True)+'\n').encode())
        return {'key_id':key_id,**entry}

    def rotate(self) -> str: return self._activate(secrets.token_bytes(32))

    def verify_keyring(self) -> tuple[bool, str]:
        try:
            meta=json.loads(self.active_path.read_text()); purpose=meta.get("key_purpose")
            if purpose not in ("promotion-and-rollback-control","adapter-control-authority"): return False,"key_purpose_mismatch"
            key_id=str(meta["active_key_id"]); key=self._key_file(key_id).read_bytes(); epoch=self.key_epoch(key_id)
            if int(meta.get('active_epoch',epoch)) != epoch: return False,'active_epoch_mismatch'
            if self._key_id(key)!=key_id: return False,"active_key_digest_mismatch"
            if self.is_key_revoked(key_id): return False,"active_key_revoked"
            if self.key_path.read_bytes()!=key: return False,"legacy_mirror_mismatch"
            for kid in self.list_key_ids():
                if self._key_id(self._key_file(kid).read_bytes())!=kid: return False,f"key_digest_mismatch:{kid}"
            return True,f"active={key_id};epoch={epoch};keys={len(self.list_key_ids())};revoked={len(self._load_revocations())};purpose={purpose}"
        except Exception as e: return False,f"keyring_error:{type(e).__name__}"

    def verify_usage_store(self)->tuple[bool,str]:
        try:
            with sqlite3.connect(self.usage_db) as c:
                result=str(c.execute('PRAGMA integrity_check').fetchone()[0]); count=int(c.execute('SELECT COUNT(*) FROM used_authorizations').fetchone()[0])
            return result=='ok',f'integrity={result};consumed={count}'
        except sqlite3.Error as e: return False,f'usage_store_error:{type(e).__name__}'

    def authorization_usage(self)->dict[str,int]:
        with sqlite3.connect(self.usage_db) as c:
            rows=c.execute('SELECT kind,COUNT(*) FROM used_authorizations GROUP BY kind').fetchall()
        out={'promotion':0,'rollback':0}
        for kind,count in rows: out[str(kind)]=int(count)
        return out

    def _get_key(self, key_id: str, *, allow_revoked: bool=False) -> bytes:
        key=self._key_file(key_id).read_bytes()
        if self._key_id(key)!=key_id: raise RuntimeError("control key digest mismatch")
        if not allow_revoked and self.is_key_revoked(key_id): raise RuntimeError('control key revoked')
        return key

    @staticmethod
    def qualification_digest(qualification) -> str:
        rows=[]
        for r in getattr(qualification, "results", []):
            rows.append({"scenario":getattr(r,"scenario",None),"passed":bool(getattr(r,"passed",False)),
                         "critical":bool(getattr(r,"critical",False)),"score_before":getattr(r,"score_before",None),
                         "score_after":getattr(r,"score_after",None)})
        body={"passed":bool(getattr(qualification,"passed",False)),"results":rows}
        return hashlib.sha256(_canonical(body)).hexdigest()

    def _sign_body(self, body: dict, key_id: str) -> str:
        return hmac.new(self._get_key(key_id), _canonical(body), hashlib.sha256).hexdigest()

    def issue(self, *, candidate_id: str, qualification, old_cache_identity: str,
              new_cache_identity: str, ttl_seconds: float = 300.0, now: float | None = None) -> PromotionAuthorization:
        if ttl_seconds <= 0: raise ValueError("ttl_seconds must be positive")
        now=time.time() if now is None else float(now); key_id=self.active_key_id
        body={"candidate_id":candidate_id,"qualification_digest":self.qualification_digest(qualification),
              "old_cache_identity":old_cache_identity,"new_cache_identity":new_cache_identity,"key_id":key_id,"key_epoch":self.key_epoch(key_id),
              "issued_at":now,"expires_at":now+ttl_seconds,"nonce":secrets.token_hex(16)}
        return PromotionAuthorization(**body,signature=self._sign_body(body,key_id))

    def verify(self, authorization: PromotionAuthorization, *, candidate_id: str, qualification,
               old_cache_identity: str, new_cache_identity: str, now: float | None = None) -> bool:
        now=time.time() if now is None else float(now)
        if now > authorization.expires_at or authorization.issued_at > now + 5.0: return False
        if authorization.candidate_id != candidate_id: return False
        try:
            if int(authorization.key_epoch) != self.key_epoch(authorization.key_id): return False
        except Exception: return False
        if authorization.qualification_digest != self.qualification_digest(qualification): return False
        if authorization.old_cache_identity != old_cache_identity or authorization.new_cache_identity != new_cache_identity: return False
        body=asdict(authorization); sig=body.pop("signature")
        try: want=self._sign_body(body,authorization.key_id)
        except (OSError,RuntimeError,ValueError): return False
        return hmac.compare_digest(sig,want)

    @staticmethod
    def reason_digest(reason: str) -> str:
        if not isinstance(reason,str) or not reason.strip(): raise ValueError("rollback reason must be non-empty")
        return hashlib.sha256(reason.strip().encode("utf-8")).hexdigest()

    def issue_rollback(self, *, current_adapter_id: str, target_adapter_id: str, old_cache_identity: str,
                       new_cache_identity: str, reason: str, ttl_seconds: float = 300.0,
                       now: float | None = None) -> RollbackAuthorization:
        if ttl_seconds <= 0: raise ValueError("ttl_seconds must be positive")
        now=time.time() if now is None else float(now); key_id=self.active_key_id
        body={"current_adapter_id":current_adapter_id,"target_adapter_id":target_adapter_id,
              "old_cache_identity":old_cache_identity,"new_cache_identity":new_cache_identity,
              "reason_digest":self.reason_digest(reason),"key_id":key_id,"key_epoch":self.key_epoch(key_id),"issued_at":now,
              "expires_at":now+ttl_seconds,"nonce":secrets.token_hex(16)}
        return RollbackAuthorization(**body,signature=self._sign_body(body,key_id))

    def verify_rollback(self, authorization: RollbackAuthorization, *, current_adapter_id: str,
                        target_adapter_id: str, old_cache_identity: str, new_cache_identity: str,
                        reason: str, now: float | None = None) -> bool:
        now=time.time() if now is None else float(now)
        if now > authorization.expires_at or authorization.issued_at > now + 5.0: return False
        if authorization.current_adapter_id != current_adapter_id or authorization.target_adapter_id != target_adapter_id: return False
        try:
            if int(authorization.key_epoch) != self.key_epoch(authorization.key_id): return False
        except Exception: return False
        if authorization.old_cache_identity != old_cache_identity or authorization.new_cache_identity != new_cache_identity: return False
        try: rd=self.reason_digest(reason)
        except ValueError: return False
        if authorization.reason_digest != rd: return False
        body=asdict(authorization); sig=body.pop("signature")
        try: want=self._sign_body(body,authorization.key_id)
        except (OSError,RuntimeError,ValueError): return False
        return hmac.compare_digest(sig,want)

    def _consume(self, *, kind:str, nonce:str, key_id:str, binding:dict)->bool:
        binding_digest=hashlib.sha256(_canonical(binding)).hexdigest()
        with ProcessFileLock(self.usage_lock,timeout=10.0):
            with sqlite3.connect(self.usage_db) as c:
                c.execute('BEGIN IMMEDIATE')
                if c.execute('SELECT 1 FROM used_authorizations WHERE nonce=?',(nonce,)).fetchone() is not None:
                    c.rollback(); return False
                c.execute('INSERT INTO used_authorizations(nonce,kind,key_id,binding_digest,used_at) VALUES(?,?,?,?,?)',
                          (nonce,kind,key_id,binding_digest,time.time()))
                c.commit(); return True

    def consume(self, authorization: PromotionAuthorization, *, candidate_id: str, qualification,
                old_cache_identity: str, new_cache_identity: str, now:float|None=None)->bool:
        if not self.verify(authorization,candidate_id=candidate_id,qualification=qualification,
                           old_cache_identity=old_cache_identity,new_cache_identity=new_cache_identity,now=now): return False
        return self._consume(kind='promotion',nonce=authorization.nonce,key_id=authorization.key_id,binding=asdict(authorization))

    def consume_rollback(self, authorization: RollbackAuthorization, *, current_adapter_id:str,target_adapter_id:str,
                         old_cache_identity:str,new_cache_identity:str,reason:str,now:float|None=None)->bool:
        if not self.verify_rollback(authorization,current_adapter_id=current_adapter_id,target_adapter_id=target_adapter_id,
                                    old_cache_identity=old_cache_identity,new_cache_identity=new_cache_identity,reason=reason,now=now): return False
        return self._consume(kind='rollback',nonce=authorization.nonce,key_id=authorization.key_id,binding=asdict(authorization))

    def sign_payload(self, payload: dict, *, purpose: str) -> tuple[str,str]:
        key_id=self.active_key_id; body={"purpose":purpose,"payload":payload}
        return key_id,self._sign_body(body,key_id)

    def verify_payload(self, payload: dict, *, purpose: str, key_id: str, signature: str) -> bool:
        try: want=self._sign_body({"purpose":purpose,"payload":payload},key_id)
        except (OSError,RuntimeError,ValueError): return False
        return hmac.compare_digest(signature,want)
