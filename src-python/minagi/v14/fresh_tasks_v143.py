from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from egai.common.canonical import digest, validate_digest
from minagi.egai.canonical import sha256_json
from .fresh_tasks import HiddenTaskCommitment, FreshTaskLease

@dataclass(frozen=True)
class FreshTaskConsumptionReceiptV143:
    task_id:str
    lease_id:str
    consumer_id:str
    generation:int
    commitment_digest:str
    task_digest:str
    consumed_at:float
    authority_id:str
    authority_generation:int
    signer_key_id:str=""
    signature_b64:str=""
    schema:str="mini-agi-v14.1-alpha3-fresh-task-consumption-receipt-v1"
    def __post_init__(self): validate_digest(self.commitment_digest); validate_digest(self.task_digest)
    def unsigned(self): return replace(self,signer_key_id="",signature_b64="")
    @property
    def digest(self): return digest(self)

class FreshTaskMetadataStoreV143:
    """Public metadata store. It has no handle to hidden task contents or salts."""
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.conn=sqlite3.connect(self.path,timeout=5.0,isolation_level=None); self.conn.row_factory=sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL"); self.conn.execute("PRAGMA synchronous=FULL"); self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY,generation INTEGER NOT NULL,commitment_digest TEXT NOT NULL UNIQUE,
          created_at REAL NOT NULL,state TEXT NOT NULL CHECK(state IN ('SEALED','LEASED','CONSUMED','CLOSED')),
          lease_id TEXT UNIQUE,consumer_id TEXT,issued_at REAL,expires_at REAL,consumed_at REAL);
        """)
    def register(self,c:HiddenTaskCommitment):
        with self.conn:self.conn.execute("INSERT INTO tasks(task_id,generation,commitment_digest,created_at,state) VALUES(?,?,?,?,?)",
            (c.task_id,c.generation,c.commitment_digest,c.created_at,"SEALED"))
    def lease(self,*,task_id:str,consumer_id:str,ttl_seconds:float=300.0)->FreshTaskLease:
        now=time.time(); exp=now+float(ttl_seconds); lid="LEASE-"+uuid.uuid4().hex
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("UPDATE tasks SET state='SEALED',lease_id=NULL,consumer_id=NULL,issued_at=NULL,expires_at=NULL WHERE task_id=? AND state='LEASED' AND expires_at < ?",(task_id,now))
            r=self.conn.execute("SELECT generation,commitment_digest FROM tasks WHERE task_id=?",(task_id,)).fetchone()
            if r is None: raise KeyError(task_id)
            cur=self.conn.execute("UPDATE tasks SET state='LEASED',lease_id=?,consumer_id=?,issued_at=?,expires_at=? WHERE task_id=? AND state='SEALED'",(lid,consumer_id,now,exp,task_id))
            if cur.rowcount != 1: raise PermissionError("fresh task unavailable")
            self.conn.execute("COMMIT")
        except Exception:self.conn.execute("ROLLBACK");raise
        return FreshTaskLease(lid,task_id,int(r["generation"]),str(consumer_id),r["commitment_digest"],now,exp)
    def consume_claim(self,lease:FreshTaskLease):
        now=time.time(); self.conn.execute("BEGIN IMMEDIATE")
        try:
            r=self.conn.execute("SELECT * FROM tasks WHERE task_id=?",(lease.task_id,)).fetchone()
            if r is None: raise KeyError(lease.task_id)
            if r["state"]!="LEASED" or r["lease_id"]!=lease.lease_id or r["consumer_id"]!=lease.consumer_id: raise PermissionError("invalid fresh-task lease")
            if float(r["expires_at"]) < now: raise PermissionError("fresh-task lease expired")
            cur=self.conn.execute("UPDATE tasks SET state='CONSUMED',consumed_at=? WHERE task_id=? AND lease_id=? AND state='LEASED'",(now,lease.task_id,lease.lease_id))
            if cur.rowcount!=1: raise PermissionError("fresh-task lease lost")
            self.conn.execute("COMMIT"); return now
        except Exception:self.conn.execute("ROLLBACK");raise
    def close(self,task_id:str):
        with self.conn:
            cur=self.conn.execute("UPDATE tasks SET state='CLOSED' WHERE task_id=? AND state='CONSUMED'",(task_id,))
            if cur.rowcount!=1: raise PermissionError("only consumed task may close")

class FreshTaskVaultAuthorityV143:
    """Secret-bearing authority. Give evaluator service this object; never give it to learner/runtime."""
    def __init__(self,path,*,authority_id:str,authority_generation:int,signer):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.authority_id=authority_id; self.generation=int(authority_generation); self.signer=signer
        self.db=sqlite3.connect(self.path,timeout=5.0,isolation_level=None); self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS secrets(task_id TEXT PRIMARY KEY,task_json TEXT NOT NULL,salt TEXT NOT NULL,created_at REAL NOT NULL)")
    def seal(self,task,*,generation:int,metadata:FreshTaskMetadataStoreV143)->HiddenTaskCommitment:
        tid="FT-"+uuid.uuid4().hex; salt=secrets.token_hex(32); created=time.time(); commitment=sha256_json({"task":task,"salt":salt})
        with self.db:self.db.execute("INSERT INTO secrets(task_id,task_json,salt,created_at) VALUES(?,?,?,?)",(tid,json.dumps(task,sort_keys=True,separators=(",",":")),salt,created))
        c=HiddenTaskCommitment(tid,int(generation),commitment,created)
        try:metadata.register(c)
        except Exception:
            with self.db:self.db.execute("DELETE FROM secrets WHERE task_id=?",(tid,))
            raise
        return c
    def consume(self,lease:FreshTaskLease,*,metadata:FreshTaskMetadataStoreV143):
        secret=self.db.execute("SELECT task_json FROM secrets WHERE task_id=?",(lease.task_id,)).fetchone()
        if secret is None: raise RuntimeError("fresh-task secret absent")
        task=json.loads(secret["task_json"]); td=sha256_json(task)
        consumed=metadata.consume_claim(lease)
        rec=FreshTaskConsumptionReceiptV143(lease.task_id,lease.lease_id,lease.consumer_id,lease.generation,lease.commitment_digest,td,consumed,self.authority_id,self.generation)
        env=self.signer.sign(asdict(rec.unsigned())); rec=replace(rec,signer_key_id=env.key_id,signature_b64=env.signature_b64)
        return task,rec
