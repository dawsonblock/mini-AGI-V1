from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from minagi.egai.canonical import sha256_json


@dataclass(frozen=True)
class HiddenTaskCommitment:
    task_id: str
    generation: int
    commitment_digest: str
    created_at: float
    @property
    def digest(self) -> str: return sha256_json(asdict(self))


@dataclass(frozen=True)
class FreshTaskLease:
    lease_id: str
    task_id: str
    generation: int
    consumer_id: str
    commitment_digest: str
    issued_at: float
    expires_at: float
    @property
    def digest(self) -> str: return sha256_json(asdict(self))


class DurableFreshTaskAuthority:
    """Restart-safe one-shot hidden task authority.

    Alpha2 separates public lease metadata from the sealed task vault. A process
    that is only granted access to the metadata DB can see commitments and lease
    state but cannot read task bodies or salts. Lease acquisition uses an atomic
    conditional update and expired leases can be reclaimed explicitly.
    """
    def __init__(self, path: str | Path, *, vault_path: str | Path | None = None):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.vault_path=Path(vault_path) if vault_path is not None else self.path.with_name(self.path.stem+".vault.sqlite3")
        self.conn=sqlite3.connect(self.path,timeout=5.0,isolation_level=None); self.conn.row_factory=sqlite3.Row
        self.vault=sqlite3.connect(self.vault_path,timeout=5.0,isolation_level=None); self.vault.row_factory=sqlite3.Row
        for c in (self.conn,self.vault):
            c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); c.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(
          task_id TEXT PRIMARY KEY, generation INTEGER NOT NULL, commitment_digest TEXT NOT NULL UNIQUE,
          created_at REAL NOT NULL, state TEXT NOT NULL CHECK(state IN ('SEALED','LEASED','CONSUMED','CLOSED')),
          lease_id TEXT UNIQUE, consumer_id TEXT, issued_at REAL, expires_at REAL, consumed_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_generation_state ON tasks(generation,state);
        """)
        self.vault.executescript("""
        CREATE TABLE IF NOT EXISTS task_secrets(
          task_id TEXT PRIMARY KEY, task_json TEXT NOT NULL, salt TEXT NOT NULL, created_at REAL NOT NULL
        );
        """)

    @staticmethod
    def commitment(task, salt: str) -> str: return sha256_json({"task":task,"salt":str(salt)})

    def seal(self, task, *, generation: int) -> HiddenTaskCommitment:
        task_id="FT-"+uuid.uuid4().hex; salt=secrets.token_hex(32); created=time.time(); commitment=self.commitment(task,salt)
        # Vault first; metadata second. If metadata fails, remove orphaned secret.
        self.vault.execute("BEGIN IMMEDIATE")
        try:
            self.vault.execute("INSERT INTO task_secrets(task_id,task_json,salt,created_at) VALUES(?,?,?,?)",
                               (task_id,json.dumps(task,sort_keys=True,separators=(",",":")),salt,created))
            self.vault.execute("COMMIT")
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute("INSERT INTO tasks(task_id,generation,commitment_digest,created_at,state) VALUES(?,?,?,?,?)",
                                  (task_id,int(generation),commitment,created,"SEALED"))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                with self.vault: self.vault.execute("DELETE FROM task_secrets WHERE task_id=?",(task_id,))
                raise
        except Exception:
            try:self.vault.execute("ROLLBACK")
            except sqlite3.OperationalError:pass
            raise
        return HiddenTaskCommitment(task_id,int(generation),commitment,created)

    def reclaim_expired(self, *, now: float | None=None) -> int:
        now=time.time() if now is None else float(now)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur=self.conn.execute("UPDATE tasks SET state='SEALED',lease_id=NULL,consumer_id=NULL,issued_at=NULL,expires_at=NULL WHERE state='LEASED' AND expires_at IS NOT NULL AND expires_at < ?",(now,))
            self.conn.execute("COMMIT"); return int(cur.rowcount)
        except Exception:
            self.conn.execute("ROLLBACK"); raise

    def lease(self, *, task_id: str, consumer_id: str, ttl_seconds: float=300.0) -> FreshTaskLease:
        now=time.time(); expires=now+float(ttl_seconds); lease_id="LEASE-"+uuid.uuid4().hex
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("UPDATE tasks SET state='SEALED',lease_id=NULL,consumer_id=NULL,issued_at=NULL,expires_at=NULL WHERE task_id=? AND state='LEASED' AND expires_at < ?",(task_id,now))
            row=self.conn.execute("SELECT generation,commitment_digest FROM tasks WHERE task_id=?",(task_id,)).fetchone()
            if row is None: raise KeyError(task_id)
            cur=self.conn.execute("UPDATE tasks SET state='LEASED',lease_id=?,consumer_id=?,issued_at=?,expires_at=? WHERE task_id=? AND state='SEALED'",
                                  (lease_id,str(consumer_id),now,expires,task_id))
            if cur.rowcount != 1: raise PermissionError("hidden task is not available for first use")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
        return FreshTaskLease(lease_id,task_id,int(row["generation"]),str(consumer_id),row["commitment_digest"],now,expires)

    def consume(self, lease: FreshTaskLease):
        now=time.time(); self.conn.execute("BEGIN IMMEDIATE")
        try:
            row=self.conn.execute("SELECT * FROM tasks WHERE task_id=?",(lease.task_id,)).fetchone()
            if row is None: raise KeyError(lease.task_id)
            if row["state"]!="LEASED" or row["lease_id"]!=lease.lease_id or row["consumer_id"]!=lease.consumer_id:
                raise PermissionError("fresh task lease is invalid or already consumed")
            if float(row["expires_at"]) < now: raise PermissionError("fresh task lease expired")
            cur=self.conn.execute("UPDATE tasks SET state='CONSUMED',consumed_at=? WHERE task_id=? AND state='LEASED' AND lease_id=?",
                                  (now,lease.task_id,lease.lease_id))
            if cur.rowcount != 1: raise PermissionError("fresh task lease lost during consume")
            secret=self.vault.execute("SELECT task_json FROM task_secrets WHERE task_id=?",(lease.task_id,)).fetchone()
            if secret is None: raise RuntimeError("sealed task secret missing")
            self.conn.execute("COMMIT"); return json.loads(secret["task_json"])
        except Exception:
            self.conn.execute("ROLLBACK"); raise

    def reveal(self, task_id: str) -> dict:
        row=self.conn.execute("SELECT * FROM tasks WHERE task_id=?",(task_id,)).fetchone()
        if row is None: raise KeyError(task_id)
        if row["state"] not in {"CONSUMED","CLOSED"}: raise PermissionError("task cannot be revealed before consumption")
        secret=self.vault.execute("SELECT task_json,salt FROM task_secrets WHERE task_id=?",(task_id,)).fetchone()
        if secret is None: raise RuntimeError("sealed task secret missing")
        task=json.loads(secret["task_json"]); salt=secret["salt"]
        return {"task_id":task_id,"task":task,"salt":salt,"commitment_digest":row["commitment_digest"],
                "commitment_valid":self.commitment(task,salt)==row["commitment_digest"]}

    def close(self, task_id: str) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur=self.conn.execute("UPDATE tasks SET state='CLOSED' WHERE task_id=? AND state='CONSUMED'",(task_id,))
            if cur.rowcount != 1: raise PermissionError("only consumed tasks can be closed")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK"); raise
