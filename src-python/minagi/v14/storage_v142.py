from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope
from .storage import ImmutableCAS


@dataclass(frozen=True)
class AuditCheckpointV142:
    checkpoint_id: str
    event_seq: int
    event_digest: str
    authority_generation: int
    created_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha2-audit-checkpoint-v1"

    def __post_init__(self) -> None:
        validate_digest(self.event_digest)
        if self.event_seq < 1 or self.authority_generation < 0:
            raise ValueError("invalid audit checkpoint generation/sequence")

    def unsigned(self) -> "AuditCheckpointV142":
        return replace(self, signer_key_id="", signature_b64="")

    @property
    def digest(self) -> str:
        return digest(self)


class GovernanceDBV142:
    """Authoritative alpha2 governance database.

    Artifact bodies live in ImmutableCAS. This store only accepts a fully bound,
    signed evaluation -> signed qualification -> signed promotion chain.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates(
          candidate_digest TEXT PRIMARY KEY,
          proposal_digest TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS builds(
          build_digest TEXT PRIMARY KEY,
          candidate_digest TEXT NOT NULL UNIQUE,
          created_at REAL NOT NULL,
          FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS evaluations(
          evaluation_digest TEXT PRIMARY KEY,
          candidate_digest TEXT NOT NULL UNIQUE,
          build_digest TEXT NOT NULL UNIQUE,
          evaluator_key_id TEXT NOT NULL,
          evaluator_generation INTEGER NOT NULL,
          metrics_digest TEXT NOT NULL,
          created_at REAL NOT NULL,
          FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT,
          FOREIGN KEY(build_digest) REFERENCES builds(build_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS qualifications(
          qualification_digest TEXT PRIMARY KEY,
          candidate_digest TEXT NOT NULL UNIQUE,
          evaluation_digest TEXT NOT NULL UNIQUE,
          decision TEXT NOT NULL CHECK(decision IN ('PROMOTE','REJECT')),
          qualifier_key_id TEXT NOT NULL,
          qualifier_generation INTEGER NOT NULL,
          authority_generation INTEGER NOT NULL,
          policy_generation INTEGER NOT NULL,
          metrics_digest TEXT NOT NULL,
          created_at REAL NOT NULL,
          FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT,
          FOREIGN KEY(evaluation_digest) REFERENCES evaluations(evaluation_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS promotions(
          authorization_digest TEXT PRIMARY KEY,
          candidate_digest TEXT NOT NULL UNIQUE,
          qualification_digest TEXT NOT NULL UNIQUE,
          authority_generation INTEGER NOT NULL,
          policy_generation INTEGER NOT NULL,
          signer_key_id TEXT NOT NULL,
          mutation_scopes_json TEXT NOT NULL,
          created_at REAL NOT NULL,
          FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT,
          FOREIGN KEY(qualification_digest) REFERENCES qualifications(qualification_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS activations(
          activation_digest TEXT PRIMARY KEY,
          candidate_digest TEXT NOT NULL,
          authorization_digest TEXT NOT NULL,
          runtime_manifest_digest TEXT NOT NULL,
          previous_runtime_manifest_digest TEXT,
          created_at REAL NOT NULL,
          FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT,
          FOREIGN KEY(authorization_digest) REFERENCES promotions(authorization_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS runtime_head(
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          runtime_manifest_digest TEXT NOT NULL,
          activation_digest TEXT NOT NULL,
          updated_at REAL NOT NULL,
          FOREIGN KEY(activation_digest) REFERENCES activations(activation_digest) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS audit_events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          event_digest TEXT NOT NULL UNIQUE,
          previous_digest TEXT,
          event_type TEXT NOT NULL,
          object_digest TEXT NOT NULL,
          actor TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_checkpoints(
          checkpoint_digest TEXT PRIMARY KEY,
          checkpoint_id TEXT NOT NULL UNIQUE,
          event_seq INTEGER NOT NULL,
          event_digest TEXT NOT NULL,
          authority_generation INTEGER NOT NULL,
          signer_key_id TEXT NOT NULL,
          signature_b64 TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        """)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    @staticmethod
    def _d(value: str) -> str:
        return validate_digest(value)

    def _audit(self, db: sqlite3.Connection, *, event_type: str, object_digest: str, actor: str, payload: dict[str, Any]) -> str:
        self._d(object_digest)
        row = db.execute("SELECT event_digest FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = None if row is None else str(row["event_digest"])
        body = {"previous_digest": previous, "event_type": str(event_type), "object_digest": object_digest,
                "actor": str(actor), "payload": payload}
        event_digest = digest(body)
        db.execute("INSERT INTO audit_events(event_digest,previous_digest,event_type,object_digest,actor,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                   (event_digest, previous, str(event_type), object_digest, str(actor),
                    json.dumps(payload, sort_keys=True, separators=(",", ":")), time.time()))
        return event_digest

    def register_candidate(self, *, candidate_digest: str, proposal_digest: str, actor: str = "candidate-builder") -> None:
        candidate_digest, proposal_digest = self._d(candidate_digest), self._d(proposal_digest)
        with self.transaction() as db:
            db.execute("INSERT INTO candidates(candidate_digest,proposal_digest,created_at) VALUES(?,?,?)",
                       (candidate_digest, proposal_digest, time.time()))
            self._audit(db, event_type="candidate.registered", object_digest=candidate_digest, actor=actor,
                        payload={"proposal_digest": proposal_digest})

    def record_build(self, *, candidate_digest: str, build_digest: str, actor: str = "builder") -> None:
        candidate_digest, build_digest = self._d(candidate_digest), self._d(build_digest)
        with self.transaction() as db:
            db.execute("INSERT INTO builds(build_digest,candidate_digest,created_at) VALUES(?,?,?)",
                       (build_digest, candidate_digest, time.time()))
            self._audit(db, event_type="candidate.built", object_digest=build_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest})

    def record_evaluation(self, *, candidate_digest: str, build_digest: str, evaluation_digest: str,
                          evaluator_key_id: str, evaluator_generation: int, metrics_digest: str,
                          actor: str = "evaluation-authority") -> None:
        candidate_digest, build_digest, evaluation_digest, metrics_digest = map(self._d,
            (candidate_digest, build_digest, evaluation_digest, metrics_digest))
        if not evaluator_key_id or evaluator_generation < 0:
            raise ValueError("signed evaluator identity/generation required")
        with self.transaction() as db:
            row = db.execute("SELECT candidate_digest FROM builds WHERE build_digest=?", (build_digest,)).fetchone()
            if row is None or row["candidate_digest"] != candidate_digest:
                raise PermissionError("evaluation build/candidate binding mismatch")
            db.execute("INSERT INTO evaluations(evaluation_digest,candidate_digest,build_digest,evaluator_key_id,evaluator_generation,metrics_digest,created_at) VALUES(?,?,?,?,?,?,?)",
                       (evaluation_digest, candidate_digest, build_digest, evaluator_key_id, int(evaluator_generation), metrics_digest, time.time()))
            self._audit(db, event_type="candidate.evaluated", object_digest=evaluation_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "build_digest": build_digest,
                                 "evaluator_key_id": evaluator_key_id, "evaluator_generation": int(evaluator_generation),
                                 "metrics_digest": metrics_digest})

    def record_qualification(self, *, candidate_digest: str, evaluation_digest: str, qualification_digest: str,
                             decision: str, qualifier_key_id: str, qualifier_generation: int,
                             authority_generation: int, policy_generation: int, metrics_digest: str,
                             actor: str = "qualification-authority") -> None:
        candidate_digest, evaluation_digest, qualification_digest, metrics_digest = map(self._d,
            (candidate_digest, evaluation_digest, qualification_digest, metrics_digest))
        decision = str(decision).upper()
        if decision not in {"PROMOTE", "REJECT"}:
            raise ValueError("invalid qualification decision")
        if not qualifier_key_id or qualifier_generation < 0:
            raise ValueError("signed qualifier identity/generation required")
        with self.transaction() as db:
            row = db.execute("SELECT candidate_digest,metrics_digest FROM evaluations WHERE evaluation_digest=?", (evaluation_digest,)).fetchone()
            if row is None or row["candidate_digest"] != candidate_digest:
                raise PermissionError("qualification evaluation/candidate binding mismatch")
            if row["metrics_digest"] != metrics_digest:
                raise PermissionError("qualification metrics/evaluation mismatch")
            db.execute("INSERT INTO qualifications(qualification_digest,candidate_digest,evaluation_digest,decision,qualifier_key_id,qualifier_generation,authority_generation,policy_generation,metrics_digest,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (qualification_digest, candidate_digest, evaluation_digest, decision, qualifier_key_id,
                        int(qualifier_generation), int(authority_generation), int(policy_generation), metrics_digest, time.time()))
            self._audit(db, event_type="candidate.qualified", object_digest=qualification_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "evaluation_digest": evaluation_digest,
                                 "decision": decision, "qualifier_key_id": qualifier_key_id,
                                 "qualifier_generation": int(qualifier_generation), "metrics_digest": metrics_digest,
                                 "authority_generation": int(authority_generation), "policy_generation": int(policy_generation)})

    def record_promotion(self, *, candidate_digest: str, qualification_digest: str, authorization_digest: str,
                         authority_generation: int, policy_generation: int, signer_key_id: str,
                         mutation_scopes: tuple[str, ...], actor: str = "promotion-authority") -> None:
        candidate_digest, qualification_digest, authorization_digest = map(self._d,
            (candidate_digest, qualification_digest, authorization_digest))
        scopes = tuple(sorted(set(str(x) for x in mutation_scopes if str(x))))
        if not scopes:
            raise ValueError("promotion authorization requires mutation scope")
        with self.transaction() as db:
            q = db.execute("SELECT * FROM qualifications WHERE qualification_digest=?", (qualification_digest,)).fetchone()
            if q is None or q["candidate_digest"] != candidate_digest:
                raise PermissionError("promotion qualification/candidate binding mismatch")
            if q["decision"] != "PROMOTE":
                raise PermissionError("rejected qualification cannot be promoted")
            if int(q["authority_generation"]) != int(authority_generation) or int(q["policy_generation"]) != int(policy_generation):
                raise PermissionError("promotion generation mismatch")
            db.execute("INSERT INTO promotions(authorization_digest,candidate_digest,qualification_digest,authority_generation,policy_generation,signer_key_id,mutation_scopes_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (authorization_digest, candidate_digest, qualification_digest, int(authority_generation),
                        int(policy_generation), str(signer_key_id), json.dumps(scopes), time.time()))
            self._audit(db, event_type="candidate.authorized", object_digest=authorization_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "qualification_digest": qualification_digest,
                                 "mutation_scopes": scopes, "signer_key_id": str(signer_key_id),
                                 "authority_generation": int(authority_generation), "policy_generation": int(policy_generation)})

    def activation(self, authorization_digest: str):
        return self.conn.execute("SELECT * FROM promotions WHERE authorization_digest=?", (self._d(authorization_digest),)).fetchone()

    def qualification(self, qualification_digest: str):
        return self.conn.execute("SELECT * FROM qualifications WHERE qualification_digest=?", (self._d(qualification_digest),)).fetchone()

    def evaluation(self, evaluation_digest: str):
        return self.conn.execute("SELECT * FROM evaluations WHERE evaluation_digest=?", (self._d(evaluation_digest),)).fetchone()

    def candidate(self, candidate_digest: str):
        return self.conn.execute("SELECT * FROM candidates WHERE candidate_digest=?", (self._d(candidate_digest),)).fetchone()

    def build(self, build_digest: str):
        return self.conn.execute("SELECT * FROM builds WHERE build_digest=?", (self._d(build_digest),)).fetchone()

    def require_authorization(self, authorization_digest: str, *, mutation_scope: str) -> sqlite3.Row:
        row = self.activation(authorization_digest)
        if row is None:
            raise PermissionError("promotion authorization not found in authoritative store")
        scopes = set(json.loads(row["mutation_scopes_json"]))
        if mutation_scope not in scopes:
            raise PermissionError(f"promotion authorization does not permit {mutation_scope}")
        return row

    def activate(self, *, candidate_digest: str, authorization_digest: str, runtime_manifest_digest: str,
                 activation_digest: str, actor: str = "persistence-gateway") -> None:
        candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest = map(self._d,
            (candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest))
        with self.transaction() as db:
            p = db.execute("SELECT candidate_digest,mutation_scopes_json FROM promotions WHERE authorization_digest=?",
                           (authorization_digest,)).fetchone()
            if p is None or p["candidate_digest"] != candidate_digest:
                raise PermissionError("activation authorization/candidate binding mismatch")
            if "runtime.activate" not in set(json.loads(p["mutation_scopes_json"])):
                raise PermissionError("promotion authorization lacks runtime.activate scope")
            head = db.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
            previous = None if head is None else str(head["runtime_manifest_digest"])
            db.execute("INSERT INTO activations(activation_digest,candidate_digest,authorization_digest,runtime_manifest_digest,previous_runtime_manifest_digest,created_at) VALUES(?,?,?,?,?,?)",
                       (activation_digest, candidate_digest, authorization_digest, runtime_manifest_digest, previous, time.time()))
            db.execute("INSERT INTO runtime_head(singleton,runtime_manifest_digest,activation_digest,updated_at) VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET runtime_manifest_digest=excluded.runtime_manifest_digest,activation_digest=excluded.activation_digest,updated_at=excluded.updated_at",
                       (runtime_manifest_digest, activation_digest, time.time()))
            self._audit(db, event_type="runtime.activated", object_digest=activation_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "authorization_digest": authorization_digest,
                                 "runtime_manifest_digest": runtime_manifest_digest, "previous_runtime_manifest_digest": previous})

    def current_runtime_manifest(self) -> str | None:
        row = self.conn.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
        return None if row is None else str(row["runtime_manifest_digest"])

    def verify_audit_chain(self) -> bool:
        previous = None
        for row in self.conn.execute("SELECT * FROM audit_events ORDER BY seq"):
            payload = json.loads(row["payload_json"])
            expected = digest({"previous_digest": previous, "event_type": row["event_type"],
                               "object_digest": row["object_digest"], "actor": row["actor"], "payload": payload})
            if row["previous_digest"] != previous or row["event_digest"] != expected:
                raise RuntimeError("audit hash chain verification failed")
            previous = row["event_digest"]
        return True

    def create_audit_checkpoint(self, *, signer, authority_generation: int) -> AuditCheckpointV142:
        row = self.conn.execute("SELECT seq,event_digest FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("cannot checkpoint an empty audit ledger")
        unsigned = AuditCheckpointV142(
            checkpoint_id=f"ACP14-{int(time.time()*1_000_000)}",
            event_seq=int(row["seq"]), event_digest=str(row["event_digest"]),
            authority_generation=int(authority_generation), created_at=time.time())
        env = signer.sign(asdict(unsigned.unsigned()))
        cp = replace(unsigned, signer_key_id=env.key_id, signature_b64=env.signature_b64)
        with self.transaction() as db:
            db.execute("INSERT INTO audit_checkpoints(checkpoint_digest,checkpoint_id,event_seq,event_digest,authority_generation,signer_key_id,signature_b64,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (cp.digest, cp.checkpoint_id, cp.event_seq, cp.event_digest, cp.authority_generation,
                        cp.signer_key_id, cp.signature_b64, cp.created_at))
        return cp

    def verify_audit_checkpoints(self, *, verifier, trusted_key_ids: set[str]) -> bool:
        trusted = set(trusted_key_ids)
        self.verify_audit_chain()
        for row in self.conn.execute("SELECT * FROM audit_checkpoints ORDER BY event_seq"):
            cp = AuditCheckpointV142(
                checkpoint_id=row["checkpoint_id"], event_seq=int(row["event_seq"]), event_digest=row["event_digest"],
                authority_generation=int(row["authority_generation"]), created_at=float(row["created_at"]),
                signer_key_id=row["signer_key_id"], signature_b64=row["signature_b64"])
            if cp.signer_key_id not in trusted:
                raise PermissionError("untrusted audit checkpoint signer")
            ev = self.conn.execute("SELECT event_digest FROM audit_events WHERE seq=?", (cp.event_seq,)).fetchone()
            if ev is None or ev["event_digest"] != cp.event_digest:
                raise RuntimeError("audit checkpoint/event binding mismatch")
            if not verifier.verify(asdict(cp.unsigned()), SignedEnvelope(cp.signer_key_id, cp.signature_b64)):
                raise PermissionError("invalid audit checkpoint signature")
        return True

    def close(self) -> None:
        self.conn.close()
