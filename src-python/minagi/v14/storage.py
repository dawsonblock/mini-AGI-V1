from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any, Iterator

from egai.common.canonical import canonical_bytes, digest, validate_digest

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImmutableCAS:
    """Small fail-closed content-addressed store.

    Objects are immutable and addressed only by validated sha256 digests.  The
    parser deliberately rejects alternate algorithms, short hashes, separators,
    traversal tokens and arbitrary filenames before filesystem access occurs.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "sha256").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate(value: str) -> str:
        if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
            raise ValueError("invalid CAS digest")
        return value

    def _path(self, value: str) -> Path:
        value = self._validate(value)
        hexpart = value.split(":", 1)[1]
        return self.root / "sha256" / hexpart[:2] / hexpart

    def put_bytes(self, payload: bytes) -> str:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("CAS payload must be bytes-like")
        data = bytes(payload)
        value = "sha256:" + hashlib.sha256(data).hexdigest()
        target = self._path(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != value.split(":", 1)[1]:
                raise RuntimeError("CAS corruption detected")
            return value
        tmp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        try:
            fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        return value

    def put_json(self, obj: Any) -> str:
        return self.put_bytes(canonical_bytes(obj))

    def get_bytes(self, value: str) -> bytes:
        target = self._path(value)
        data = target.read_bytes()
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != value:
            raise RuntimeError("CAS digest mismatch")
        return data

    def exists(self, value: str) -> bool:
        return self._path(value).is_file()


@dataclass(frozen=True)
class ChainRecord:
    artifact_digest: str
    candidate_digest: str
    parent_digest: str
    created_at: float


class GovernanceDB:
    """Transactional authoritative state for the governed artifact chain.

    The database records *bindings*, not mutable artifact bodies. Artifact bytes
    live in ImmutableCAS. SQLite foreign keys and transactions make it impossible
    to publish a qualification, promotion or activation without its prerequisite
    chain being present in the same authority store.
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
        self.conn.executescript(
            """
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
              created_at REAL NOT NULL,
              FOREIGN KEY(candidate_digest) REFERENCES candidates(candidate_digest) ON DELETE RESTRICT,
              FOREIGN KEY(build_digest) REFERENCES builds(build_digest) ON DELETE RESTRICT
            );
            CREATE TABLE IF NOT EXISTS qualifications(
              qualification_digest TEXT PRIMARY KEY,
              candidate_digest TEXT NOT NULL UNIQUE,
              evaluation_digest TEXT NOT NULL UNIQUE,
              decision TEXT NOT NULL CHECK(decision IN ('PROMOTE','REJECT')),
              authority_generation INTEGER NOT NULL,
              policy_generation INTEGER NOT NULL,
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
            """
        )

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
        body = {
            "previous_digest": previous,
            "event_type": str(event_type),
            "object_digest": object_digest,
            "actor": str(actor),
            "payload": payload,
        }
        event_digest = digest(body)
        db.execute(
            "INSERT INTO audit_events(event_digest,previous_digest,event_type,object_digest,actor,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (event_digest, previous, str(event_type), object_digest, str(actor), json.dumps(payload, sort_keys=True, separators=(",", ":")), time.time()),
        )
        return event_digest

    def register_candidate(self, *, candidate_digest: str, proposal_digest: str, actor: str = "candidate-builder") -> None:
        candidate_digest, proposal_digest = self._d(candidate_digest), self._d(proposal_digest)
        with self.transaction() as db:
            db.execute("INSERT INTO candidates(candidate_digest,proposal_digest,created_at) VALUES(?,?,?)", (candidate_digest, proposal_digest, time.time()))
            self._audit(db, event_type="candidate.registered", object_digest=candidate_digest, actor=actor, payload={"proposal_digest": proposal_digest})

    def record_build(self, *, candidate_digest: str, build_digest: str, actor: str = "builder") -> None:
        candidate_digest, build_digest = self._d(candidate_digest), self._d(build_digest)
        with self.transaction() as db:
            db.execute("INSERT INTO builds(build_digest,candidate_digest,created_at) VALUES(?,?,?)", (build_digest, candidate_digest, time.time()))
            self._audit(db, event_type="candidate.built", object_digest=build_digest, actor=actor, payload={"candidate_digest": candidate_digest})

    def record_evaluation(self, *, candidate_digest: str, build_digest: str, evaluation_digest: str, actor: str = "independent-evaluator") -> None:
        candidate_digest, build_digest, evaluation_digest = map(self._d, (candidate_digest, build_digest, evaluation_digest))
        with self.transaction() as db:
            row = db.execute("SELECT candidate_digest FROM builds WHERE build_digest=?", (build_digest,)).fetchone()
            if row is None or row["candidate_digest"] != candidate_digest:
                raise PermissionError("evaluation build/candidate binding mismatch")
            db.execute("INSERT INTO evaluations(evaluation_digest,candidate_digest,build_digest,created_at) VALUES(?,?,?,?)", (evaluation_digest, candidate_digest, build_digest, time.time()))
            self._audit(db, event_type="candidate.evaluated", object_digest=evaluation_digest, actor=actor, payload={"candidate_digest": candidate_digest, "build_digest": build_digest})

    def record_qualification(self, *, candidate_digest: str, evaluation_digest: str, qualification_digest: str,
                             decision: str, authority_generation: int, policy_generation: int, actor: str = "qualifier") -> None:
        candidate_digest, evaluation_digest, qualification_digest = map(self._d, (candidate_digest, evaluation_digest, qualification_digest))
        decision = str(decision).upper()
        if decision not in {"PROMOTE", "REJECT"}:
            raise ValueError("invalid qualification decision")
        with self.transaction() as db:
            row = db.execute("SELECT candidate_digest FROM evaluations WHERE evaluation_digest=?", (evaluation_digest,)).fetchone()
            if row is None or row["candidate_digest"] != candidate_digest:
                raise PermissionError("qualification evaluation/candidate binding mismatch")
            db.execute(
                "INSERT INTO qualifications(qualification_digest,candidate_digest,evaluation_digest,decision,authority_generation,policy_generation,created_at) VALUES(?,?,?,?,?,?,?)",
                (qualification_digest, candidate_digest, evaluation_digest, decision, int(authority_generation), int(policy_generation), time.time()),
            )
            self._audit(db, event_type="candidate.qualified", object_digest=qualification_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "evaluation_digest": evaluation_digest, "decision": decision,
                                 "authority_generation": int(authority_generation), "policy_generation": int(policy_generation)})

    def record_promotion(self, *, candidate_digest: str, qualification_digest: str, authorization_digest: str,
                         authority_generation: int, policy_generation: int, signer_key_id: str,
                         actor: str = "promotion-authority") -> None:
        candidate_digest, qualification_digest, authorization_digest = map(self._d, (candidate_digest, qualification_digest, authorization_digest))
        with self.transaction() as db:
            q = db.execute("SELECT * FROM qualifications WHERE qualification_digest=?", (qualification_digest,)).fetchone()
            if q is None or q["candidate_digest"] != candidate_digest:
                raise PermissionError("promotion qualification/candidate binding mismatch")
            if q["decision"] != "PROMOTE":
                raise PermissionError("rejected qualification cannot be promoted")
            if int(q["authority_generation"]) != int(authority_generation) or int(q["policy_generation"]) != int(policy_generation):
                raise PermissionError("promotion generation mismatch")
            db.execute(
                "INSERT INTO promotions(authorization_digest,candidate_digest,qualification_digest,authority_generation,policy_generation,signer_key_id,created_at) VALUES(?,?,?,?,?,?,?)",
                (authorization_digest, candidate_digest, qualification_digest, int(authority_generation), int(policy_generation), str(signer_key_id), time.time()),
            )
            self._audit(db, event_type="candidate.authorized", object_digest=authorization_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "qualification_digest": qualification_digest,
                                 "authority_generation": int(authority_generation), "policy_generation": int(policy_generation), "signer_key_id": str(signer_key_id)})

    def activate(self, *, candidate_digest: str, authorization_digest: str, runtime_manifest_digest: str,
                 activation_digest: str, actor: str = "persistence-gateway") -> None:
        candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest = map(
            self._d, (candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest)
        )
        with self.transaction() as db:
            p = db.execute("SELECT candidate_digest FROM promotions WHERE authorization_digest=?", (authorization_digest,)).fetchone()
            if p is None or p["candidate_digest"] != candidate_digest:
                raise PermissionError("activation authorization/candidate binding mismatch")
            head = db.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
            previous = None if head is None else str(head["runtime_manifest_digest"])
            db.execute(
                "INSERT INTO activations(activation_digest,candidate_digest,authorization_digest,runtime_manifest_digest,previous_runtime_manifest_digest,created_at) VALUES(?,?,?,?,?,?)",
                (activation_digest, candidate_digest, authorization_digest, runtime_manifest_digest, previous, time.time()),
            )
            db.execute(
                "INSERT INTO runtime_head(singleton,runtime_manifest_digest,activation_digest,updated_at) VALUES(1,?,?,?) "
                "ON CONFLICT(singleton) DO UPDATE SET runtime_manifest_digest=excluded.runtime_manifest_digest,activation_digest=excluded.activation_digest,updated_at=excluded.updated_at",
                (runtime_manifest_digest, activation_digest, time.time()),
            )
            self._audit(db, event_type="runtime.activated", object_digest=activation_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "authorization_digest": authorization_digest,
                                 "runtime_manifest_digest": runtime_manifest_digest, "previous_runtime_manifest_digest": previous})

    def current_runtime_manifest(self) -> str | None:
        row = self.conn.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
        return None if row is None else str(row["runtime_manifest_digest"])

    def qualification(self, qualification_digest: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM qualifications WHERE qualification_digest=?", (self._d(qualification_digest),)).fetchone()

    def verify_audit_chain(self) -> bool:
        previous = None
        for row in self.conn.execute("SELECT * FROM audit_events ORDER BY seq"):
            payload = json.loads(row["payload_json"])
            body = {
                "previous_digest": previous,
                "event_type": row["event_type"],
                "object_digest": row["object_digest"],
                "actor": row["actor"],
                "payload": payload,
            }
            expected = digest(body)
            if row["previous_digest"] != previous or row["event_digest"] != expected:
                raise RuntimeError("audit hash chain verification failed")
            previous = row["event_digest"]
        return True

    def close(self) -> None:
        self.conn.close()
