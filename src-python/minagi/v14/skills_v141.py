from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
import uuid

from egai.common.canonical import digest, validate_digest

SKILL_TRANSITIONS = {
    None: {"DISCOVERED"},
    "DISCOVERED": {"PROPOSED", "RETIRED"},
    "PROPOSED": {"CANDIDATE", "RETIRED"},
    "CANDIDATE": {"QUALIFIED", "RETIRED"},
    "QUALIFIED": {"ACTIVE", "RETIRED"},
    "ACTIVE": {"DEGRADED", "RETIRED"},
    "DEGRADED": {"ACTIVE", "RETIRED"},
    "RETIRED": set(),
}

@dataclass(frozen=True)
class SkillRevisionV141:
    skill_id: str
    revision: int
    status: str
    procedure_digest: str
    positive_evidence: tuple[str, ...]
    negative_evidence: tuple[str, ...]
    failure_envelope_digest: str
    supersedes_digest: str = ""
    created_at: float = 0.0
    schema: str = "mini-agi-v14.1-skill-revision-v1"

    def __post_init__(self) -> None:
        if self.status not in {x for xs in SKILL_TRANSITIONS.values() for x in xs} | set(SKILL_TRANSITIONS):
            raise ValueError("invalid skill status")
        validate_digest(self.procedure_digest); validate_digest(self.failure_envelope_digest)
        for d in self.positive_evidence + self.negative_evidence: validate_digest(d)
        if self.supersedes_digest: validate_digest(self.supersedes_digest)
        if not self.created_at: object.__setattr__(self,"created_at",time.time())

    @property
    def digest(self) -> str: return digest(self)

class SkillLifecycleStore:
    def __init__(self, path: str | Path, authorization_gate=None):
        self.authorization_gate=authorization_gate
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.path,isolation_level=None); self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS skill_revisions(
          skill_digest TEXT PRIMARY KEY, skill_id TEXT NOT NULL, revision INTEGER NOT NULL,
          status TEXT NOT NULL, body_json TEXT NOT NULL, created_at REAL NOT NULL,
          UNIQUE(skill_id,revision));
        CREATE TABLE IF NOT EXISTS skill_heads(
          skill_id TEXT PRIMARY KEY, skill_digest TEXT NOT NULL, status TEXT NOT NULL, updated_at REAL NOT NULL);
        """)

    def transition(self, *, status: str, procedure_digest: str, positive_evidence: tuple[str,...], negative_evidence: tuple[str,...],
                   failure_envelope_digest: str, skill_id: str | None = None, authorization_digest: str = "") -> SkillRevisionV141:
        status=str(status).upper(); skill_id=skill_id or "SK-"+uuid.uuid4().hex
        if status in {"ACTIVE","RETIRED"}:
            validate_digest(authorization_digest)
            if self.authorization_gate is not None:
                self.authorization_gate.require(authorization_digest, mutation_scope="skill.activate")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            head=self.db.execute("SELECT skill_digest,status FROM skill_heads WHERE skill_id=?",(skill_id,)).fetchone()
            previous_status=None if head is None else str(head["status"]); previous_digest="" if head is None else str(head["skill_digest"])
            if status not in SKILL_TRANSITIONS.get(previous_status,set()):
                raise PermissionError(f"illegal skill transition {previous_status!r} -> {status!r}")
            row=self.db.execute("SELECT MAX(revision) AS r FROM skill_revisions WHERE skill_id=?",(skill_id,)).fetchone(); rev=int(row["r"] or 0)+1
            obj=SkillRevisionV141(skill_id,rev,status,procedure_digest,tuple(positive_evidence),tuple(negative_evidence),failure_envelope_digest,previous_digest)
            body=json.dumps(asdict(obj),sort_keys=True,separators=(",",":"))
            self.db.execute("INSERT INTO skill_revisions(skill_digest,skill_id,revision,status,body_json,created_at) VALUES(?,?,?,?,?,?)",
                            (obj.digest,skill_id,rev,status,body,obj.created_at))
            self.db.execute("INSERT INTO skill_heads(skill_id,skill_digest,status,updated_at) VALUES(?,?,?,?) ON CONFLICT(skill_id) DO UPDATE SET skill_digest=excluded.skill_digest,status=excluded.status,updated_at=excluded.updated_at",
                            (skill_id,obj.digest,status,time.time()))
            self.db.execute("COMMIT"); return obj
        except Exception:
            self.db.execute("ROLLBACK"); raise

    def history(self, skill_id: str) -> tuple[SkillRevisionV141,...]:
        out=[]
        for row in self.db.execute("SELECT body_json FROM skill_revisions WHERE skill_id=? ORDER BY revision",(skill_id,)):
            body=json.loads(row["body_json"]); body["positive_evidence"]=tuple(body["positive_evidence"]);body["negative_evidence"]=tuple(body["negative_evidence"]); out.append(SkillRevisionV141(**body))
        return tuple(out)
