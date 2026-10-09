from __future__ import annotations

import json
import time
from typing import Iterable, Mapping

from egai.common.canonical import digest, validate_digest
from .storage_v143 import GovernanceDBV143


class GovernanceDBV144(GovernanceDBV143):
    """Alpha4 single-authority database for governance + learned-state heads.

    Artifact bytes still live in CAS.  This DB atomically consumes exact mutation
    commitments, advances belief/skill heads, and activates the RuntimeManifest in
    one SQLite transaction so a crash cannot leave learned state and runtime head
    on opposite sides of a promotion.
    """

    def __init__(self, path):
        super().__init__(path)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS verified_experiences(
              verified_evidence_digest TEXT PRIMARY KEY,
              experience_digest TEXT NOT NULL UNIQUE,
              evidence_root_digest TEXT NOT NULL,
              verification_receipt_digest TEXT NOT NULL UNIQUE,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS belief_revisions_v144(
              belief_digest TEXT PRIMARY KEY,
              belief_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              body_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE(belief_id,revision)
            );
            CREATE TABLE IF NOT EXISTS belief_heads_v144(
              belief_id TEXT PRIMARY KEY,
              belief_digest TEXT NOT NULL,
              updated_at REAL NOT NULL,
              FOREIGN KEY(belief_digest) REFERENCES belief_revisions_v144(belief_digest) ON DELETE RESTRICT
            );
            CREATE TABLE IF NOT EXISTS skill_revisions_v144(
              skill_digest TEXT PRIMARY KEY,
              skill_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              status TEXT NOT NULL,
              body_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE(skill_id,revision)
            );
            CREATE TABLE IF NOT EXISTS skill_heads_v144(
              skill_id TEXT PRIMARY KEY,
              skill_digest TEXT NOT NULL,
              status TEXT NOT NULL,
              updated_at REAL NOT NULL,
              FOREIGN KEY(skill_digest) REFERENCES skill_revisions_v144(skill_digest) ON DELETE RESTRICT
            );
            """
        )

    def record_verified_experience_v144(self, *, experience_digest: str, evidence_root_digest: str,
                                        verification_receipt_digest: str, verified_evidence_digest: str,
                                        actor: str = "experience-pipeline"):
        vals = tuple(map(validate_digest, (experience_digest, evidence_root_digest,
                                           verification_receipt_digest, verified_evidence_digest)))
        experience_digest, evidence_root_digest, verification_receipt_digest, verified_evidence_digest = vals
        with self.transaction() as db:
            db.execute(
                "INSERT INTO verified_experiences(verified_evidence_digest,experience_digest,evidence_root_digest,verification_receipt_digest,created_at) VALUES(?,?,?,?,?)",
                (verified_evidence_digest, experience_digest, evidence_root_digest, verification_receipt_digest, time.time()),
            )
            self._audit(db, event_type="experience.verified.v144", object_digest=verified_evidence_digest,
                        actor=actor, payload={"experience_digest": experience_digest,
                                              "evidence_root_digest": evidence_root_digest,
                                              "verification_receipt_digest": verification_receipt_digest})

    @staticmethod
    def _require_open_commitment(db, authorization_digest: str, scope: str, target_digest: str):
        row = db.execute(
            "SELECT consumed_at FROM mutation_commitments WHERE authorization_digest=? AND mutation_scope=? AND target_digest=?",
            (authorization_digest, scope, target_digest),
        ).fetchone()
        if row is None:
            raise PermissionError(f"authorization missing exact {scope} mutation")
        if row["consumed_at"] is not None:
            raise PermissionError(f"{scope} mutation authorization already consumed")

    @staticmethod
    def _consume_commitment(db, authorization_digest: str, scope: str, target_digest: str, now: float):
        cur = db.execute(
            "UPDATE mutation_commitments SET consumed_at=? WHERE authorization_digest=? AND mutation_scope=? AND target_digest=? AND consumed_at IS NULL",
            (now, authorization_digest, scope, target_digest),
        )
        if cur.rowcount != 1:
            raise PermissionError(f"{scope} mutation authorization absent or already consumed")

    def _apply_belief(self, db, body: Mapping, target_digest: str, now: float):
        if body.get("schema") != "mini-agi-v14.1-alpha4-belief-revision-v1":
            raise PermissionError("invalid belief mutation schema")
        if digest(dict(body)) != target_digest:
            raise PermissionError("belief target digest/body mismatch")
        belief_id = str(body["belief_id"]); revision = int(body["revision"])
        head = db.execute("SELECT belief_digest FROM belief_heads_v144 WHERE belief_id=?", (belief_id,)).fetchone()
        prev = "" if head is None else str(head["belief_digest"])
        if str(body.get("supersedes_digest", "")) != prev:
            raise PermissionError("belief supersedes current head mismatch")
        expected_revision = 1 if head is None else int(db.execute(
            "SELECT revision FROM belief_revisions_v144 WHERE belief_digest=?", (prev,)).fetchone()["revision"]) + 1
        if revision != expected_revision:
            raise PermissionError("belief revision is not next authoritative revision")
        db.execute(
            "INSERT INTO belief_revisions_v144(belief_digest,belief_id,revision,body_json,created_at) VALUES(?,?,?,?,?)",
            (target_digest, belief_id, revision, json.dumps(dict(body), sort_keys=True, separators=(",", ":")), float(body["created_at"])),
        )
        db.execute(
            "INSERT INTO belief_heads_v144(belief_id,belief_digest,updated_at) VALUES(?,?,?) ON CONFLICT(belief_id) DO UPDATE SET belief_digest=excluded.belief_digest,updated_at=excluded.updated_at",
            (belief_id, target_digest, now),
        )

    def _apply_skill(self, db, body: Mapping, target_digest: str, now: float):
        if body.get("schema") != "mini-agi-v14.1-alpha4-skill-revision-v1":
            raise PermissionError("invalid skill mutation schema")
        if digest(dict(body)) != target_digest:
            raise PermissionError("skill target digest/body mismatch")
        skill_id = str(body["skill_id"]); revision = int(body["revision"]); status = str(body["status"])
        head = db.execute("SELECT skill_digest,status FROM skill_heads_v144 WHERE skill_id=?", (skill_id,)).fetchone()
        prev = "" if head is None else str(head["skill_digest"])
        if str(body.get("supersedes_digest", "")) != prev:
            raise PermissionError("skill supersedes current head mismatch")
        expected_revision = 1 if head is None else int(db.execute(
            "SELECT revision FROM skill_revisions_v144 WHERE skill_digest=?", (prev,)).fetchone()["revision"]) + 1
        if revision != expected_revision:
            raise PermissionError("skill revision is not next authoritative revision")
        db.execute(
            "INSERT INTO skill_revisions_v144(skill_digest,skill_id,revision,status,body_json,created_at) VALUES(?,?,?,?,?,?)",
            (target_digest, skill_id, revision, status, json.dumps(dict(body), sort_keys=True, separators=(",", ":")), float(body["created_at"])),
        )
        db.execute(
            "INSERT INTO skill_heads_v144(skill_id,skill_digest,status,updated_at) VALUES(?,?,?,?) ON CONFLICT(skill_id) DO UPDATE SET skill_digest=excluded.skill_digest,status=excluded.status,updated_at=excluded.updated_at",
            (skill_id, target_digest, status, now),
        )

    def activate_with_mutations_v144(self, *, candidate_digest: str, authorization_digest: str,
                                     runtime_manifest_digest: str, activation_digest: str,
                                     mutations: Iterable[tuple[str, str, Mapping]], actor: str = "persistence-gateway"):
        candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest = map(
            validate_digest, (candidate_digest, authorization_digest, runtime_manifest_digest, activation_digest)
        )
        mutations = tuple((str(scope), validate_digest(target), dict(body)) for scope, target, body in mutations)
        now = time.time()
        with self.transaction() as db:
            p = db.execute("SELECT * FROM promotions WHERE authorization_digest=?", (authorization_digest,)).fetchone()
            if p is None or p["candidate_digest"] != candidate_digest:
                raise PermissionError("activation authorization/candidate mismatch")
            if p["runtime_manifest_digest"] != runtime_manifest_digest:
                raise PermissionError("authorization not bound to runtime manifest")
            if p["status"] != "ISSUED":
                raise PermissionError("promotion authorization not open")
            self._require_open_commitment(db, authorization_digest, "runtime.activate", runtime_manifest_digest)
            for scope, target, body in mutations:
                self._require_open_commitment(db, authorization_digest, scope, target)
                if scope == "belief.promote":
                    self._apply_belief(db, body, target, now)
                elif scope == "skill.activate":
                    self._apply_skill(db, body, target, now)
                else:
                    raise PermissionError(f"alpha4 transactional persistence does not support scope {scope}")
                self._consume_commitment(db, authorization_digest, scope, target, now)
                self._audit(db, event_type="mutation.applied.v144", object_digest=target, actor=actor,
                            payload={"authorization_digest": authorization_digest, "mutation_scope": scope})
            self._consume_commitment(db, authorization_digest, "runtime.activate", runtime_manifest_digest, now)
            head = db.execute("SELECT runtime_manifest_digest FROM runtime_head WHERE singleton=1").fetchone()
            previous = None if head is None else str(head["runtime_manifest_digest"])
            db.execute(
                "INSERT INTO activations(activation_digest,candidate_digest,authorization_digest,runtime_manifest_digest,previous_runtime_manifest_digest,created_at) VALUES(?,?,?,?,?,?)",
                (activation_digest, candidate_digest, authorization_digest, runtime_manifest_digest, previous, now),
            )
            db.execute(
                "INSERT INTO runtime_head(singleton,runtime_manifest_digest,activation_digest,updated_at) VALUES(1,?,?,?) ON CONFLICT(singleton) DO UPDATE SET runtime_manifest_digest=excluded.runtime_manifest_digest,activation_digest=excluded.activation_digest,updated_at=excluded.updated_at",
                (runtime_manifest_digest, activation_digest, now),
            )
            db.execute("UPDATE promotions SET status='CONSUMED',consumed_at=? WHERE authorization_digest=?", (now, authorization_digest))
            self._audit(db, event_type="runtime.atomic_content_activated.v144", object_digest=activation_digest, actor=actor,
                        payload={"candidate_digest": candidate_digest, "authorization_digest": authorization_digest,
                                 "runtime_manifest_digest": runtime_manifest_digest,
                                 "previous_runtime_manifest_digest": previous,
                                 "mutation_targets": [(s, t) for s, t, _ in mutations]})

    def belief_head(self, belief_id: str):
        return self.conn.execute("SELECT * FROM belief_heads_v144 WHERE belief_id=?", (str(belief_id),)).fetchone()

    def skill_head(self, skill_id: str):
        return self.conn.execute("SELECT * FROM skill_heads_v144 WHERE skill_id=?", (str(skill_id),)).fetchone()
