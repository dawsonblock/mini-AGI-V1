from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Mapping

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import sha256_json
from .authority_artifacts import (
    AuthorityArtifactStore, BuildManifestRC14, CandidateManifestRC14, EvaluationBundleRC14,
    PromotionAuthorizationRC14, PromotionDecisionRC14, QualificationEvidenceRC14, QualificationRecordRC14,
    ContinualExperimentEvidenceRC14,
)
from .authority_state import AuthorityStateRegistry
from .models import CandidateState, require_digest


LEGAL_TRANSITIONS: dict[str | None, set[str]] = {
    None: {CandidateState.REGISTERED.value},
    CandidateState.REGISTERED.value: {CandidateState.BUILT.value, CandidateState.REJECTED.value},
    CandidateState.BUILT.value: {CandidateState.EVALUATED.value, CandidateState.REJECTED.value},
    CandidateState.EVALUATED.value: {CandidateState.QUALIFIED.value, CandidateState.REJECTED.value},
    CandidateState.QUALIFIED.value: {CandidateState.AUTHORIZED.value, CandidateState.REJECTED.value},
    CandidateState.AUTHORIZED.value: {CandidateState.EPOCH_PREPARED.value, CandidateState.REVOKED.value},
    CandidateState.EPOCH_PREPARED.value: {CandidateState.COMMITTED.value, CandidateState.REVOKED.value},
    CandidateState.COMMITTED.value: {CandidateState.WITNESSED.value, CandidateState.REVOKED.value},
    CandidateState.WITNESSED.value: {CandidateState.ATTESTED.value, CandidateState.REVOKED.value},
    CandidateState.ATTESTED.value: {CandidateState.SERVABLE.value, CandidateState.REVOKED.value},
    CandidateState.SERVABLE.value: {CandidateState.ROLLED_BACK.value, CandidateState.RETIRED.value, CandidateState.REVOKED.value},
    CandidateState.ROLLED_BACK.value: {CandidateState.RETIRED.value},
    CandidateState.REJECTED.value: set(), CandidateState.REVOKED.value: {CandidateState.RETIRED.value}, CandidateState.RETIRED.value: set(),
}

PRIVILEGED_STATES = {
    CandidateState.QUALIFIED.value, CandidateState.AUTHORIZED.value, CandidateState.EPOCH_PREPARED.value,
    CandidateState.COMMITTED.value, CandidateState.WITNESSED.value, CandidateState.ATTESTED.value,
    CandidateState.SERVABLE.value, CandidateState.ROLLED_BACK.value, CandidateState.REVOKED.value, CandidateState.RETIRED.value,
}

_STAGE_KIND = {
    CandidateState.REGISTERED.value: "candidate_manifest",
    CandidateState.BUILT.value: "build_manifest",
    CandidateState.EVALUATED.value: "evaluation_bundle",
    CandidateState.QUALIFIED.value: "qualification_record",
    CandidateState.AUTHORIZED.value: "promotion_authorization",
}


class TransactionalCandidateStateStore:
    """Artifact-bound process-safe candidate lifecycle with signed authority transitions."""

    def __init__(self, root: str | Path, *, verifiers: Mapping[str, Any] | None = None,
                 artifacts: AuthorityArtifactStore | None = None, authority_registry: AuthorityStateRegistry | None = None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "candidate_state.sqlite3"; self.lock_path = self.root / "candidate_state.lock"
        self.verifiers = dict(verifiers or {}); self.artifacts = artifacts or AuthorityArtifactStore(self.root / "authority_artifacts")
        self.authority_registry = authority_registry
        self._init_db()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None); c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); c.execute("PRAGMA busy_timeout=30000"); return c

    def _init_db(self):
        with self._connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS candidate_state(candidate_digest TEXT PRIMARY KEY,state TEXT NOT NULL,last_event_digest TEXT NOT NULL,authority_generation INTEGER NOT NULL,policy_generation INTEGER NOT NULL,updated_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS candidate_events(seq INTEGER PRIMARY KEY AUTOINCREMENT,transition_id TEXT NOT NULL UNIQUE,nonce TEXT NOT NULL UNIQUE,event_digest TEXT NOT NULL UNIQUE,candidate_digest TEXT NOT NULL,previous_state TEXT,new_state TEXT NOT NULL,previous_event_digest TEXT,body_json TEXT NOT NULL,receipt_json TEXT,created_ns INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_candidate_events_candidate ON candidate_events(candidate_digest,seq);
            """)

    def snapshot(self, candidate_digest: str) -> dict[str, Any]:
        require_digest(candidate_digest, field_name="candidate_digest")
        with self._connect() as c: row = c.execute("SELECT * FROM candidate_state WHERE candidate_digest=?", (candidate_digest,)).fetchone()
        return {"state": None, "last_event_digest": None, "authority_generation": None, "policy_generation": None} if row is None else dict(row)

    @staticmethod
    def transition_body(*, candidate_digest: str, previous_state: str | None, new_state: str,
                        previous_event_digest: str | None, actor: str, authority_generation: int,
                        policy_generation: int, stage_artifact_digest: str = "", qualification_digest: str = "",
                        authorization_digest: str = "", authority_state_digest: str = "", reason: str = "",
                        transition_id: str | None = None, nonce: str | None = None) -> dict[str, Any]:
        require_digest(candidate_digest, field_name="candidate_digest")
        for n, v in (("stage_artifact_digest", stage_artifact_digest), ("qualification_digest", qualification_digest), ("authorization_digest", authorization_digest), ("authority_state_digest", authority_state_digest)):
            if v: require_digest(v, field_name=n)
        return {"schema": "egai-rc14-candidate-transition-v2", "transition_id": transition_id or "CT-" + uuid.uuid4().hex,
                "nonce": nonce or uuid.uuid4().hex, "candidate_digest": candidate_digest, "previous_state": previous_state,
                "new_state": str(new_state), "previous_event_digest": previous_event_digest, "actor": str(actor),
                "authority_generation": int(authority_generation), "policy_generation": int(policy_generation),
                "stage_artifact_digest": stage_artifact_digest, "qualification_digest": qualification_digest,
                "authorization_digest": authorization_digest, "authority_state_digest": authority_state_digest, "reason": str(reason)}

    def _last_body(self, c, candidate_digest: str) -> dict[str, Any] | None:
        row = c.execute("SELECT body_json FROM candidate_events WHERE candidate_digest=? ORDER BY seq DESC LIMIT 1", (candidate_digest,)).fetchone()
        return None if row is None else json.loads(row[0])

    def _validate_stage(self, c, *, candidate_digest: str, new_state: str, stage_digest: str, qualification_digest: str, authorization_digest: str, previous_stage_digest: str | None = None, bound_authority_state_digest: str | None = None) -> None:
        kind = _STAGE_KIND.get(new_state)
        if kind is None: return
        if not stage_digest: raise PermissionError(f"{new_state} requires a bound {kind} artifact")
        obj = self.artifacts.get(stage_digest, expected_kind=kind)
        if previous_stage_digest is None:
            last = self._last_body(c, candidate_digest)
            prev_stage = "" if last is None else str(last.get("stage_artifact_digest") or "")
        else:
            prev_stage = str(previous_stage_digest or "")
        if new_state == CandidateState.REGISTERED.value:
            if stage_digest != candidate_digest or not isinstance(obj, CandidateManifestRC14): raise PermissionError("REGISTERED must bind its CandidateManifest digest")
        elif new_state == CandidateState.BUILT.value:
            if not isinstance(obj, BuildManifestRC14) or obj.candidate_digest != candidate_digest: raise PermissionError("build manifest candidate binding mismatch")
        elif new_state == CandidateState.EVALUATED.value:
            if not isinstance(obj, EvaluationBundleRC14) or obj.candidate_digest != candidate_digest or obj.build_digest != prev_stage: raise PermissionError("evaluation bundle build/candidate binding mismatch")
        elif new_state == CandidateState.QUALIFIED.value:
            if not isinstance(obj, QualificationRecordRC14) or obj.candidate_digest != candidate_digest or obj.evaluation_digest != prev_stage or not obj.passed:
                raise PermissionError("qualification record is not a passing evaluation-bound record")
            manifest = self.artifacts.get(candidate_digest, expected_kind="candidate_manifest")
            if obj.proposal_digest != manifest.proposal_digest or obj.policy_digest != manifest.policy_digest:
                raise PermissionError("qualification proposal/policy binding mismatch")
            if not obj.qualification_evidence_digest:
                raise PermissionError("RC14.2 QUALIFIED transition requires derived qualification evidence")
            evidence = self.artifacts.get(obj.qualification_evidence_digest, expected_kind="qualification_evidence")
            if not isinstance(evidence, QualificationEvidenceRC14):
                raise PermissionError("qualification evidence artifact missing")
            if evidence.candidate_digest != candidate_digest or evidence.evaluation_digest != obj.evaluation_digest:
                raise PermissionError("qualification evidence candidate/evaluation mismatch")
            expected_flags = {
                "retention_passed": evidence.retention_passed,
                "negative_controls_passed": evidence.negative_controls_passed,
                "security_passed": evidence.security_passed,
                "independently_verified": evidence.independently_verified,
                "replicated": evidence.replicated,
                "independently_reproduced": evidence.independently_reproduced,
                "operator_approved": evidence.operator_approved,
            }
            for field, value in expected_flags.items():
                if bool(getattr(obj, field)) != bool(value):
                    raise PermissionError(f"qualification record {field} does not match derived evidence")
            if obj.fresh_ood_validated and not evidence.fresh_ood_validated:
                raise PermissionError("qualification record claims fresh OOD without derived evidence")
            if obj.continual_experiment_evidence_digest != evidence.continual_experiment_evidence_digest:
                raise PermissionError("qualification record continual-experiment evidence mismatch")
            if obj.continual_experiment_evidence_digest:
                experiment_evidence = self.artifacts.get(obj.continual_experiment_evidence_digest, expected_kind="continual_experiment_evidence")
                if not isinstance(experiment_evidence, ContinualExperimentEvidenceRC14) or not experiment_evidence.qualified:
                    raise PermissionError("continual-experiment evidence is not qualified")
                if experiment_evidence.candidate_digest != candidate_digest or experiment_evidence.proposal_digest != obj.proposal_digest:
                    raise PermissionError("continual-experiment evidence candidate/proposal mismatch")
            if qualification_digest and qualification_digest != stage_digest:
                raise PermissionError("qualification digest must equal bound qualification artifact")
        elif new_state == CandidateState.AUTHORIZED.value:
            if not isinstance(obj, PromotionAuthorizationRC14) or obj.candidate_digest != candidate_digest: raise PermissionError("promotion authorization candidate mismatch")
            if obj.qualification_digest != prev_stage or qualification_digest != prev_stage: raise PermissionError("promotion authorization qualification mismatch")
            if authorization_digest != stage_digest: raise PermissionError("authorization digest must equal bound promotion authorization")
            decision = self.artifacts.get(obj.promotion_decision_digest, expected_kind="promotion_decision")
            if not isinstance(decision, PromotionDecisionRC14) or not decision.approved or decision.candidate_digest != candidate_digest or decision.qualification_digest != prev_stage or decision.policy_digest != obj.policy_digest:
                raise PermissionError("promotion decision does not authorize this qualification")
            if self.authority_registry is not None:
                expected_authority = bound_authority_state_digest or (self.authority_registry.current().digest if self.authority_registry.current() is not None else "")
                if obj.authority_state_digest != expected_authority: raise PermissionError("promotion authorization is not bound to expected authority state")

    def transition(self, *, candidate_digest: str, new_state: str | CandidateState, actor: str, authority_generation: int,
                   policy_generation: int, stage_artifact_digest: str = "", qualification_digest: str = "",
                   authorization_digest: str = "", authority_state_digest: str = "", receipt: dict[str, Any] | None = None,
                   reason: str = "") -> dict[str, Any]:
        require_digest(candidate_digest, field_name="candidate_digest")
        new_state = new_state.value if isinstance(new_state, CandidateState) else str(new_state)
        if new_state not in {s.value for s in CandidateState}: raise ValueError("unknown candidate state")
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute("SELECT * FROM candidate_state WHERE candidate_digest=?", (candidate_digest,)).fetchone()
                previous = None if row is None else str(row["state"]); previous_event = None if row is None else str(row["last_event_digest"])
                if new_state not in LEGAL_TRANSITIONS.get(previous, set()): raise PermissionError(f"illegal candidate transition {previous!r} -> {new_state!r}")
                self._validate_stage(c, candidate_digest=candidate_digest, new_state=new_state, stage_digest=stage_artifact_digest, qualification_digest=qualification_digest, authorization_digest=authorization_digest, bound_authority_state_digest=authority_state_digest or None)
                privileged = new_state in PRIVILEGED_STATES
                if privileged:
                    if self.authority_registry is None: raise PermissionError("AuthorityStateRegistry required for privileged candidate transition")
                    if not authority_state_digest: raise PermissionError("authority_state_digest required for privileged transition")
                    verifier = self.verifiers.get(new_state)
                    if verifier is None: raise PermissionError(f"no trusted verifier configured for privileged state {new_state}")
                    if not isinstance(receipt, dict) or not isinstance(receipt.get("body"), dict): raise PermissionError("signed transition receipt required")
                    self.authority_registry.assert_current(authority_state_digest=authority_state_digest, authority_generation=authority_generation, policy_generation=policy_generation, receipt=receipt)
                    body = dict(receipt["body"])
                    expected = self.transition_body(candidate_digest=candidate_digest, previous_state=previous, new_state=new_state, previous_event_digest=previous_event,
                        actor=actor, authority_generation=authority_generation, policy_generation=policy_generation, stage_artifact_digest=stage_artifact_digest,
                        qualification_digest=qualification_digest, authorization_digest=authorization_digest, authority_state_digest=authority_state_digest,
                        reason=reason, transition_id=str(body.get("transition_id") or ""), nonce=str(body.get("nonce") or ""))
                    if not body.get("transition_id") or not body.get("nonce") or body != expected or not verifier.verify(receipt, expected_body=body): raise PermissionError("candidate transition receipt binding/signature mismatch")
                else:
                    body = self.transition_body(candidate_digest=candidate_digest, previous_state=previous, new_state=new_state, previous_event_digest=previous_event,
                        actor=actor, authority_generation=authority_generation, policy_generation=policy_generation, stage_artifact_digest=stage_artifact_digest,
                        qualification_digest=qualification_digest, authorization_digest=authorization_digest, authority_state_digest="", reason=reason)
                    receipt = None
                event_digest = sha256_json({"body": body, "receipt": receipt})
                try:
                    c.execute("INSERT INTO candidate_events(transition_id,nonce,event_digest,candidate_digest,previous_state,new_state,previous_event_digest,body_json,receipt_json,created_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (body["transition_id"], body["nonce"], event_digest, candidate_digest, previous, new_state, previous_event, json.dumps(body, sort_keys=True, separators=(",", ":")), None if receipt is None else json.dumps(receipt, sort_keys=True, separators=(",", ":")), time.time_ns()))
                    c.execute("INSERT INTO candidate_state(candidate_digest,state,last_event_digest,authority_generation,policy_generation,updated_ns) VALUES(?,?,?,?,?,?) ON CONFLICT(candidate_digest) DO UPDATE SET state=excluded.state,last_event_digest=excluded.last_event_digest,authority_generation=excluded.authority_generation,policy_generation=excluded.policy_generation,updated_ns=excluded.updated_ns",
                              (candidate_digest, new_state, event_digest, int(authority_generation), int(policy_generation), time.time_ns()))
                    c.execute("COMMIT")
                except sqlite3.IntegrityError as exc: raise RuntimeError("candidate transition replay/collision") from exc
                return {"candidate_digest": candidate_digest, "state": new_state, "event_digest": event_digest, "body": body}
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()


    def history(self, candidate_digest: str) -> list[dict[str, Any]]:
        """Return the immutable signed lifecycle history for one candidate."""
        require_digest(candidate_digest, field_name="candidate_digest")
        with self._connect() as c:
            rows = c.execute("SELECT seq,event_digest,body_json,receipt_json,created_ns FROM candidate_events WHERE candidate_digest=? ORDER BY seq", (candidate_digest,)).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event_digest": str(row["event_digest"]),
                "body": json.loads(row["body_json"]),
                "receipt": None if row["receipt_json"] is None else json.loads(row["receipt_json"]),
                "created_ns": int(row["created_ns"]),
            }
            for row in rows
        ]

    def stage_chain(self, candidate_digest: str) -> dict[str, str]:
        """Map lifecycle states to their bound immutable stage artifact digests."""
        out: dict[str, str] = {}
        for event in self.history(candidate_digest):
            body = event["body"]
            stage = str(body.get("stage_artifact_digest") or "")
            if stage:
                out[str(body["new_state"])] = stage
        return out

    def verify(self) -> dict[str, int]:
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok": raise RuntimeError("candidate state sqlite integrity failure")
            events = c.execute("SELECT * FROM candidate_events ORDER BY seq").fetchall(); heads = c.execute("SELECT * FROM candidate_state").fetchall()
        per: dict[str, tuple[str | None, str | None, int, int, str]] = {}
        for row in events:
            candidate = str(row["candidate_digest"]); prev_state, prev_event, prev_auth, prev_policy, prev_stage = per.get(candidate, (None, None, -1, -1, ""))
            body = json.loads(row["body_json"]); receipt = None if row["receipt_json"] is None else json.loads(row["receipt_json"])
            if body.get("candidate_digest") != candidate or body.get("previous_state") != prev_state or body.get("previous_event_digest") != prev_event: raise RuntimeError("candidate history discontinuity")
            if body.get("new_state") not in LEGAL_TRANSITIONS.get(prev_state, set()): raise RuntimeError("illegal candidate transition in stored history")
            expected_digest = sha256_json({"body": body, "receipt": receipt})
            if expected_digest != row["event_digest"]: raise RuntimeError("candidate event digest mismatch")
            # Revalidate immutable stage binding against history already reconstructed.
            with self._connect() as c2: self._validate_stage(c2, candidate_digest=candidate, new_state=body["new_state"], stage_digest=str(body.get("stage_artifact_digest") or ""), qualification_digest=str(body.get("qualification_digest") or ""), authorization_digest=str(body.get("authorization_digest") or ""), previous_stage_digest=prev_stage, bound_authority_state_digest=str(body.get("authority_state_digest") or "") or None)
            if body["new_state"] in PRIVILEGED_STATES:
                verifier = self.verifiers.get(body["new_state"])
                if verifier is None or receipt is None or not verifier.verify(receipt, expected_body=body): raise RuntimeError("stored privileged candidate transition is not trusted")
                if self.authority_registry is None: raise RuntimeError("authority registry missing during verification")
                self.authority_registry.assert_known(authority_state_digest=body["authority_state_digest"], authority_generation=body["authority_generation"], policy_generation=body["policy_generation"], receipt=receipt)
            if int(body.get("authority_generation", -1)) < prev_auth or int(body.get("policy_generation", -1)) < prev_policy: raise RuntimeError("candidate generation rollback in stored history")
            per[candidate] = (str(body["new_state"]), expected_digest, int(body["authority_generation"]), int(body["policy_generation"]), str(body.get("stage_artifact_digest") or ""))
        heads_by = {str(h["candidate_digest"]): h for h in heads}
        if set(heads_by) != set(per): raise RuntimeError("candidate head/event-history bijection failure")
        for candidate, current in per.items():
            head = heads_by[candidate]
            if current[0] != head["state"] or current[1] != head["last_event_digest"] or current[2] != head["authority_generation"] or current[3] != head["policy_generation"]: raise RuntimeError("candidate head does not match event history")
        return {"candidates": len(heads), "events": len(events)}
