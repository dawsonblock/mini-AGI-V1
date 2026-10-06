from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
import uuid
from typing import Any, Iterator, Mapping

from kvcontinual.execution.durability.file_lock import ProcessFileLock
from minagi.egai.canonical import atomic_write_json, sha256_json
from .authority_artifacts import AuthorityArtifactStore, ExternalWitnessRC14, RuntimeAttestationRC14, PromotionAuthorizationRC14, PromotionDecisionRC14, QualificationRecordRC14
from .authority_state import AuthorityStateRegistry
from .models import EpochState, ExecutionContext, require_digest


@dataclass(frozen=True)
class StateEpochRC14:
    """Immutable closure of state visible to one serving request generation."""
    epoch: int
    predecessor_epoch_digest: str | None
    foundation_model_digest: str
    tokenizer_digest: str
    evidence_root: str
    belief_snapshot_digest: str
    episodic_memory_root: str
    semantic_memory_root: str
    skill_graph_root: str
    procedure_root: str
    retrieval_policy_digest: str
    retrieval_index_digest: str
    adapter_set_digest: str
    isolated_neural_memory_root: str
    plasticity_policy_digest: str
    governance_policy_digest: str
    execution_manifest_digest: str
    runtime_config_digest: str
    qualification_bundle_digest: str
    promotion_decision_digest: str
    promotion_authorization_digest: str
    source_tree_digest: str
    dependency_lock_digest: str
    build_provenance_digest: str
    target_platform_policy_digest: str
    runtime_attestation_policy_digest: str
    created_at: str
    schema: str = "egai-state-epoch-v3"

    def validate(self):
        if self.schema != "egai-state-epoch-v3": raise ValueError("unsupported StateEpoch schema")
        if int(self.epoch) < 0: raise ValueError("epoch must be non-negative")
        if self.predecessor_epoch_digest is not None: require_digest(self.predecessor_epoch_digest, field_name="predecessor_epoch_digest")
        for name, value in asdict(self).items():
            if name in {"epoch", "predecessor_epoch_digest", "created_at", "schema"}: continue
            if name.endswith("_digest") or name.endswith("_root"): require_digest(str(value), field_name=name)
        if not self.created_at: raise ValueError("created_at required")

    @property
    def digest(self) -> str: self.validate(); return sha256_json(asdict(self))
    @property
    def runtime_manifest_digest(self) -> str: return self.execution_manifest_digest
    @property
    def skill_snapshot_digest(self) -> str: return self.skill_graph_root
    def to_dict(self): return {**asdict(self), "digest": self.digest}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]):
        doc = dict(raw); got = str(doc.pop("digest", "")); obj = cls(**doc); obj.validate()
        if got and got != obj.digest: raise RuntimeError("StateEpoch digest mismatch")
        return obj


class StateEpochRegistryRC14:
    REQUIRED_COMPONENTS = (
        "foundation_model_digest", "tokenizer_digest", "evidence_root", "belief_snapshot_digest", "episodic_memory_root",
        "semantic_memory_root", "skill_graph_root", "procedure_root", "retrieval_policy_digest", "retrieval_index_digest",
        "adapter_set_digest", "isolated_neural_memory_root", "plasticity_policy_digest", "governance_policy_digest",
        "execution_manifest_digest", "runtime_config_digest", "qualification_bundle_digest", "promotion_decision_digest",
        "promotion_authorization_digest", "source_tree_digest", "dependency_lock_digest", "build_provenance_digest",
        "target_platform_policy_digest", "runtime_attestation_policy_digest",
    )
    TRANSITIONS = {
        EpochState.PREPARED.value: {EpochState.AUTHORIZED.value}, EpochState.AUTHORIZED.value: {EpochState.LOCALLY_COMMITTED.value},
        EpochState.LOCALLY_COMMITTED.value: {EpochState.EXTERNALLY_WITNESSED.value}, EpochState.EXTERNALLY_WITNESSED.value: {EpochState.ATTESTED.value},
        EpochState.ATTESTED.value: {EpochState.SERVABLE.value}, EpochState.SERVABLE.value: {EpochState.RETIRED.value}, EpochState.RETIRED.value: set(),
    }

    def __init__(self, root: str | Path, *, verifiers: Mapping[str, Any], artifacts: AuthorityArtifactStore | None = None,
                 authority_registry: AuthorityStateRegistry | None = None, witness_verifiers: Mapping[str, Any] | None = None,
                 runtime_attestation_verifiers: Mapping[str, Any] | None = None, strict_serving_artifacts: bool = False):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.objects = self.root / "objects"; self.objects.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "epochs.sqlite3"; self.lock_path = self.root / "epochs.lock"; self.verifiers = dict(verifiers)
        self.artifacts = artifacts or AuthorityArtifactStore(self.root / "authority_artifacts"); self.authority_registry = authority_registry
        self.witness_verifiers = dict(witness_verifiers or {})
        self.runtime_attestation_verifiers = dict(runtime_attestation_verifiers or {})
        self.strict_serving_artifacts = bool(strict_serving_artifacts)
        self._init_db()

    def _connect(self):
        c = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None); c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA synchronous=FULL"); c.execute("PRAGMA foreign_keys=ON"); c.execute("PRAGMA busy_timeout=30000"); return c

    def _init_db(self):
        with self._connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS epochs(epoch_digest TEXT PRIMARY KEY,epoch INTEGER NOT NULL UNIQUE,predecessor_digest TEXT,manifest_json TEXT NOT NULL,state TEXT NOT NULL,authorization_digest TEXT,witness_digest TEXT,attestation_digest TEXT,created_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS control_state(singleton INTEGER PRIMARY KEY CHECK(singleton=1),committed_epoch_digest TEXT,serving_epoch_digest TEXT,generation INTEGER NOT NULL,transition_seq INTEGER NOT NULL,transition_digest TEXT);
            INSERT OR IGNORE INTO control_state(singleton,committed_epoch_digest,serving_epoch_digest,generation,transition_seq,transition_digest) VALUES(1,NULL,NULL,0,0,NULL);
            CREATE TABLE IF NOT EXISTS epoch_transitions(seq INTEGER PRIMARY KEY AUTOINCREMENT,transition_id TEXT NOT NULL UNIQUE,nonce TEXT NOT NULL UNIQUE,epoch_digest TEXT NOT NULL,from_state TEXT NOT NULL,to_state TEXT NOT NULL,previous_transition_digest TEXT,transition_digest TEXT NOT NULL UNIQUE,body_json TEXT NOT NULL,receipt_json TEXT NOT NULL,committed_ns INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS epoch_leases(lease_id TEXT PRIMARY KEY,epoch_digest TEXT NOT NULL,request_id TEXT NOT NULL,acquired_ns INTEGER NOT NULL,expires_ns INTEGER NOT NULL,released_ns INTEGER);
            CREATE INDEX IF NOT EXISTS idx_epoch_leases_live ON epoch_leases(epoch_digest,released_ns,expires_ns);
            """)

    def _object_path(self, digest): require_digest(digest, field_name="epoch_digest"); return self.objects / f"{digest[7:]}.json"
    def current_committed_digest(self):
        with self._connect() as c: return c.execute("SELECT committed_epoch_digest FROM control_state WHERE singleton=1").fetchone()[0]
    def current_serving_digest(self):
        with self._connect() as c: return c.execute("SELECT serving_epoch_digest FROM control_state WHERE singleton=1").fetchone()[0]
    def active(self):
        d = self.current_serving_digest(); return None if d is None else self.get(str(d))[0]
    def snapshot(self):
        with self._connect() as c: return dict(c.execute("SELECT * FROM control_state WHERE singleton=1").fetchone())

    def get(self, digest):
        require_digest(digest, field_name="epoch_digest")
        with self._connect() as c: row = c.execute("SELECT manifest_json,state FROM epochs WHERE epoch_digest=?", (digest,)).fetchone()
        if row is None: raise KeyError(digest)
        epoch = StateEpochRC14.from_dict(json.loads(row["manifest_json"]))
        if epoch.digest != digest: raise RuntimeError("epoch DB key does not match manifest digest")
        return epoch, EpochState(str(row["state"]))

    @staticmethod
    def _signed_artifact_digest(c: sqlite3.Connection, epoch_digest: str, to_state: str) -> str:
        row = c.execute(
            "SELECT body_json FROM epoch_transitions WHERE epoch_digest=? AND to_state=? ORDER BY seq DESC LIMIT 1",
            (epoch_digest, str(to_state)),
        ).fetchone()
        if row is None:
            return ""
        body = json.loads(row["body_json"])
        return str(body.get("artifact_digest") or "")

    def _validate_witness_artifact(self, epoch: StateEpochRC14, artifact_digest: str, *, authority_state_digest: str, transition_head_digest: str) -> ExternalWitnessRC14:
        obj = self.artifacts.get(artifact_digest, expected_kind="external_witness")
        if not isinstance(obj, ExternalWitnessRC14):
            raise PermissionError("invalid external witness artifact")
        if obj.epoch_digest != epoch.digest or obj.transition_head_digest != transition_head_digest:
            raise PermissionError("external witness does not bind committed epoch/transition head")
        if obj.authority_state_digest != authority_state_digest:
            raise PermissionError("external witness authority-state mismatch")
        verifier = self.witness_verifiers.get(obj.witness_id)
        if verifier is None or not verifier.verify(obj.receipt, expected_body=obj.signed_body):
            raise PermissionError("external witness signature is not independently trusted")
        return obj

    def _validate_runtime_attestation_artifact(self, epoch: StateEpochRC14, artifact_digest: str, *, authority_state_digest: str, witness_digest: str) -> RuntimeAttestationRC14:
        obj = self.artifacts.get(artifact_digest, expected_kind="runtime_attestation")
        if not isinstance(obj, RuntimeAttestationRC14):
            raise PermissionError("invalid runtime attestation artifact")
        expected = {
            "epoch_digest": epoch.digest,
            "witness_digest": witness_digest,
            "foundation_model_digest": epoch.foundation_model_digest,
            "tokenizer_digest": epoch.tokenizer_digest,
            "adapter_set_digest": epoch.adapter_set_digest,
            "isolated_neural_memory_root": epoch.isolated_neural_memory_root,
            "execution_manifest_digest": epoch.execution_manifest_digest,
            "runtime_config_digest": epoch.runtime_config_digest,
            "source_tree_digest": epoch.source_tree_digest,
            "dependency_lock_digest": epoch.dependency_lock_digest,
            "build_provenance_digest": epoch.build_provenance_digest,
            "target_platform_policy_digest": epoch.target_platform_policy_digest,
            "runtime_attestation_policy_digest": epoch.runtime_attestation_policy_digest,
            "authority_state_digest": authority_state_digest,
        }
        for field, value in expected.items():
            if getattr(obj, field) != value:
                raise PermissionError(f"runtime attestation {field} mismatch")
        verifier = self.runtime_attestation_verifiers.get(obj.attestor_id)
        if verifier is None or not verifier.verify(obj.receipt, expected_body=obj.signed_body):
            raise PermissionError("runtime attestation signature is not independently trusted")
        return obj

    def _validate_promotion_closure(self, components: Mapping[str, str], *, require_current_authority: bool = False) -> None:
        q = self.artifacts.get(str(components["qualification_bundle_digest"]), expected_kind="qualification_record")
        d = self.artifacts.get(str(components["promotion_decision_digest"]), expected_kind="promotion_decision")
        a = self.artifacts.get(str(components["promotion_authorization_digest"]), expected_kind="promotion_authorization")
        if not isinstance(q, QualificationRecordRC14) or not q.passed: raise PermissionError("StateEpoch requires a passing qualification record")
        if not isinstance(d, PromotionDecisionRC14) or not d.approved or d.qualification_digest != q.digest: raise PermissionError("StateEpoch promotion decision/qualification mismatch")
        if not isinstance(a, PromotionAuthorizationRC14) or a.qualification_digest != q.digest or a.promotion_decision_digest != d.digest: raise PermissionError("StateEpoch promotion authorization closure mismatch")
        if d.candidate_digest != q.candidate_digest or a.candidate_digest != q.candidate_digest:
            raise PermissionError("StateEpoch qualification/decision/authorization candidate mismatch")
        if q.policy_digest != components["governance_policy_digest"] or d.policy_digest != q.policy_digest or a.policy_digest != q.policy_digest: raise PermissionError("StateEpoch governance-policy closure mismatch")
        if self.authority_registry is not None:
            self.authority_registry.get(a.authority_state_digest)
            if require_current_authority:
                current = self.authority_registry.current()
                if current is None or a.authority_state_digest != current.digest: raise PermissionError("StateEpoch promotion authorization is not bound to current authority state")

    def prepare(self, components: Mapping[str, str], *, predecessor: str | None = None):
        missing = [k for k in self.REQUIRED_COMPONENTS if k not in components]
        if missing: raise ValueError("missing StateEpoch components: " + ", ".join(missing))
        extras = set(components) - set(self.REQUIRED_COMPONENTS)
        if extras: raise ValueError("unknown StateEpoch components: " + ", ".join(sorted(extras)))
        for key in self.REQUIRED_COMPONENTS: require_digest(str(components[key]), field_name=key)
        self._validate_promotion_closure(components, require_current_authority=True)
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                committed = c.execute("SELECT committed_epoch_digest FROM control_state WHERE singleton=1").fetchone()[0]
                if predecessor is None: predecessor = committed
                if predecessor != committed: raise RuntimeError("new epoch must descend from current committed epoch")
                number = int(c.execute("SELECT COALESCE(MAX(epoch),-1)+1 FROM epochs").fetchone()[0])
                epoch = StateEpochRC14(epoch=number, predecessor_epoch_digest=predecessor, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **{k: str(components[k]) for k in self.REQUIRED_COMPONENTS})
                path = self._object_path(epoch.digest); atomic_write_json(path, epoch.to_dict())
                c.execute("INSERT INTO epochs(epoch_digest,epoch,predecessor_digest,manifest_json,state,created_ns) VALUES(?,?,?,?,?,?)", (epoch.digest, epoch.epoch, predecessor, json.dumps(epoch.to_dict(), sort_keys=True, separators=(",", ":")), EpochState.PREPARED.value, time.time_ns()))
                c.execute("COMMIT"); return epoch
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()

    @staticmethod
    def transition_body(*, epoch_digest: str, from_state: str, to_state: str, previous_transition_digest: str | None,
                        authority_generation: int, policy_generation: int, authority_state_digest: str = "", artifact_digest: str = "",
                        previous_serving_digest: str | None = None, reason: str = "", transition_id: str | None = None, nonce: str | None = None):
        require_digest(epoch_digest, field_name="epoch_digest")
        for n, v in (("previous_transition_digest", previous_transition_digest), ("authority_state_digest", authority_state_digest), ("artifact_digest", artifact_digest), ("previous_serving_digest", previous_serving_digest)):
            if v: require_digest(str(v), field_name=n)
        return {"schema": "egai-rc14-state-epoch-transition-v2", "transition_id": transition_id or "ET-" + uuid.uuid4().hex, "nonce": nonce or uuid.uuid4().hex,
                "epoch_digest": epoch_digest, "from_state": str(from_state), "to_state": str(to_state), "previous_transition_digest": previous_transition_digest,
                "authority_generation": int(authority_generation), "policy_generation": int(policy_generation), "authority_state_digest": authority_state_digest,
                "artifact_digest": artifact_digest, "previous_serving_digest": previous_serving_digest, "reason": str(reason)}

    def _transition(self, digest: str, *, to_state: EpochState, receipt: dict[str, Any], artifact_digest: str = "", authority_generation: int,
                    policy_generation: int, authority_state_digest: str, reason: str = ""):
        require_digest(digest, field_name="epoch_digest")
        verifier = self.verifiers.get(to_state.value)
        if verifier is None: raise PermissionError(f"no trusted verifier configured for epoch transition {to_state.value}")
        if self.authority_registry is None: raise PermissionError("AuthorityStateRegistry required for StateEpoch transitions")
        if not isinstance(receipt, dict) or not isinstance(receipt.get("body"), dict): raise PermissionError("signed StateEpoch transition receipt required")
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute("SELECT * FROM epochs WHERE epoch_digest=?", (digest,)).fetchone()
                if row is None: raise KeyError(digest)
                current = str(row["state"])
                if to_state.value not in self.TRANSITIONS.get(current, set()): raise PermissionError(f"illegal StateEpoch transition {current} -> {to_state.value}")
                ctl = c.execute("SELECT * FROM control_state WHERE singleton=1").fetchone(); epoch = StateEpochRC14.from_dict(json.loads(row["manifest_json"]))
                if to_state is EpochState.AUTHORIZED:
                    if artifact_digest != epoch.promotion_authorization_digest: raise PermissionError("StateEpoch authorization must equal manifest promotion_authorization_digest")
                    auth = self.artifacts.get(artifact_digest, expected_kind="promotion_authorization")
                    if not isinstance(auth, PromotionAuthorizationRC14) or auth.qualification_digest != epoch.qualification_bundle_digest or auth.promotion_decision_digest != epoch.promotion_decision_digest or auth.policy_digest != epoch.governance_policy_digest or auth.authority_state_digest != authority_state_digest:
                        raise PermissionError("promotion authorization does not bind StateEpoch qualification/decision/policy/authority state")
                if to_state in {EpochState.EXTERNALLY_WITNESSED, EpochState.ATTESTED} and not artifact_digest: raise PermissionError("required witness/attestation artifact digest missing")
                if self.strict_serving_artifacts and to_state is EpochState.EXTERNALLY_WITNESSED:
                    self._validate_witness_artifact(epoch, artifact_digest, authority_state_digest=authority_state_digest, transition_head_digest=str(ctl["transition_digest"] or ""))
                if self.strict_serving_artifacts and to_state is EpochState.ATTESTED:
                    witness_digest = self._signed_artifact_digest(c, digest, EpochState.EXTERNALLY_WITNESSED.value)
                    if not witness_digest or str(row["witness_digest"] or "") != witness_digest:
                        raise PermissionError("runtime attestation requires an untampered signed external-witness transition")
                    self._validate_runtime_attestation_artifact(epoch, artifact_digest, authority_state_digest=authority_state_digest, witness_digest=witness_digest)
                if self.strict_serving_artifacts and to_state is EpochState.SERVABLE:
                    witness_digest = self._signed_artifact_digest(c, digest, EpochState.EXTERNALLY_WITNESSED.value)
                    attestation_digest = self._signed_artifact_digest(c, digest, EpochState.ATTESTED.value)
                    if not witness_digest or not attestation_digest:
                        raise PermissionError("SERVABLE requires signed witness and runtime-attestation transitions")
                    if str(row["witness_digest"] or "") != witness_digest or str(row["attestation_digest"] or "") != attestation_digest:
                        raise PermissionError("SERVABLE metadata does not match signed witness/attestation history")
                    w = self.artifacts.get(witness_digest, expected_kind="external_witness")
                    if not isinstance(w, ExternalWitnessRC14) or w.epoch_digest != epoch.digest or w.authority_state_digest != authority_state_digest:
                        raise PermissionError("SERVABLE external witness binding mismatch")
                    verifier_w = self.witness_verifiers.get(w.witness_id)
                    if verifier_w is None or not verifier_w.verify(w.receipt, expected_body=w.signed_body):
                        raise PermissionError("SERVABLE external witness is not independently trusted")
                    self._validate_runtime_attestation_artifact(epoch, attestation_digest, authority_state_digest=authority_state_digest, witness_digest=witness_digest)
                if to_state is EpochState.LOCALLY_COMMITTED and row["predecessor_digest"] != ctl["committed_epoch_digest"]: raise RuntimeError("epoch predecessor no longer matches committed head")
                if to_state is EpochState.SERVABLE and ctl["committed_epoch_digest"] != digest: raise RuntimeError("only current committed epoch may become SERVABLE")
                if to_state is EpochState.RETIRED:
                    if ctl["serving_epoch_digest"] == digest: raise PermissionError("current serving epoch cannot be retired")
                    live = int(c.execute("SELECT COUNT(*) FROM epoch_leases WHERE epoch_digest=? AND released_ns IS NULL AND expires_ns>=?", (digest, time.time_ns())).fetchone()[0])
                    if live: raise PermissionError("epoch has live request leases and cannot be retired")
                self.authority_registry.assert_current(authority_state_digest=authority_state_digest, authority_generation=authority_generation, policy_generation=policy_generation, receipt=receipt)
                body = dict(receipt["body"]); prev_serving = ctl["serving_epoch_digest"] if to_state in {EpochState.SERVABLE, EpochState.RETIRED} else body.get("previous_serving_digest")
                expected = self.transition_body(epoch_digest=digest, from_state=current, to_state=to_state.value, previous_transition_digest=ctl["transition_digest"], authority_generation=authority_generation,
                    policy_generation=policy_generation, authority_state_digest=authority_state_digest, artifact_digest=artifact_digest, previous_serving_digest=prev_serving, reason=reason,
                    transition_id=str(body.get("transition_id") or ""), nonce=str(body.get("nonce") or ""))
                if not body.get("transition_id") or not body.get("nonce") or body != expected or not verifier.verify(receipt, expected_body=body): raise PermissionError("StateEpoch transition receipt binding/signature mismatch")
                td = sha256_json({"body": body, "receipt": receipt})
                try:
                    c.execute("INSERT INTO epoch_transitions(transition_id,nonce,epoch_digest,from_state,to_state,previous_transition_digest,transition_digest,body_json,receipt_json,committed_ns) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (body["transition_id"], body["nonce"], digest, current, to_state.value, ctl["transition_digest"], td, json.dumps(body, sort_keys=True, separators=(",", ":")), json.dumps(receipt, sort_keys=True, separators=(",", ":")), time.time_ns()))
                    field_sql = ""; args: tuple[Any, ...] = ()
                    if to_state is EpochState.AUTHORIZED: field_sql = ", authorization_digest=?"; args = (artifact_digest,)
                    elif to_state is EpochState.EXTERNALLY_WITNESSED: field_sql = ", witness_digest=?"; args = (artifact_digest,)
                    elif to_state is EpochState.ATTESTED: field_sql = ", attestation_digest=?"; args = (artifact_digest,)
                    c.execute(f"UPDATE epochs SET state=?{field_sql} WHERE epoch_digest=?", (to_state.value, *args, digest))
                    generation = int(ctl["generation"]); seq = int(ctl["transition_seq"]) + 1; committed = ctl["committed_epoch_digest"]; serving = ctl["serving_epoch_digest"]
                    if to_state is EpochState.LOCALLY_COMMITTED: committed = digest; generation += 1
                    if to_state is EpochState.SERVABLE: serving = digest; generation += 1
                    c.execute("UPDATE control_state SET committed_epoch_digest=?,serving_epoch_digest=?,generation=?,transition_seq=?,transition_digest=? WHERE singleton=1", (committed, serving, generation, seq, td))
                    c.execute("COMMIT")
                except sqlite3.IntegrityError as exc: raise RuntimeError("StateEpoch transition replay/collision") from exc
                return {"epoch_digest": digest, "state": to_state.value, "transition_digest": td, "generation": generation, "transition_seq": seq}
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()

    def authorize(self, digest, *, authorization_digest, receipt, authority_generation, policy_generation, authority_state_digest, reason="qualified promotion"):
        return self._transition(digest, to_state=EpochState.AUTHORIZED, receipt=receipt, artifact_digest=authorization_digest, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)
    def local_commit(self, digest, *, receipt, authority_generation, policy_generation, authority_state_digest, reason="local commit"):
        return self._transition(digest, to_state=EpochState.LOCALLY_COMMITTED, receipt=receipt, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)
    def witness(self, digest, *, witness_digest, receipt, authority_generation, policy_generation, authority_state_digest, reason="external witness"):
        return self._transition(digest, to_state=EpochState.EXTERNALLY_WITNESSED, receipt=receipt, artifact_digest=witness_digest, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)
    def attest(self, digest, *, attestation_digest, receipt, authority_generation, policy_generation, authority_state_digest, reason="runtime attestation"):
        return self._transition(digest, to_state=EpochState.ATTESTED, receipt=receipt, artifact_digest=attestation_digest, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)
    def make_servable(self, digest, *, receipt, authority_generation, policy_generation, authority_state_digest, reason="servable activation"):
        return self._transition(digest, to_state=EpochState.SERVABLE, receipt=receipt, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)
    def retire(self, digest, *, receipt, authority_generation, policy_generation, authority_state_digest, reason="retire superseded epoch"):
        return self._transition(digest, to_state=EpochState.RETIRED, receipt=receipt, authority_generation=authority_generation, policy_generation=policy_generation, authority_state_digest=authority_state_digest, reason=reason)

    @contextmanager
    def pin(self, request_id: str, *, ttl_seconds: int = 3600) -> Iterator[ExecutionContext]:
        if not request_id: raise ValueError("request_id required")
        lease_id = "EL-" + uuid.uuid4().hex; now = time.time_ns(); expires = now + int(ttl_seconds) * 1_000_000_000
        with ProcessFileLock(self.lock_path, timeout=30.0):
            c = self._connect(); c.execute("BEGIN IMMEDIATE")
            try:
                digest = c.execute("SELECT serving_epoch_digest FROM control_state WHERE singleton=1").fetchone()[0]
                if digest is None: raise RuntimeError("no SERVABLE StateEpoch")
                row = c.execute("SELECT epoch,state FROM epochs WHERE epoch_digest=?", (digest,)).fetchone()
                if row is None or row["state"] != EpochState.SERVABLE.value: raise RuntimeError("serving pointer is not SERVABLE")
                c.execute("INSERT INTO epoch_leases VALUES(?,?,?,?,?,NULL)", (lease_id, digest, request_id, now, expires)); c.execute("COMMIT")
                ctx = ExecutionContext(request_id, lease_id, str(digest), int(row["epoch"]), now, expires)
            except Exception:
                if c.in_transaction: c.execute("ROLLBACK")
                raise
            finally: c.close()
        try: yield ctx
        finally:
            with ProcessFileLock(self.lock_path, timeout=30.0):
                with self._connect() as c:
                    c.execute("BEGIN IMMEDIATE"); c.execute("UPDATE epoch_leases SET released_ns=? WHERE lease_id=? AND released_ns IS NULL", (time.time_ns(), lease_id)); c.execute("COMMIT")

    def validate_context(self, context: ExecutionContext) -> StateEpochRC14:
        with self._connect() as c:
            row = c.execute("SELECT l.*,e.epoch,e.state FROM epoch_leases l JOIN epochs e ON e.epoch_digest=l.epoch_digest WHERE lease_id=?", (context.lease_id,)).fetchone()
        if row is None or row["request_id"] != context.request_id or row["epoch_digest"] != context.epoch_digest or int(row["epoch"]) != context.epoch_generation: raise PermissionError("execution context lease binding mismatch")
        if row["released_ns"] is not None or int(row["expires_ns"]) < time.time_ns(): raise PermissionError("execution context lease is not live")
        if row["state"] not in {EpochState.SERVABLE.value}: raise PermissionError("execution context epoch is not servable")
        return self.get(context.epoch_digest)[0]

    def prepare_rollback(self, target_epoch_digest: str, *, overrides: Mapping[str, str] | None = None):
        target, _ = self.get(target_epoch_digest); data = asdict(target)
        for key in ("epoch", "predecessor_epoch_digest", "created_at", "schema"): data.pop(key, None)
        if overrides: data.update({str(k): str(v) for k, v in overrides.items()})
        return self.prepare(data)

    def verify(self):
        with self._connect() as c:
            if str(c.execute("PRAGMA integrity_check").fetchone()[0]) != "ok": raise RuntimeError("StateEpoch sqlite integrity failure")
            epochs = c.execute("SELECT * FROM epochs ORDER BY epoch").fetchall(); transitions = c.execute("SELECT * FROM epoch_transitions ORDER BY seq").fetchall(); ctl = dict(c.execute("SELECT * FROM control_state WHERE singleton=1").fetchone())
            live_leases = int(c.execute("SELECT COUNT(*) FROM epoch_leases WHERE released_ns IS NULL AND expires_ns>=?", (time.time_ns(),)).fetchone()[0])
        by_digest: dict[str, sqlite3.Row] = {}; reconstructed: dict[str, dict[str, Any]] = {}
        seen_epochs: set[str] = set()
        for i, row in enumerate(epochs):
            epoch = StateEpochRC14.from_dict(json.loads(row["manifest_json"]))
            if epoch.epoch != i: raise RuntimeError("StateEpoch numbering discontinuity")
            if epoch.predecessor_epoch_digest is not None and epoch.predecessor_epoch_digest not in seen_epochs: raise RuntimeError("StateEpoch predecessor references missing/later epoch")
            if epoch.digest != row["epoch_digest"] or row["predecessor_digest"] != epoch.predecessor_epoch_digest: raise RuntimeError("stored StateEpoch manifest/key mismatch")
            path = self._object_path(epoch.digest)
            if not path.exists() or StateEpochRC14.from_dict(json.loads(path.read_text(encoding="utf-8"))).digest != epoch.digest: raise RuntimeError("StateEpoch CAS object missing or invalid")
            self._validate_promotion_closure(asdict(epoch))
            by_digest[epoch.digest] = row; reconstructed[epoch.digest] = {"state": EpochState.PREPARED.value, "authorization_digest": None, "witness_digest": None, "attestation_digest": None}; seen_epochs.add(epoch.digest)
        prev_td = None; generation = 0; committed = None; serving = None
        for row in transitions:
            body = json.loads(row["body_json"]); receipt = json.loads(row["receipt_json"]); d = str(row["epoch_digest"])
            if d not in reconstructed: raise RuntimeError("StateEpoch transition references missing epoch")
            if body.get("epoch_digest") != d or body.get("from_state") != row["from_state"] or body.get("to_state") != row["to_state"] or body.get("previous_transition_digest") != row["previous_transition_digest"]: raise RuntimeError("StateEpoch transition row/body mismatch")
            if row["previous_transition_digest"] != prev_td: raise RuntimeError("StateEpoch global transition-chain discontinuity")
            if reconstructed[d]["state"] != row["from_state"] or row["to_state"] not in self.TRANSITIONS.get(row["from_state"], set()): raise RuntimeError("StateEpoch reconstructed lifecycle discontinuity")
            if sha256_json({"body": body, "receipt": receipt}) != row["transition_digest"]: raise RuntimeError("StateEpoch transition digest mismatch")
            verifier = self.verifiers.get(str(row["to_state"]))
            if verifier is None or not verifier.verify(receipt, expected_body=body): raise RuntimeError("stored StateEpoch transition signature invalid")
            if self.authority_registry is None: raise RuntimeError("authority registry missing during StateEpoch verification")
            self.authority_registry.assert_known(authority_state_digest=body["authority_state_digest"], authority_generation=body["authority_generation"], policy_generation=body["policy_generation"], receipt=receipt)
            epoch = StateEpochRC14.from_dict(json.loads(by_digest[d]["manifest_json"])); artifact = str(body.get("artifact_digest") or "")
            if row["to_state"] == EpochState.AUTHORIZED.value:
                if artifact != epoch.promotion_authorization_digest: raise RuntimeError("stored StateEpoch promotion authorization mismatch")
                reconstructed[d]["authorization_digest"] = artifact
            elif row["to_state"] == EpochState.EXTERNALLY_WITNESSED.value:
                if self.strict_serving_artifacts:
                    self._validate_witness_artifact(epoch, artifact, authority_state_digest=body["authority_state_digest"], transition_head_digest=str(row["previous_transition_digest"] or ""))
                reconstructed[d]["witness_digest"] = artifact
            elif row["to_state"] == EpochState.ATTESTED.value:
                if self.strict_serving_artifacts:
                    witness_digest = str(reconstructed[d]["witness_digest"] or "")
                    if not witness_digest:
                        raise RuntimeError("runtime attestation precedes validated witness")
                    self._validate_runtime_attestation_artifact(epoch, artifact, authority_state_digest=body["authority_state_digest"], witness_digest=witness_digest)
                reconstructed[d]["attestation_digest"] = artifact
            elif row["to_state"] == EpochState.LOCALLY_COMMITTED.value: committed = d; generation += 1
            elif row["to_state"] == EpochState.SERVABLE.value: serving = d; generation += 1
            elif row["to_state"] == EpochState.RETIRED.value and serving == d: raise RuntimeError("transition history retires active serving epoch")
            reconstructed[d]["state"] = str(row["to_state"]); prev_td = str(row["transition_digest"])
        if ctl["transition_digest"] != prev_td or int(ctl["transition_seq"]) != len(transitions): raise RuntimeError("control transition head/sequence mismatch")
        if ctl["committed_epoch_digest"] != committed or ctl["serving_epoch_digest"] != serving or int(ctl["generation"]) != generation: raise RuntimeError("control state does not reconstruct from signed transition history")
        for d, row in by_digest.items():
            r = reconstructed[d]
            for field in ("state", "authorization_digest", "witness_digest", "attestation_digest"):
                if row[field] != r[field]: raise RuntimeError(f"StateEpoch stored {field} does not match signed transition history")
        if serving is not None and reconstructed[serving]["state"] != EpochState.SERVABLE.value: raise RuntimeError("serving pointer does not reference SERVABLE epoch")
        return {"epochs": len(epochs), "transitions": len(transitions), "committed_epoch_digest": committed, "serving_epoch_digest": serving, "generation": generation, "live_leases": live_leases}
