from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import time
import uuid

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope


@dataclass(frozen=True)
class StateEpochV145:
    epoch_number: int
    predecessor_digest: str
    runtime_manifest_digest: str
    qualification_digest: str
    authorization_digest: str
    dependency_snapshot_digest: str
    created_at: float
    schema: str = "mini-agi-v14.1-alpha5-state-epoch-v1"
    def __post_init__(self):
        if self.epoch_number < 1: raise ValueError("epoch number must be >=1")
        if self.predecessor_digest: validate_digest(self.predecessor_digest)
        for d in (self.runtime_manifest_digest,self.qualification_digest,self.authorization_digest,self.dependency_snapshot_digest): validate_digest(d)
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class EpochTransitionReceiptV145:
    epoch_digest: str
    from_state: str
    to_state: str
    artifact_digest: str
    authority_id: str
    authority_generation: int
    nonce: str
    issued_at: float
    signer_key_id: str = ""
    signature_b64: str = ""
    schema: str = "mini-agi-v14.1-alpha5-epoch-transition-receipt-v1"
    def __post_init__(self):
        validate_digest(self.epoch_digest)
        if self.artifact_digest: validate_digest(self.artifact_digest)
        if not self.nonce: raise ValueError("transition nonce required")
    def unsigned(self): return replace(self,signer_key_id="",signature_b64="")
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class StateEpochLeaseV145:
    lease_id: str
    epoch_digest: str
    request_id: str
    acquired_at: float
    expires_at: float
    @property
    def digest(self): return digest(self)


class EpochTransitionAuthorityV145:
    def __init__(self, *, authority_id: str, authority_generation: int, signer):
        self.authority_id=str(authority_id); self.generation=int(authority_generation); self.signer=signer
    def issue(self, *, epoch_digest: str, from_state: str, to_state: str, artifact_digest: str):
        rec=EpochTransitionReceiptV145(validate_digest(epoch_digest),str(from_state),str(to_state),
            validate_digest(artifact_digest),self.authority_id,self.generation,uuid.uuid4().hex,time.time())
        env=self.signer.sign(asdict(rec.unsigned()))
        return replace(rec,signer_key_id=env.key_id,signature_b64=env.signature_b64)


class EpochTransitionValidatorV145:
    def __init__(self, *, verifier, trusted_key_ids, authority_generation: int):
        self.verifier=verifier; self.trusted={str(x) for x in trusted_key_ids}; self.generation=int(authority_generation)
    def validate(self, rec: EpochTransitionReceiptV145, *, epoch_digest: str, from_state: str, to_state: str):
        if rec.signer_key_id not in self.trusted: raise PermissionError("untrusted StateEpoch authority")
        if rec.authority_generation != self.generation: raise PermissionError("StateEpoch authority generation mismatch")
        if (rec.epoch_digest,rec.from_state,rec.to_state)!=(epoch_digest,from_state,to_state): raise PermissionError("StateEpoch transition binding mismatch")
        if not self.verifier.verify(asdict(rec.unsigned()),SignedEnvelope(rec.signer_key_id,rec.signature_b64)): raise PermissionError("invalid StateEpoch transition signature")
        return True


class StateEpochRegistryV145:
    TRANSITIONS={
        "LOCALLY_COMMITTED":{"EXTERNALLY_WITNESSED"},
        "EXTERNALLY_WITNESSED":{"ATTESTED"},
        "ATTESTED":{"SERVABLE"},
        "SERVABLE":{"RETIRED"},
        "RETIRED":set(),
    }
    def __init__(self, *, governance_db, cas, transition_validator: EpochTransitionValidatorV145):
        self.db=governance_db; self.cas=cas; self.validator=transition_validator

    def prepare(self, *, runtime_manifest_digest: str, qualification_digest: str, authorization_digest: str):
        for d in (runtime_manifest_digest,qualification_digest,authorization_digest): validate_digest(d); self.cas.get_bytes(d)
        predecessor=self.db.serving_epoch_v145() or ""
        epoch=StateEpochV145(self.db.next_epoch_number_v145(),predecessor,runtime_manifest_digest,qualification_digest,
                             authorization_digest,self.db.dependency_snapshot_digest_v145(),time.time())
        ed=self.cas.put_json(asdict(epoch))
        if ed != epoch.digest: raise RuntimeError("StateEpoch canonical digest mismatch")
        with self.db.transaction() as db:
            db.execute("INSERT INTO state_epochs_v145(epoch_digest,epoch_number,predecessor_digest,runtime_manifest_digest,qualification_digest,authorization_digest,state,body_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (epoch.digest,epoch.epoch_number,predecessor or None,runtime_manifest_digest,qualification_digest,authorization_digest,"PREPARED",json.dumps(asdict(epoch),sort_keys=True,separators=(",",":")),epoch.created_at))
            self.db._audit(db,event_type="state-epoch.prepared.v145",object_digest=epoch.digest,actor="state-epoch-registry",payload={"runtime_manifest_digest":runtime_manifest_digest})
        return epoch

    def mark_local_commit(self, *, epoch: StateEpochV145, activation_digest: str):
        activation_digest=validate_digest(activation_digest); self.cas.get_bytes(activation_digest)
        row=self.db.state_epoch_row_v145(epoch.digest)
        if row is None or row["state"] != "PREPARED": raise PermissionError("StateEpoch is not prepared")
        if self.db.current_runtime_manifest() != epoch.runtime_manifest_digest: raise PermissionError("runtime head does not match StateEpoch manifest")
        a=self.db.conn.execute("SELECT activation_digest,authorization_digest,runtime_manifest_digest FROM activations WHERE activation_digest=?",(activation_digest,)).fetchone()
        if a is None or a["authorization_digest"] != epoch.authorization_digest or a["runtime_manifest_digest"] != epoch.runtime_manifest_digest:
            raise PermissionError("activation does not close over StateEpoch")
        td=digest({"schema":"mini-agi-v14.1-alpha5-local-commit-transition-v1","epoch_digest":epoch.digest,"activation_digest":activation_digest})
        with self.db.transaction() as db:
            cur=db.execute("UPDATE state_epochs_v145 SET state='LOCALLY_COMMITTED',activation_digest=? WHERE epoch_digest=? AND state='PREPARED'",(activation_digest,epoch.digest))
            if cur.rowcount != 1: raise RuntimeError("StateEpoch local-commit race")
            db.execute("INSERT INTO epoch_transitions_v145(transition_digest,epoch_digest,from_state,to_state,artifact_digest,body_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (td,epoch.digest,"PREPARED","LOCALLY_COMMITTED",activation_digest,json.dumps({"activation_digest":activation_digest},sort_keys=True),time.time()))
            self.db._audit(db,event_type="state-epoch.locally-committed.v145",object_digest=epoch.digest,actor="state-epoch-registry",payload={"activation_digest":activation_digest})
        return td

    def transition(self, *, epoch_digest: str, receipt: EpochTransitionReceiptV145):
        epoch_digest=validate_digest(epoch_digest); row=self.db.state_epoch_row_v145(epoch_digest)
        if row is None: raise KeyError(epoch_digest)
        current=str(row["state"]); target=str(receipt.to_state)
        if target not in self.TRANSITIONS.get(current,set()): raise PermissionError(f"illegal StateEpoch transition {current}->{target}")
        self.validator.validate(receipt,epoch_digest=epoch_digest,from_state=current,to_state=target)
        rd=self.cas.put_json(asdict(receipt))
        if rd != receipt.digest: raise RuntimeError("transition receipt canonical digest mismatch")
        with self.db.transaction() as db:
            cur=db.execute("UPDATE state_epochs_v145 SET state=?,witness_digest=CASE WHEN ?='EXTERNALLY_WITNESSED' THEN ? ELSE witness_digest END,attestation_digest=CASE WHEN ?='ATTESTED' THEN ? ELSE attestation_digest END WHERE epoch_digest=? AND state=?",
                (target,target,receipt.artifact_digest,target,receipt.artifact_digest,epoch_digest,current))
            if cur.rowcount != 1: raise RuntimeError("StateEpoch transition race")
            if target == "SERVABLE":
                db.execute("INSERT INTO serving_epoch_head_v145(singleton,epoch_digest,updated_at) VALUES(1,?,?) ON CONFLICT(singleton) DO UPDATE SET epoch_digest=excluded.epoch_digest,updated_at=excluded.updated_at",(epoch_digest,time.time()))
            db.execute("INSERT INTO epoch_transitions_v145(transition_digest,epoch_digest,from_state,to_state,artifact_digest,signer_key_id,body_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (receipt.digest,epoch_digest,current,target,receipt.artifact_digest,receipt.signer_key_id,json.dumps(asdict(receipt),sort_keys=True,separators=(",",":")),time.time()))
            self.db._audit(db,event_type=f"state-epoch.{target.lower()}.v145",object_digest=epoch_digest,actor=receipt.authority_id,payload={"receipt_digest":receipt.digest,"artifact_digest":receipt.artifact_digest})
        return receipt.digest

    def acquire_request_lease(self, *, request_id: str, ttl_seconds: float=300.0):
        epoch_digest=self.db.serving_epoch_v145()
        if not epoch_digest: raise RuntimeError("no SERVABLE StateEpoch")
        self.db.require_usable_v145(epoch_digest)
        row=self.db.state_epoch_row_v145(epoch_digest)
        if row is None or row["state"] != "SERVABLE": raise RuntimeError("serving head is not SERVABLE")
        now=time.time(); lease=StateEpochLeaseV145("EL-"+uuid.uuid4().hex,epoch_digest,str(request_id),now,now+float(ttl_seconds))
        with self.db.transaction() as db:
            db.execute("INSERT INTO epoch_leases_v145(lease_id,epoch_digest,request_id,acquired_at,expires_at) VALUES(?,?,?,?,?)",
                       (lease.lease_id,lease.epoch_digest,lease.request_id,lease.acquired_at,lease.expires_at))
        return lease

    def validate_request_lease(self, lease: StateEpochLeaseV145):
        row=self.db.conn.execute("SELECT * FROM epoch_leases_v145 WHERE lease_id=?",(lease.lease_id,)).fetchone()
        if row is None or row["released_at"] is not None: raise PermissionError("StateEpoch lease absent/released")
        if row["epoch_digest"] != lease.epoch_digest or row["request_id"] != lease.request_id: raise PermissionError("StateEpoch lease binding mismatch")
        if float(row["expires_at"]) < time.time(): raise PermissionError("StateEpoch lease expired")
        epoch=self.db.state_epoch_row_v145(lease.epoch_digest)
        if epoch is None or epoch["state"] not in {"SERVABLE","RETIRED"}: raise PermissionError("leased StateEpoch unavailable")
        self.db.require_usable_v145(lease.epoch_digest)
        return True

    def release_request_lease(self, lease_id: str):
        with self.db.transaction() as db:
            cur=db.execute("UPDATE epoch_leases_v145 SET released_at=? WHERE lease_id=? AND released_at IS NULL",(time.time(),str(lease_id)))
            if cur.rowcount != 1: raise PermissionError("StateEpoch lease already released/absent")
