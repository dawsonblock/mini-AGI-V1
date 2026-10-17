"""v4.0.3 Phase 2 — authenticated registry state machine.

Required sequences (production authorization must survive legitimate
restarts while rejecting stale, missing or tampered authority state):

    Register -> Revoke -> Restart                 -> Blocked
    Register -> Revoke -> Delete ledger           -> Refused
    Register -> Revoke -> Load old snapshot       -> Refused
    Register -> Revoke -> Requalify -> Restart    -> Active
    Register -> Suspend -> Restart                -> Blocked
    Register -> Requalify with invalid signature  -> Blocked

The authority split under test: REGISTRY.json is the signed mutable
snapshot, EVENTS.jsonl is the signed hash-chained transition ledger,
and HEAD.json is the durable state authority held under separate
custody. Production loads use ``require_authority=True``.
"""
import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (  # noqa: E402
    AuthorityRecord, AuthorityRegistry)
from minagi.v161.authenticated_registry import (  # noqa: E402
    GENESIS_DIGEST, RegistryEventLog, RegistryStateAuthority)
from minagi.v161.verified_routing import (  # noqa: E402
    QualifiedAdapterManifest, VerifiedAdapterRouter)

DIG = "sha256:" + "a" * 64
DIG2 = "sha256:" + "b" * 64
DIG3 = "sha256:" + "c" * 64


def _b64key(s: Ed25519Signer) -> str:
    return base64.b64encode(s.public_bytes()).decode()


def _manifest(aid: str, signer: Ed25519Signer, *, gen: int = 1,
              evidence: str = DIG2,
              family_terms=("add", "math")) -> QualifiedAdapterManifest:
    return QualifiedAdapterManifest(
        adapter_id=aid, adapter_digest=DIG, model_id="model/m1",
        model_revision="rev-1", qualified_families=("math",),
        qualification_evidence_digest=evidence,
        scorer_artifact_digest=DIG3, family_terms=family_terms,
        priority=0.9, qualification_generation=gen).sign(signer)


class _World:
    """One governed registry deployment: manifest authority, registry
    signer, event ledger, and a separately-custodied state authority."""

    def __init__(self, tmp_path: Path):
        d = tmp_path / "registry"
        d.mkdir()
        self.reg_path = d / "REGISTRY.json"
        self.ev_path = d / "EVENTS.jsonl"
        # the authority head lives OUTSIDE the registry directory —
        # same-attacker-controlled storage is not the boundary
        self.auth_path = tmp_path / "authority" / "HEAD.json"
        self.qual_signer = Ed25519Signer.generate()   # manifests
        self.state_signer = Ed25519Signer.generate()  # state + events
        self.authority = AuthorityRegistry([
            AuthorityRecord(role="qualification",
                            key_id=self.qual_signer.key_id,
                            public_key_b64=_b64key(self.qual_signer)),
            AuthorityRecord(role="revocation",
                            key_id=self.state_signer.key_id,
                            public_key_b64=_b64key(self.state_signer))])
        self.verifier = self.authority.verifier()

    def open_router(self, **kw) -> VerifiedAdapterRouter:
        return VerifiedAdapterRouter(
            authority=self.authority, required_role="qualification",
            state_signer=self.state_signer,
            event_log=self.ev_path, state_authority=self.auth_path,
            health_window=4, min_health=0.5, **kw)

    def save(self, r: VerifiedAdapterRouter):
        self.reg_path.write_text(r.to_json())

    def load(self, **kw) -> VerifiedAdapterRouter:
        kw.setdefault("registry_verifier", self.verifier)
        kw.setdefault("event_log", self.ev_path)
        kw.setdefault("state_authority", self.auth_path)
        kw.setdefault("require_authority", True)
        kw.setdefault("authority", self.authority)
        kw.setdefault("required_role", "qualification")
        kw.setdefault("state_signer", self.state_signer)
        kw.setdefault("health_window", 4)
        kw.setdefault("min_health", 0.5)
        return VerifiedAdapterRouter.from_json(self.reg_path, **kw)


def _route(r: VerifiedAdapterRouter):
    return r.route("add two math numbers", family="math",
                   family_authenticated=True, model_revision="rev-1")


# ---------- the six required sequences --------------------------------

def test_register_revoke_restart_blocked(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer))
    w.save(r)
    assert _route(w.load()).selected == "a1"

    r2 = w.load()
    r2.revoke("a1")
    w.save(r2)
    d = _route(w.load())
    assert d.selected is None
    assert any("revoked" in x for x in d.refused)


def test_register_revoke_delete_ledger_refused(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer))
    r.revoke("a1")
    w.save(r)

    w.ev_path.unlink()
    with pytest.raises(ValueError, match="ledger"):
        w.load()
    # lenient load WITHOUT the authority configured keeps the adapter
    # revoked via the signed snapshot — a missing ledger only makes
    # the authorization state unprovable, it cannot resurrect access
    r = w.load(require_authority=False, state_authority=None)
    assert "a1" in r._revoked
    assert any("ledger" in x for x in r.registry_warnings)


def test_register_revoke_load_old_snapshot_refused(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer))
    w.save(r)
    stale = tmp_path / "STALE_REGISTRY.json"
    stale.write_text(w.reg_path.read_text())   # signed, seq=N, no revoke

    r.revoke("a1")
    w.save(r)                                   # head now attests seq>N

    # the current file still loads — the authority attests it
    assert "a1" in w.load()._revoked

    # the old snapshot is still a VALIDLY SIGNED document — the durable
    # authority is what proves it is not the newest accepted state
    old_kw = dict(registry_verifier=w.verifier,
                  event_log=w.ev_path, state_authority=w.auth_path,
                  require_authority=True, authority=w.authority)
    with pytest.raises(ValueError, match="authority head|seq"):
        VerifiedAdapterRouter.from_json(stale, **old_kw)
    # and under the stale snapshot alone — signed, internally
    # consistent — the adapter would look LIVE; the separately-held
    # authority head is the difference between rollback and refusal
    relaxed = VerifiedAdapterRouter.from_json(
        stale, registry_verifier=w.verifier,
        authority=w.authority, health_window=4, min_health=0.5)
    assert _route(relaxed).selected == "a1"


def test_register_revoke_requalify_restart_active(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer, gen=1, evidence=DIG2))
    r.revoke("a1")
    w.save(r)
    assert _route(w.load()).selected is None

    # the independent qualifier authorizes generation 2 with fresh
    # evidence — reinstatement is a NEW qualification generation
    r2 = w.load()
    r2.reinstate(_manifest("a1", w.qual_signer, gen=2, evidence=DIG3))
    w.save(r2)

    r3 = w.load()
    d = _route(r3)
    assert d.selected == "a1"
    assert r3._generations["a1"] == 2
    # and the OLD generation manifest cannot activate anymore
    r3.register(_manifest("a1", w.qual_signer, gen=1, evidence=DIG2))
    assert "outdated" in r3.ineligible()["a1"]


def test_register_suspend_restart_blocked(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer))
    w.save(r)
    r2 = w.load()
    for _ in range(4):
        r2.record_outcome("a1", False)          # health suspension
    w.save(r2)
    d = _route(w.load())
    assert d.selected is None
    assert any("suspended" in x for x in d.refused)

    # a signed ADAPTER_RESTORED event lifts the suspension —
    # restoration from suspension is not a requalification
    r3 = w.load()
    assert r3.restore("a1")
    w.save(r3)
    assert _route(w.load()).selected == "a1"


def test_requalify_with_invalid_signature_blocked(tmp_path):
    w = _World(tmp_path)
    r = w.open_router()
    r.register(_manifest("a1", w.qual_signer))
    r.revoke("a1")
    w.save(r)

    # forge a REQUALIFIED event with a key the authority doesn't know
    forger = Ed25519Signer.generate()
    forged = {"schema": "mini-agi-v4.0-registry-event-v1",
              "seq": 2, "at": 1, "event": "ADAPTER_REQUALIFIED",
              "adapter_id": "a1", "qualification_generation": 2,
              "artifact_digest": DIG, "model_revision": "rev-1",
              "evidence_digest": DIG3,
              "prev_event_digest":
                  RegistryEventLog(w.ev_path).last_digest(),
              "signer_key_id": forger.key_id}
    forged["signature_b64"] = forger.sign(forged).signature_b64
    forged["event_digest"] = digest(forged)
    with w.ev_path.open("a") as fh:
        fh.write(json.dumps(forged, sort_keys=True) + "\n")

    # the durable authority refuses first (head digest changed), and
    # even without it the event fold itself rejects the forgery
    with pytest.raises(ValueError, match="ledger|event chain"):
        w.load()
    r2 = w.load(require_authority=False, state_authority=None)
    assert any("signature invalid" in x
               for x in r2.registry_warnings)
    assert _route(r2).selected is None


# ---------- ledger / authority units -----------------------------------

def test_event_chain_detects_truncation_and_reorder(tmp_path):
    s = Ed25519Signer.generate()
    log = RegistryEventLog(tmp_path / "e.jsonl")
    log.append(s, "ADAPTER_REGISTERED", "a1", qualification_generation=1)
    log.append(s, "ADAPTER_REVOKED", "a1")
    from egai.common.crypto import Ed25519Verifier
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    _, failures = log.verify(v)
    assert failures == []

    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join(lines[:1]) + "\n")   # drop REVOKED
    _, failures = log.verify(v)
    # truncation is structurally consistent — this is why the state
    # authority binds the head digest; the log itself stays clean here
    assert failures == []
    log.append(s, "ADAPTER_REGISTERED", "a2", qualification_generation=1)
    _, failures = log.verify(v)
    assert failures == []
    # re-append after truncation: seq restarts at 1 — detectable only
    # against the authority head. Reorder instead:
    log.path.write_text("\n".join(reversed(
        log.path.read_text().splitlines())) + "\n")
    _, failures = log.verify(v)
    assert any("seq" in f or "prev_event_digest" in f for f in failures)


def test_state_authority_is_monotonic(tmp_path):
    s = Ed25519Signer.generate()
    auth = RegistryStateAuthority(tmp_path / "HEAD.json")
    auth.record_head(s, seq=3, state_digest=DIG)
    with pytest.raises(ValueError, match="monotonic"):
        auth.record_head(s, seq=3, state_digest=DIG2)
    with pytest.raises(ValueError, match="monotonic"):
        auth.record_head(s, seq=1, state_digest=DIG)
    auth.record_head(s, seq=4, state_digest=DIG2)
    from egai.common.crypto import Ed25519Verifier
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    assert auth.check(4, DIG2, verifier=v) == ""
    assert "seq" in auth.check(3, DIG2, verifier=v)
    assert "state digest" in auth.check(4, DIG, verifier=v)
    # missing authority is a failure, not a pass
    gone = RegistryStateAuthority(tmp_path / "GONE.json")
    assert gone.check(4, DIG2, verifier=v) == \
        "registry state authority unavailable"


def test_events_carry_binding_fields(tmp_path):
    s = Ed25519Signer.generate()
    log = RegistryEventLog(tmp_path / "e.jsonl")
    e = log.append(s, "ADAPTER_QUALIFIED", "a1",
                   qualification_generation=2, artifact_digest=DIG,
                   model_revision="rev-7", evidence_digest=DIG2)
    assert e["artifact_digest"] == DIG
    assert e["model_revision"] == "rev-7"
    assert e["evidence_digest"] == DIG2
    assert e["prev_event_digest"] == GENESIS_DIGEST
    assert e["seq"] == 0 and e["signer_key_id"] == s.key_id
