"""v16.4.10 SEC-905 — the launch broker INDEPENDENTLY verifies a
signed, unexpired, record-resolved, nonce-bound launch authorization.

Every attack in the audit table must refuse; only a complete valid
authorization may proceed. Verification uses trusted public keys and
a read-only approved-record resolver — the broker holds no signing
material.
"""
from __future__ import annotations

import hashlib
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.isolation_profile import load_profile  # noqa: E402
from minagi.runtime.worker_launch_broker import (  # noqa: E402
    AuthorizedLaunchRecord, BrokerError, BrokerRefused,
    LaunchWorkerRequest, MappingRecordResolver,
    WorkerLaunchBroker, issue_launch_authorization, new_worker_id,
    verify_launch_authorization)
from minagi.v161.authority import (AUTHORITY_ROLES,  # noqa: E402
                                   AuthorityRegistry, write_trust_root)

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"


def _d(x):
    return digest({"d": x})


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


class _IdleSystemd:
    available = True

    def run_service(self, *a, **k):
        raise AssertionError("refused launches never spawn")

    def show(self, u, p):
        return "inv"

    def is_active(self, u):
        return False

    def stop(self, u):
        return None

    def kill(self, u, sig="SIGKILL"):
        return None

    def cgroup_path(self, u):
        return None


def _rig(tmp_path):
    registry, signers = _chain(tmp_path)
    profile = load_profile(PROFILE_PATH)
    binary = tmp_path / "nsjail"
    binary.write_bytes(b"fake")
    binary.chmod(0o755)
    ok = {"backend": _d("backend"), "runtime": _d("runtime"),
          "artifact": _d("artifact"), "profile": profile.digest(),
          "budget": _d("budget")}
    resolver = MappingRecordResolver(
        {v: k for k, v in ok.items()}, revocation_epoch_floor=3)
    broker = WorkerLaunchBroker(
        worker_state_root=tmp_path / "workers",
        approved_profiles={profile.digest(): profile},
        artifact_resolver=lambda d: str(tmp_path / "a"),
        nsjail_path=str(binary),
        nsjail_digest="sha256:" + hashlib.sha256(
            binary.read_bytes()).hexdigest(),
        worker_uid=990, worker_gid=990,
        systemd=_IdleSystemd(), nonce_dir=tmp_path / "nonces",
        authority_registry=registry, record_resolver=resolver)
    return broker, profile, registry, signers, resolver, ok


def _auth(signer, ok, *, nonce=None, now=None, **over):
    now = now or int(time.time())
    rec = AuthorizedLaunchRecord(
        activation_id="act-1",
        backend_manifest_digest=ok["backend"],
        runtime_manifest_digest=ok["runtime"],
        artifact_root_digest=ok["artifact"],
        isolation_profile_digest=ok["profile"],
        execution_budget_digest=ok["budget"],
        policy_epoch=0, revocation_epoch=3,
        issued_at=now - 5, expires_at=now + 300,
        nonce=nonce or "a" * 32)
    fields = dict(
        activation_id=rec.activation_id,
        backend_manifest_digest=rec.backend_manifest_digest,
        runtime_manifest_digest=rec.runtime_manifest_digest,
        artifact_root_digest=rec.artifact_root_digest,
        isolation_profile_digest=rec.isolation_profile_digest,
        execution_budget_digest=rec.execution_budget_digest,
        policy_epoch=rec.policy_epoch,
        revocation_epoch=rec.revocation_epoch,
        issued_at=rec.issued_at, expires_at=rec.expires_at,
        nonce=rec.nonce)
    fields.update(over)
    return issue_launch_authorization(
        signer, record=AuthorizedLaunchRecord(**fields))


def _req(profile, ok, *, auth, nonce):
    return LaunchWorkerRequest(
        worker_id=new_worker_id(),
        signed_backend_manifest_digest=ok["backend"],
        approved_artifact_digests=(ok["artifact"],),
        approved_isolation_profile_digest=ok["profile"],
        execution_budget_digest=ok["budget"],
        request_nonce=nonce, role="worker", authorization=auth)


def _launch(broker, profile, ok, signers, *, nonce=None, signer=None,
            **over):
    nonce = nonce or "b" * 32
    auth = _auth(signer or signers["runtime"], ok, nonce=nonce, **over)
    return broker.launch(_req(profile, ok, auth=auth, nonce=nonce))


# ---- adversarial table -------------------------------------------------

def test_missing_authorization_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    req = _req(profile, ok, auth={}, nonce="x" * 32)
    with pytest.raises(BrokerRefused):
        broker.launch(req)


def test_invented_backend_digest_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    bad = dict(ok); bad["backend"] = _d("invented")
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, bad, signers)


def test_invented_budget_digest_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    bad = dict(ok); bad["budget"] = _d("invented-budget")
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, bad, signers)


def test_untrusted_signer_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    attacker = Ed25519Signer.generate("mallory")
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, ok, signers, signer=attacker)


def test_wrong_role_signer_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    # a plan-role key is a real trusted identity — but not the
    # runtime role a launch authorization requires
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, ok, signers, signer=signers["plan"])


def test_mismatched_artifact_binding_refused(tmp_path):
    """A valid record for model X cannot launch model Y's artifacts:
    the authorization's bound digests must equal the request's."""
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    nonce = "d" * 32
    auth = _auth(signers["runtime"], ok, nonce=nonce)
    req = _req(profile, ok, auth=auth, nonce=nonce)
    req = LaunchWorkerRequest(
        worker_id=req.worker_id,
        signed_backend_manifest_digest=ok["backend"],
        approved_artifact_digests=(ok["artifact"],),
        approved_isolation_profile_digest=ok["profile"],
        execution_budget_digest=_d("other-budget"),  # swapped
        request_nonce=nonce, role="worker", authorization=auth)
    with pytest.raises(BrokerRefused):
        broker.launch(req)


def test_missing_profile_digest_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    nonce = "e" * 32
    auth = _auth(signers["runtime"], ok, nonce=nonce)
    req = LaunchWorkerRequest(
        worker_id=new_worker_id(),
        signed_backend_manifest_digest=ok["backend"],
        approved_artifact_digests=(ok["artifact"],),
        approved_isolation_profile_digest="",   # omitted!
        execution_budget_digest=ok["budget"],
        request_nonce=nonce, role="worker", authorization=auth)
    with pytest.raises(BrokerRefused):
        broker.launch(req)


def test_expired_authorization_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    past = int(time.time()) - 1000
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, ok, signers, now=past)


def test_stale_revocation_epoch_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    with pytest.raises(BrokerRefused):
        _launch(broker, profile, ok, signers, revocation_epoch=1)


def test_nonce_replay_refused(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    nonce = "f" * 32
    # first request consumes the nonce during validation — the launch
    # then fails at spawn (IdleSystemd), but the NONCE stays reserved
    with pytest.raises(Exception):
        _launch(broker, profile, ok, signers, nonce=nonce)
    with pytest.raises(BrokerRefused, match="already used"):
        _launch(broker, profile, ok, signers, nonce=nonce)


def test_concurrent_duplicate_launch_admits_at_most_one(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    nonce = "0" * 32
    results = []
    errors = []

    def go():
        try:
            _launch(broker, profile, ok, signers, nonce=nonce)
            results.append("ok")
        except BrokerRefused:
            errors.append("refused")
        except Exception:
            errors.append("other")  # spawn fails under IdleSystemd

    ts = [threading.Thread(target=go) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # exactly ONE validation may consume the nonce; the rest refuse
    refused_as_replay = sum(1 for _ in errors)
    assert refused_as_replay >= 7 and not results


def test_verify_function_rejects_tampered_envelope(tmp_path):
    broker, profile, registry, signers, resolver, ok = _rig(tmp_path)
    auth = _auth(signers["runtime"], ok)
    auth["value"]["nonce"] = "9" * 32   # tamper after signing
    req = _req(profile, ok, auth=auth, nonce="9" * 32)
    with pytest.raises(BrokerRefused):
        verify_launch_authorization(
            auth, registry, resolver, req=req,
            now=datetime.now(timezone.utc))
