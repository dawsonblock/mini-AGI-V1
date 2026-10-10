"""v16.4.4 — verified runtime closure: durable administrative audit
(WP-D), inference budgets (WP-E), and cold-start recovery (WP-F),
plus the supervisor-level in-flight quarantine invariant (WP-B/G3).
"""
import json
import os
import socket
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
from minagi.runtime.authority_store import (  # noqa: E402
    AuthorityStore, AuthorityStoreError, StoreCorrupt)
from minagi.runtime.inference_policy import (  # noqa: E402
    BudgetExceeded, InferenceBudgetPolicyV1)
from minagi.runtime.recovery_manager import RecoveryManager  # noqa: E402
from minagi.runtime.service import (  # noqa: E402
    ServiceRefused, SupervisorService)
from minagi.runtime.serving_router import (  # noqa: E402
    RoutingRefused, ServingRouter)
from minagi.runtime.supervisor import (  # noqa: E402
    ServingState, ServingSupervisor)
from minagi.security.admission_grants import issue_grant  # noqa: E402
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())


class FakeBackend:
    backend_id = "hf-peft"

    def __init__(self, *, gate=None):
        self.loaded = []
        self.unloaded = []
        self.gate = gate or threading.Event()

    def load(self, snapshot):
        handle = {"from": str(snapshot.root)}
        self.loaded.append(handle)
        return handle

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):
        self.gate.wait(timeout=10)
        return {"echo": request.get("prompt", "")}

    def unload(self, handle):
        self.unloaded.append(handle)


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _artifacts(tmp_path, tag="a"):
    model = tmp_path / f"model-{tag}"
    if not model.exists():
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "gpt2"}')
    adir = tmp_path / f"adapter-{tag}"
    if not adir.exists():
        adir.mkdir()
        (adir / "adapter_model.safetensors").write_bytes(
            b"w-" + tag.encode())
    return ({"model": close_tree(model).digest,
             "adapter": close_tree(adir).digest},
            {"model": str(model), "adapter": str(adir)})


def _grant(signers, artifact_root, *, backend="hf-peft",
           audience="local-supervisor", epoch=0):
    return issue_grant(
        signers["admission"], decision_digest=digest({"d": 1}),
        qualification_digest=digest({"q": 1}),
        runtime_manifest_digest=digest({"m": 1}),
        artifact_root_digest=artifact_root, backend_id=backend,
        audience_runtime_identity=audience, now=NOW,
        revocation_epoch=epoch)


def _supervisor(tmp_path, registry, signers, *, store_dir="journal",
                router=None):
    store = AuthorityStore(Path(tmp_path) / store_dir / "authority.sqlite")
    sup = ServingSupervisor(
        store, runtime_signer=signers["runtime"],
        registry=registry, now=NOW)
    if router is not None:
        sup.router = router
    return sup


def _drive_into(sup, signers, tmp_path, backend, *, tag="a",
                snapshot_root=None):
    """Drive a full activation, staging the snapshot under
    snapshot_root/<aid> so the recovery manager can find it."""
    digests, paths = _artifacts(tmp_path, tag)
    grant = _grant(signers, digest(digests),
                   backend=getattr(backend, "backend_id", "hf-peft"))
    act = sup.request()
    aid = act.activation_id
    sup.authorize(aid, grant)
    dest = (Path(snapshot_root) / aid if snapshot_root
            else tmp_path / "snap" / tag)
    sup.stage(aid, stage_snapshot(
        dest, paths, expected_digests=digests,
        manifest_digest=digest({"m": 1})))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid


# ---------- WP-B/G3: in-flight quarantine at supervisor level -------------

def test_quarantine_does_not_unload_live_inference(tmp_path):
    """G3 / Experiment 3: a request in flight on B when B is
    quarantined keeps B's resources — new requests refuse, but the
    backend is not unloaded while the lease references it."""
    registry, signers = _chain(tmp_path)
    router = ServingRouter(drain_timeout=0.1, cancel_grace=0.1)
    sup = _supervisor(tmp_path, registry, signers, router=router)
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    aid = _drive_into(sup, signers, tmp_path, backend, tag="a")

    out = []
    req = threading.Thread(target=lambda: out.append(
        router.route({"prompt": "long"})))
    req.start()
    time.sleep(0.05)
    assert router.inflight(aid) == 1

    sup.quarantine_active(reason="revoked mid-flight")
    # new traffic refused; the in-flight lease still owns the backend
    with pytest.raises(RoutingRefused):
        router.route({"prompt": "x"})
    assert not backend.unloaded
    assert router.inflight(aid) == 1

    gate.set()          # let the request finish
    req.join(timeout=5)
    assert out and out[0]["result"]["echo"] == "long"
    sup.reap()
    assert backend.unloaded, "released leases must free the model"


def test_abort_releases_after_lease_release(tmp_path):
    """abort() while a request runs: routing stops first, the model is
    released only after the lease actually ends."""
    registry, signers = _chain(tmp_path)
    router = ServingRouter(drain_timeout=0.05, cancel_grace=0.05)
    sup = _supervisor(tmp_path, registry, signers, router=router)
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    aid = _drive_into(sup, signers, tmp_path, backend, tag="a")
    lease = router.acquire_lease()
    sup.abort(aid, reason="test")
    assert not backend.unloaded
    router.release_lease(lease)
    sup.reap()
    assert backend.unloaded


# ---------- WP-D: durable administrative audit ---------------------------

def _svc(tmp_path, *, audit_store=None, insecure=True):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    svc = SupervisorService(
        tmp_path / "x.sock", launcher=None, supervisor=sup,
        backend_factories={}, uid_roles={},
        audit_signer=signers["runtime"],
        allow_insecure_dev=insecure, audit_store=audit_store)
    return svc, sup, signers


def test_privileged_op_writes_durable_audit(tmp_path):
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    svc, sup, _ = _svc(tmp_path, audit_store=store)
    svc.supervisor.store = store  # same store for ops + audit
    from minagi.runtime.access_policy import PrincipalContext
    op = PrincipalContext(uid=0, role="operator",
                          principal_id="uid:0")
    resp = svc._handle_request({"op": "recover"}, op)
    assert resp["ok"]
    events = store.admin_events()
    decisions = [e["decision"] for e in events]
    assert "allowed" in decisions and "completed" in decisions
    for e in events:
        assert e["principal_id"] == "uid:0"
        assert e["signature_b64"]
    assert store.verify_admin_chain(sup.registry, now=NOW)


def test_audit_failure_refuses_ordinary_admin_change(tmp_path):
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    svc, sup, signers = _svc(tmp_path, audit_store=store)
    sup.store = store
    from minagi.runtime.access_policy import PrincipalContext
    op = PrincipalContext(uid=0, role="operator",
                          principal_id="uid:0")

    def boom(**kw):
        raise AuthorityStoreError("audit disk full")
    store.append_admin_audit = boom
    with pytest.raises(ServiceRefused, match="audit"):
        svc._handle_request({"op": "rollback"}, op)
    # and NO state changed — the refusal happened before the side effect
    assert not store.admin_events()


def test_emergency_quarantine_survives_audit_outage(tmp_path):
    """WP-D exception: stopping unauthorized serving never depends on
    the audit database being writable — but activation blocks until
    the audit trail reconciles."""
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    svc, sup, signers = _svc(tmp_path, audit_store=store)
    sup.store = store
    backend = FakeBackend()
    _drive_into(sup, signers, tmp_path, backend, tag="a")
    assert sup.serving_state is ServingState.SERVING
    from minagi.runtime.access_policy import PrincipalContext
    op = PrincipalContext(uid=0, role="operator",
                          principal_id="uid:0")

    def boom(**kw):
        raise AuthorityStoreError("audit disk full")
    store.append_admin_audit = boom
    resp = svc._handle_request(
        {"op": "quarantine", "reason": "incident"}, op)
    assert resp["ok"]                      # the stop still happened
    assert svc._audit_broken is True
    # activation is blocked until the audit trail reconciles
    with pytest.raises(ServiceRefused, match="audit"):
        svc._handle_request({"op": "launch", "request": {}}, op)
    # reconciliation: a successful privileged-op audit clears the flag
    del store.append_admin_audit
    resp = svc._handle_request({"op": "recover"}, op)
    assert resp["ok"] and svc._audit_broken is False


def test_peer_uid_proves_local_identity(tmp_path):
    """OPS-006: peer_uid must extract the real peer uid wherever the
    platform offers a mechanism (SO_PEERCRED on Linux, getpeereid(2)
    on macOS/BSD) — not silently fall through to None."""
    from minagi.runtime.service import peer_uid
    a, b = socket.socketpair()
    try:
        uid = peer_uid(a)
        if sys.platform in ("linux", "darwin") or \
                sys.platform.startswith("freebsd"):
            assert uid == os.getuid()
        else:
            pytest.skip("no peer-credential mechanism on this platform")
    finally:
        a.close()
        b.close()


def test_audit_outage_flag_survives_restart(tmp_path):
    """RUN-403: the audit-broken state is DURABLE — a restart during the
    outage must not silently re-arm launch with a gap in the audit
    trail. The flag lives in a file next to the authority database."""
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    svc, sup, signers = _svc(tmp_path, audit_store=store)
    sup.store = store
    backend = FakeBackend()
    _drive_into(sup, signers, tmp_path, backend, tag="a")
    from minagi.runtime.access_policy import PrincipalContext
    op = PrincipalContext(uid=0, role="operator",
                          principal_id="uid:0")

    def boom(**kw):
        raise AuthorityStoreError("audit disk full")
    store.append_admin_audit = boom
    resp = svc._handle_request(
        {"op": "quarantine", "reason": "incident"}, op)
    assert resp["ok"] and svc._audit_broken is True
    flag = Path(store.db_path).parent / "audit_broken.json"
    assert flag.is_file()
    recorded = json.loads(flag.read_text())
    assert recorded["op"] == "quarantine"
    assert recorded["principal_id"] == "uid:0"
    assert recorded["set_at"] > 0

    # "restart": a new service object over the same store rehydrates
    # the durable flag — launch stays refused even though the new
    # process never saw the outage in memory.
    svc2, sup2, _ = _svc(tmp_path, audit_store=store)
    assert svc2._audit_broken is True
    del store.append_admin_audit
    with pytest.raises(ServiceRefused, match="audit"):
        svc2._handle_request({"op": "launch", "request": {}}, op)

    # reconciliation on the restarted service clears flag + file, and
    # the reconciling audit carries the recorded outage as evidence
    resp = svc2._handle_request({"op": "recover"}, op)
    assert resp["ok"] and svc2._audit_broken is False
    assert not flag.exists()
    last = store.admin_events()[-1]
    assert last["decision"] == "completed"
    assert last["detail"]["reconciles_outage"]["op"] == "quarantine"


def test_admin_chain_detects_tamper(tmp_path):
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    registry, signers = _chain(tmp_path)
    store.append_admin_audit(
        principal_id="uid:0", operation="quarantine",
        target_activation="aa" * 16, policy_digest=digest({"p": 1}),
        decision="allowed", before_state={"s": 1}, after_state={"s": 2},
        at=TS, signer=signers["runtime"])
    store._db.execute(
        "UPDATE admin_audit SET decision = 'denied' WHERE sequence = 1")
    store._db.commit()
    with pytest.raises(StoreCorrupt):
        store.verify_admin_chain(registry, now=NOW)


def test_admin_chain_verifies_signature_after_digest_rewrite(tmp_path):
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    registry, signers = _chain(tmp_path)
    store.append_admin_audit(
        principal_id="uid:0", operation="quarantine",
        target_activation="aa" * 16, policy_digest=digest({"p": 1}),
        decision="allowed", before_state={"s": 1}, after_state={"s": 2},
        at=TS, signer=signers["runtime"])
    event = store.admin_events()[0]
    payload = {
        "event_id": event["event_id"],
        "principal_id": event["principal_id"],
        "operation": event["operation"],
        "target_activation": event["target_activation"],
        "policy_digest": event["policy_digest"],
        "decision": event["decision"],
        "before_state": event["before_state"],
        "after_state": event["after_state"],
        "at": event["at"],
        "detail": event["detail"]}
    body = {
        "event_id": event["event_id"],
        "principal_id": event["principal_id"],
        "operation": event["operation"],
        "target_activation": event["target_activation"],
        "policy_digest": event["policy_digest"],
        "decision": event["decision"],
        "before_state": event["before_state"],
        "after_state": event["after_state"],
        "at": event["at"],
        "payload_digest": digest(payload),
        "previous_event_digest": event["previous_event_digest"]}
    signature = "Zm9yZ2Vk"
    event_digest = digest(dict(
        body, signer_key_id=event["signer_key_id"],
        signature_b64=signature))
    store._db.execute(
        "UPDATE admin_audit SET signature_b64 = ?, event_digest = ? "
        "WHERE sequence = 1", (signature, event_digest))
    store._db.commit()

    with pytest.raises(StoreCorrupt, match="signature invalid"):
        store.verify_admin_chain(registry, now=NOW)


def test_admin_chain_refuses_non_runtime_signer(tmp_path):
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    registry, signers = _chain(tmp_path)
    store.append_admin_audit(
        principal_id="uid:0", operation="quarantine",
        target_activation="aa" * 16, policy_digest=digest({"p": 1}),
        decision="allowed", before_state={"s": 1}, after_state={"s": 2},
        at=TS, signer=signers["admission"])

    with pytest.raises(StoreCorrupt, match="authorized runtime"):
        store.verify_admin_chain(registry, now=NOW)


# ---------- WP-E: inference budgets -------------------------------------

def test_budget_policy_validation():
    with pytest.raises(ValueError):
        InferenceBudgetPolicyV1(max_prompt_tokens=0)
    with pytest.raises(ValueError):
        InferenceBudgetPolicyV1(max_concurrent_requests=0)
    pol = InferenceBudgetPolicyV1.from_doc(
        InferenceBudgetPolicyV1(max_new_tokens=64).to_doc())
    assert pol.max_new_tokens == 64
    with pytest.raises(ValueError, match="schema"):
        InferenceBudgetPolicyV1.from_doc({"schema": "bogus"})


def test_router_concurrency_budget_refused(tmp_path):
    budget = InferenceBudgetPolicyV1(max_concurrent_requests=1,
                                     max_queued_per_principal=5)
    router = ServingRouter(budget=budget)

    class B:
        backend_id = "hf-peft"

        def infer(self, h, r):
            return r

    router.activate("aa" * 16, B(), {"id": 1})
    lease = router.acquire_lease()
    with pytest.raises(RoutingRefused, match="concurrency"):
        router.acquire_lease()
    router.release_lease(lease)
    assert router.route({"prompt": "x"})["result"]["prompt"] == "x"


def test_router_per_principal_budget_refused(tmp_path):
    budget = InferenceBudgetPolicyV1(max_concurrent_requests=10,
                                     max_queued_per_principal=1)
    router = ServingRouter(budget=budget)

    class B:
        backend_id = "hf-peft"

        def infer(self, h, r):
            return r

    router.activate("aa" * 16, B(), {"id": 1})
    lease = router.acquire_lease(principal="uid:7")
    with pytest.raises(RoutingRefused, match="per-principal"):
        router.acquire_lease(principal="uid:7")
    # a different principal is unaffected
    other = router.acquire_lease(principal="uid:8")
    router.release_lease(other)
    router.release_lease(lease)


def test_peft_infer_enforces_token_budget():
    """SEC-306: actual prompt token count is checked; oversized
    requests refuse and max_new_tokens is clamped to the bound."""
    from minagi.v161.peft_serving import PeftServingBackend

    class FakeTensor:
        def __init__(self, rows):
            self._rows = rows

        @property
        def shape(self):
            return (1, len(self._rows))

    class FakeIds(dict):
        pass

    class FakeTokenizer:
        def __init__(self, n):
            self.n = n

        def __call__(self, prompt, return_tensors="pt"):
            return FakeIds(input_ids=FakeTensor([0] * self.n))

        def decode(self, out, skip_special_tokens=True):
            return "done"

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kw):
            self.calls.append(kw)
            return [[0, 1, 2]]

    budget = InferenceBudgetPolicyV1(max_prompt_tokens=8,
                                     max_new_tokens=16)
    backend = PeftServingBackend(budget=budget)
    model = FakeModel()
    handle = {"model": model, "tokenizer": FakeTokenizer(4)}
    out = backend.infer(handle, {"prompt": "x",
                                 "max_new_tokens": 10_000})
    assert model.calls[0]["max_new_tokens"] == 16
    assert out["metrics"]["max_new_tokens_requested"] == 10_000
    assert out["metrics"]["max_new_tokens_applied"] == 16
    assert out["metrics"]["prompt_tokens"] == 4
    # over-budget prompt refuses
    handle2 = {"model": model, "tokenizer": FakeTokenizer(64)}
    with pytest.raises(BudgetExceeded):
        backend.infer(handle2, {"prompt": "too long"})


# ---------- WP-F: cold-start recovery ------------------------------------

def _snap(tmp_path, aid, digests, paths, root):
    return stage_snapshot(Path(root) / aid, paths,
                          expected_digests=digests,
                          manifest_digest=digest({"m": 1}))


def test_cold_restart_restores_live_model(tmp_path):
    """WP-F / Experiment 4: after a crash the service reauthorizes,
    reloads, health-checks and publishes a verified live model — the
    pointer alone never declares SERVING."""
    registry, signers = _chain(tmp_path)
    snaproot = tmp_path / "snaps"
    sup = _supervisor(tmp_path, registry, signers)
    backend = FakeBackend()
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    act = sup.request()
    aid = act.activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _snap(tmp_path, aid, digests, paths, snaproot))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    assert sup.serving_state is ServingState.SERVING

    # simulate crash: a fresh supervisor on the same store
    sup2 = _supervisor(tmp_path, registry, signers)
    assert sup2.serving_state is ServingState.RECOVERY_REQUIRED

    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={"hf-peft": FakeBackend}, now=NOW)
    report = mgr.restore()
    assert report["restoration"]["status"] == "restored"
    new_id = report["restoration"]["restored_activation"]
    assert new_id != aid                      # a NEW activation
    assert sup2.serving_state is ServingState.SERVING
    dep = sup2.store.deployment()
    assert dep["desired_activation_id"] == new_id
    assert dep["transition_phase"] == "ROUTED"
    # the restored grant is a DIFFERENT grant (never reused)
    grants = {e["activation_id"]: e["detail"].get("grant_id")
              for e in sup2.store.events()
              if e["event_type"] == "authorized"}
    assert grants[new_id]
    assert grants[new_id] != grants[aid]


def test_restoration_refuses_missing_artifacts(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    backend = FakeBackend()
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _snap(tmp_path, aid, digests, paths,
                         tmp_path / "snaps"))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)

    import shutil
    snap = tmp_path / "snaps" / aid
    for p in list(snap.rglob("*")) + [snap]:
        try:
            p.chmod(0o700)
        except OSError:
            pass
    shutil.rmtree(snap)

    sup2 = _supervisor(tmp_path, registry, signers)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=tmp_path / "snaps",
        backend_factories={"hf-peft": FakeBackend}, now=NOW)
    report = mgr.restore()
    assert report["restoration"]["status"] == "unavailable"
    assert sup2.serving_state is ServingState.UNAVAILABLE
    dep = sup2.store.deployment()
    assert dep["transition_phase"] == "UNAVAILABLE"


def test_restoration_refuses_tampered_artifacts(tmp_path):
    """Bytes changed after the durable commit cannot satisfy the
    recorded artifact root."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    backend = FakeBackend()
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _snap(tmp_path, aid, digests, paths,
                         tmp_path / "snaps"))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)

    # tamper with the staged adapter
    staged = tmp_path / "snaps" / aid / "adapter" / \
        "adapter_model.safetensors"
    staged.chmod(0o600)
    staged.write_bytes(b"evil-weights")

    sup2 = _supervisor(tmp_path, registry, signers)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=tmp_path / "snaps",
        backend_factories={"hf-peft": FakeBackend}, now=NOW)
    report = mgr.restore()
    assert report["restoration"]["status"] == "unavailable"


def test_fallback_restores_with_its_own_backend(tmp_path):
    class AlternateBackend(FakeBackend):
        backend_id = "alternate"

    registry, signers = _chain(tmp_path)
    snaproot = tmp_path / "snaps"
    sup = _supervisor(tmp_path, registry, signers)
    first_id = _drive_into(
        sup, signers, tmp_path, AlternateBackend(), tag="a",
        snapshot_root=snaproot)
    desired_id = _drive_into(
        sup, signers, tmp_path, FakeBackend(), tag="b",
        snapshot_root=snaproot)

    staged = snaproot / desired_id / "adapter" / \
        "adapter_model.safetensors"
    staged.chmod(0o600)
    staged.write_bytes(b"tampered desired weights")

    sup2 = _supervisor(tmp_path, registry, signers)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={"hf-peft": FakeBackend,
                           "alternate": AlternateBackend}, now=NOW)
    report = mgr.restore()

    assert report["restoration"]["status"] == "restored"
    assert report["restoration"]["restored_from"] == first_id
    assert sup2.store.read_pointer()["backend_id"] == "alternate"


def test_fallback_restoration_refuses_changed_artifacts(tmp_path):
    registry, signers = _chain(tmp_path)
    snaproot = tmp_path / "snaps"
    sup = _supervisor(tmp_path, registry, signers)
    first_id = _drive_into(
        sup, signers, tmp_path, FakeBackend(), tag="a",
        snapshot_root=snaproot)
    desired_id = _drive_into(
        sup, signers, tmp_path, FakeBackend(), tag="b",
        snapshot_root=snaproot)

    for aid in (first_id, desired_id):
        staged = snaproot / aid / "adapter" / "adapter_model.safetensors"
        staged.chmod(0o600)
        staged.write_bytes(b"tampered weights")

    sup2 = _supervisor(tmp_path, registry, signers)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={"hf-peft": FakeBackend}, now=NOW)
    report = mgr.restore()

    assert report["restoration"]["status"] == "unavailable"
    fallback = next(a for a in report["restoration"]["attempts"]
                    if a["candidate"] == first_id)
    assert "recorded authorization artifact root" in fallback["reason"]


def test_restoration_is_idempotent(tmp_path):
    """Repeating recovery does not create conflicting generations or
    reuse grants."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    backend = FakeBackend()
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _snap(tmp_path, aid, digests, paths,
                         tmp_path / "snaps"))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)

    sup2 = _supervisor(tmp_path, registry, signers)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=tmp_path / "snaps",
        backend_factories={"hf-peft": FakeBackend}, now=NOW)
    r1 = mgr.restore()
    assert r1["restoration"]["status"] == "restored"
    # second restore: nothing to restore — clean no-op
    r2 = mgr.restore()
    assert r2["requires_restoration"] is None
    dep = sup2.store.deployment()
    assert dep["transition_phase"] == "ROUTED"


# ---------- WP-A migration: v16.4.3 store -> v16.4.4 ---------------------

def test_v1643_store_migrates_deployment_row(tmp_path):
    """A v16.4.3-format database gains the deployment row, seeded from
    its serving pointer."""
    db = tmp_path / "old" / "authority.sqlite"
    db.parent.mkdir(parents=True)
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.executescript("""
    CREATE TABLE admission_grants (
        grant_id TEXT PRIMARY KEY, grant_nonce TEXT NOT NULL UNIQUE,
        grant_digest TEXT NOT NULL UNIQUE,
        activation_id TEXT NOT NULL UNIQUE,
        runtime_identity TEXT NOT NULL,
        reservation_state TEXT NOT NULL,
        reserved_at INTEGER NOT NULL, expires_at INTEGER NOT NULL);
    CREATE TABLE runtime_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        activation_id TEXT NOT NULL, event_type TEXT NOT NULL,
        from_state TEXT NOT NULL, to_state TEXT NOT NULL,
        at INTEGER NOT NULL, detail_json TEXT NOT NULL,
        payload_digest TEXT NOT NULL, previous_digest TEXT NOT NULL,
        event_digest TEXT NOT NULL UNIQUE,
        signer_key_id TEXT NOT NULL, signature_b64 TEXT NOT NULL);
    CREATE TABLE launch_requests (
        request_id TEXT PRIMARY KEY, activation_id TEXT NOT NULL,
        outcome TEXT NOT NULL, outcome_digest TEXT NOT NULL,
        recorded_at INTEGER NOT NULL);
    CREATE TABLE serving_pointer (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        activation_id TEXT NOT NULL, artifact_root_digest TEXT NOT NULL,
        backend_id TEXT NOT NULL, committed_at INTEGER NOT NULL,
        event_sequence INTEGER NOT NULL);
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO serving_pointer(id, activation_id, "
        "artifact_root_digest, backend_id, committed_at, "
        "event_sequence) VALUES(1,?,?,?,?,?)",
        ("aa" * 16, "sha256:" + "0" * 64, "hf-peft", TS, 1))
    conn.commit()
    conn.close()

    store = AuthorityStore(db)   # opens + migrates
    dep = store.deployment()
    assert dep is not None
    assert dep["deployment_generation"] == 1
    assert dep["desired_activation_id"] == "aa" * 16
    assert dep["transition_phase"] == "COMMITTED"
    ptr = store.read_pointer()
    assert ptr["activation_id"] == "aa" * 16
    # and the generation protocol still applies on top
    nxt = store.commit_activation_intent(
        candidate_id="bb" * 16, expected_generation=1, at=TS,
        artifact_root_digest=digest({"r": 1}), backend_id="hf-peft")
    assert nxt["generation"] == 2
