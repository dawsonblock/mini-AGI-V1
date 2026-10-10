"""v16.4.3 — ServingSupervisor transactional activation lifecycle.

HARDENING_PLAN WP1/WP5: grant reservations are durable + atomic in the
authority store, the commit intent and serving pointer move in one
transaction, the signed completion lands after, and recovery verifies
the hash-chained event log before reconciling — never reporting SERVING
for a model that is not resident.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import (  # noqa: E402
    AuthorityStore, AuthorityStoreError)
from minagi.runtime.supervisor import (  # noqa: E402
    ActivationError, ActivationRefused, ServingState,
    ServingSupervisor)
from minagi.security.admission_grants import issue_grant  # noqa: E402
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())


class FakeBackend:
    backend_id = "hf-peft"

    def __init__(self, *, fail_load=False, fail_probe=False):
        self.fail_load = fail_load
        self.fail_probe = fail_probe
        self.loaded = []
        self.unloaded = []

    def load(self, snapshot):
        if self.fail_load:
            raise RuntimeError("weights unreadable")
        handle = {"from": str(snapshot.root)}
        self.loaded.append(handle)
        return handle

    def health_probe(self, handle):
        if self.fail_probe:
            raise RuntimeError("probe timed out")

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


def _stage(tmp_path, tag, artifact_digests, paths=None):
    dest = tmp_path / "snap" / tag
    return stage_snapshot(
        dest, paths or _artifacts(tmp_path, tag)[1],
        expected_digests=artifact_digests, manifest_digest=digest({"m": 1}))


def _supervisor(tmp_path, registry, signers, *, store_dir="journal"):
    store = AuthorityStore(Path(tmp_path) / store_dir / "authority.sqlite")
    return ServingSupervisor(
        store, runtime_signer=signers["runtime"],
        registry=registry, now=NOW)


def _events(sup, aid):
    return sup.store.events(aid)


def _drive(sup, signers, tmp_path, backend, *, tag="a"):
    digests, paths = _artifacts(tmp_path, tag)
    grant = _grant(signers, digest(digests))
    act = sup.request()
    aid = act.activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _stage(tmp_path, tag, digests, paths))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid


# ---------- happy path ---------------------------------------------------

def test_full_lifecycle_commits_atomically(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    aid = _drive(sup, signers, tmp_path, FakeBackend())

    assert sup.active_id == aid
    assert sup.serving_state is ServingState.SERVING
    pointer = sup.active_pointer()
    assert pointer["activation_id"] == aid
    # intent is durable before the pointer, observed routing signed after
    hist = _events(sup, aid)
    kinds = [e["detail"].get("kind") or e["event_type"] for e in hist]
    assert "commit_intent" in [e["event_type"] for e in hist]
    assert "routing_observed" in [e["event_type"] for e in hist]
    completion = [e for e in hist
                  if e["event_type"] == "routing_observed"][0]
    assert completion["signer_key_id"]
    states = [e["to_state"] for e in hist]
    assert states.index("COMMITTED") < states.index("ACTIVE")
    assert isinstance(kinds, list)


def test_activation_ids_are_server_generated(tmp_path):
    """SEC-204: the trusted side picks opaque activation ids."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    act = sup.request()
    assert len(act.activation_id) == 32
    int(act.activation_id, 16)  # hex
    with pytest.raises(ActivationRefused, match="already exists"):
        sup.request(activation_id=act.activation_id)
    with pytest.raises(ActivationRefused):
        sup.request(activation_id="../../escape")
    with pytest.raises(ActivationRefused):
        sup.request(activation_id="act-1")


# ---------- stage / grant gates ------------------------------------------

def test_arbitrary_object_with_path_refused(tmp_path):
    """An object that merely quacks like a snapshot is not measured
    evidence — spec row 2."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, _paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)

    class Quacks:
        def path(self, name):
            return tmp_path

        @property
        def artifact_digests(self):
            return tuple(sorted(digests.items()))

    with pytest.raises(ActivationRefused, match="MeasuredSnapshot"):
        sup.stage(aid, Quacks())
    assert _events(sup, aid)[-1]["to_state"] == "ABORTED"


def test_grant_artifact_mismatch_aborts(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest({"adapter": "sha256:" + "0" * 64,
                                    "model": "sha256:" + "0" * 64}))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    with pytest.raises(ActivationRefused, match="does not match"):
        sup.stage(aid, _stage(tmp_path, "a", digests, paths))


def test_wrong_backend_aborts_at_prepare(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests), backend="hf-peft")
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _stage(tmp_path, "a", digests, paths))

    class OtherBackend(FakeBackend):
        backend_id = "qwen-native-cuda"

    with pytest.raises(ActivationRefused, match="authorized backend"):
        sup.prepare(aid, OtherBackend())


def test_grant_replay_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    grant = _grant(signers, digest({"x": 1}))
    a1 = sup.request().activation_id
    sup.authorize(a1, grant)
    a2 = sup.request().activation_id
    with pytest.raises(ActivationRefused, match="one activation"):
        sup.authorize(a2, grant)


def test_grant_replay_survives_restart(tmp_path):
    """SEC-201: a grant consumed before a restart stays consumed."""
    registry, signers = _chain(tmp_path)
    grant = _grant(signers, digest({"x": 1}))
    sup = _supervisor(tmp_path, registry, signers)
    a1 = sup.request().activation_id
    sup.authorize(a1, grant)
    # restart: a fresh supervisor on the same store
    fresh = _supervisor(tmp_path, registry, signers)
    a2 = fresh.request().activation_id
    with pytest.raises(ActivationRefused, match="one activation"):
        fresh.authorize(a2, grant)


# ---------- failure injection --------------------------------------------

def test_backend_load_failure_aborts_without_pointer(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _stage(tmp_path, "a", digests, paths))
    with pytest.raises(ActivationError, match="failed to load"):
        sup.prepare(aid, FakeBackend(fail_load=True))
    assert sup.active_id is None
    assert _events(sup, aid)[-1]["to_state"] == "ABORTED"


def test_failed_health_probe_unloads_and_keeps_previous(tmp_path):
    """Spec row: fail the new model health check → previous healthy
    version remains active."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    assert sup.active_id == aid1

    bad = FakeBackend(fail_probe=True)
    digests, paths = _artifacts(tmp_path, "b")
    grant = _grant(signers, digest(digests))
    aid2 = sup.request().activation_id
    sup.authorize(aid2, grant)
    sup.stage(aid2, _stage(tmp_path, "b", digests, paths))
    sup.prepare(aid2, bad)
    with pytest.raises(ActivationError, match="health probe failed"):
        sup.health_check(aid2)
    assert bad.unloaded, "failed candidate must be unloaded"
    assert sup.active_id == aid1
    assert sup.active_pointer()["activation_id"] == aid1


def test_commit_transaction_failure_aborts(tmp_path):
    """Crash the commit intent+pointer transaction → the candidate
    aborts and no pointer was moved (the txn is atomic)."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _stage(tmp_path, "a", digests, paths))
    sup.prepare(aid, FakeBackend())
    sup.health_check(aid)

    def boom(**kw):
        raise AuthorityStoreError("disk full")
    sup.store.commit_activation_intent = boom
    with pytest.raises(ActivationError, match="commit"):
        sup.commit_activation(aid)
    assert _events(sup, aid)[-1]["to_state"] == "ABORTED"
    assert sup.active_id is None
    assert sup.store.read_pointer() is None
    dep = sup.store.deployment()
    assert dep is None or dep["desired_activation_id"] == ""


def test_routing_observation_failure_rolls_back_pointer(tmp_path):
    """SEC-301: if the signed routing observation cannot be persisted
    after the durable commit, the candidate is withdrawn and the
    durable intent reconciled to the previous live runtime — nothing
    unevidenced stays routable, and A is restored durably, not just in
    memory."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    assert sup.active_id == aid1

    digests, paths = _artifacts(tmp_path, "b")
    grant = _grant(signers, digest(digests))
    aid2 = sup.request().activation_id
    sup.authorize(aid2, grant)
    sup.stage(aid2, _stage(tmp_path, "b", digests, paths))
    sup.prepare(aid2, FakeBackend())
    sup.health_check(aid2)

    def flaky(**kw):
        raise AuthorityStoreError("simulated persistence failure")
    sup.store.record_routing_observation = flaky

    with pytest.raises(ActivationError, match="routing"):
        sup.commit_activation(aid2)
    # reconciled to the live predecessor — durable AND in memory
    assert sup.active_id == aid1
    ptr = sup.active_pointer()
    assert ptr["activation_id"] == aid1
    dep = sup.store.deployment()
    assert dep["desired_activation_id"] == aid1
    assert dep["transition_phase"] == "RESTORED"


def test_publish_failure_restores_predecessor_durably(tmp_path):
    """SEC-301/A3: a routing failure after the durable commit cannot
    leave B routable — desired returns to A in the SAME durable
    generation protocol."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    from minagi.runtime.serving_router import ServingRouter
    sup.router = ServingRouter()
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    assert sup.active_id == aid1

    digests, paths = _artifacts(tmp_path, "b")
    grant = _grant(signers, digest(digests))
    aid2 = sup.request().activation_id
    sup.authorize(aid2, grant)
    sup.stage(aid2, _stage(tmp_path, "b", digests, paths))
    failed_backend = FakeBackend()
    sup.prepare(aid2, failed_backend)
    sup.health_check(aid2)

    orig_publish = sup.router.publish_route

    def flaky(*a, **kw):
        if a and a[0] == aid2:
            raise RuntimeError("routing plane down")
        return orig_publish(*a, **kw)
    sup.router.publish_route = flaky
    with pytest.raises(ActivationError, match="publication"):
        sup.commit_activation(aid2)
    assert sup.active_id == aid1
    assert sup.active_pointer()["activation_id"] == aid1
    dep = sup.store.deployment()
    assert dep["desired_activation_id"] == aid1
    # B never received traffic
    assert sup.router.inflight(aid2) == 0
    st = sup.router.route_entry(aid2)
    assert st is None or st.accepting is False
    assert aid2 not in sup._live_handles
    assert failed_backend.unloaded


def test_zero_traffic_before_durable_intent(tmp_path):
    """G1: a persistence failure during stage 1 means B is never
    published — zero requests can reach it."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    from minagi.runtime.serving_router import ServingRouter
    sup.router = ServingRouter()
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, _stage(tmp_path, "a", digests, paths))
    sup.prepare(aid, FakeBackend())
    sup.health_check(aid)

    def boom(**kw):
        raise AuthorityStoreError("disk full")
    sup.store.commit_activation_intent = boom
    with pytest.raises(ActivationError):
        sup.commit_activation(aid)
    from minagi.runtime.serving_router import RoutingRefused
    with pytest.raises(RoutingRefused):
        sup.router.route({"prompt": "x"})
    dep = sup.store.deployment()
    assert dep is None or dep["desired_activation_id"] == ""


def test_concurrent_activations_single_winner(tmp_path):
    """A5: two transitions racing on the same generation — exactly one
    wins the CAS."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    store = sup.store
    import threading
    results = {}

    def commit(tag):
        try:
            results[tag] = store.commit_activation_intent(
                candidate_id=tag * 16, expected_generation=0, at=TS,
                artifact_root_digest=digest({"r": tag}),
                backend_id="hf-peft")
        except Exception as exc:  # noqa: BLE001
            results[tag] = exc

    t1 = threading.Thread(target=commit, args=("a",))
    t2 = threading.Thread(target=commit, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()
    from minagi.runtime.authority_store import GenerationConflict
    outcomes = list(results.values())
    wins = [o for o in outcomes if isinstance(o, dict)]
    losses = [o for o in outcomes if isinstance(o, GenerationConflict)]
    assert len(wins) == 1 and len(losses) == 1
    dep = store.deployment()
    assert dep["deployment_generation"] == 1


def test_stale_generation_refused(tmp_path):
    """A transition based on generation 10 cannot overwrite a completed
    generation 11."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    dep = sup.store.deployment()
    assert dep["deployment_generation"] == 1
    from minagi.runtime.authority_store import GenerationConflict
    with pytest.raises(GenerationConflict):
        sup.store.commit_activation_intent(
            candidate_id="cc" * 8, expected_generation=0, at=TS,
            artifact_root_digest=digest({"r": 1}), backend_id="hf-peft")


# ---------- quarantine + rollback ----------------------------------------

def test_revoked_active_version_quarantined_and_rolled_back(tmp_path):
    """Spec row: revoke the currently active version → quarantine and
    restore the retained committed predecessor."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    aid2 = _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    assert sup.active_id == aid2

    sup.quarantine_active(reason="promotion revoked")
    assert sup.active_id == aid1
    assert sup.active_pointer()["activation_id"] == aid1
    assert _events(sup, aid2)[-1]["to_state"] == "QUARANTINED"


def test_rollback_restores_retained_live_runtime(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    target = sup.rollback()
    assert target == aid1
    assert sup.active_id == aid1
    hist = _events(sup, aid1)
    kinds = [e["event_type"] for e in hist]
    assert "rollback_intent" in kinds
    assert "rollback_completion" in kinds
    completion = [e for e in hist
                  if e["event_type"] == "rollback_completion"][0]
    assert completion["signer_key_id"]


def test_rollback_publish_failure_reconciles_without_compat_fallback(
        tmp_path):
    registry, signers = _chain(tmp_path)
    from minagi.runtime.serving_router import ServingRouter
    router = ServingRouter()
    sup = _supervisor(tmp_path, registry, signers)
    sup.router = router
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    aid2 = _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    publish = router.publish_route
    activate_calls = []

    def fail_target(activation_id, **kwargs):
        if activation_id == aid1:
            raise RuntimeError("stale route generation")
        return publish(activation_id, **kwargs)

    def compatibility_activate(*args, **kwargs):
        activate_calls.append(args)
        raise AssertionError("generation-free activation must not be used")

    router.publish_route = fail_target
    router.activate = compatibility_activate
    with pytest.raises(ActivationError, match="rollback routing"):
        sup.rollback()

    assert not activate_calls
    assert sup.active_pointer()["activation_id"] == aid2
    assert sup.store.deployment()["desired_activation_id"] == aid2
    assert router.route_entry(aid2).accepting
    assert not router.route_entry(aid1).accepting


def test_rollback_completion_failure_reconciles_route_and_pointer(
        tmp_path):
    registry, signers = _chain(tmp_path)
    from minagi.runtime.serving_router import ServingRouter
    router = ServingRouter()
    sup = _supervisor(tmp_path, registry, signers)
    sup.router = router
    aid1 = _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    aid2 = _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    record = sup.store.record_routing_observation

    def fail_rollback(**kwargs):
        if kwargs.get("event_type") == "rollback_completion":
            raise AuthorityStoreError("simulated completion failure")
        return record(**kwargs)

    sup.store.record_routing_observation = fail_rollback
    with pytest.raises(ActivationError, match="rollback completion"):
        sup.rollback()

    assert sup.active_pointer()["activation_id"] == aid2
    assert sup.store.deployment()["desired_activation_id"] == aid2
    assert router.route_entry(aid2).accepting
    assert not router.route_entry(aid1).accepting


def test_rollback_refuses_dead_runtime(tmp_path):
    """The pointer must never be restored to a model that is not
    actually loaded — rollback requires a live handle."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    b1 = FakeBackend()
    aid1 = _drive(sup, signers, tmp_path, b1, tag="a")
    _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    # kill the retained runtime's handle out from under the supervisor
    pair = sup._live_handles.pop(aid1)
    b1.unload(pair[1])
    with pytest.raises(ActivationRefused, match="re-admit"):
        sup.rollback()


def test_rollback_rechecks_revocation_epoch(tmp_path):
    """WP8: a rollback target authorized under a superseded revocation
    epoch is refused — rollback is not a security bypass."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    sup._revocation_epoch_provider = lambda: 5
    _drive(sup, signers, tmp_path, FakeBackend(), tag="a")
    _drive(sup, signers, tmp_path, FakeBackend(), tag="b")
    # act-1's grant was issued at revocation_epoch 0 < operative 5
    with pytest.raises(ActivationRefused, match="stale"):
        sup.rollback()


# ---------- crash recovery -----------------------------------------------

def _write_history(store, aid, states, *, signer=None):
    src = ""
    for st in states:
        store.append_event(activation_id=aid, event_type="transition",
                           from_state=src, to_state=st, at=TS,
                           signer=signer)
        src = st


def test_recovery_aborts_committed_but_unrouted(tmp_path):
    """Crash between durable intent and pointer swap: the candidate
    never routed → abort on recovery."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    _write_history(store, "a1" * 8,
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED",
                    "READY", "COMMITTED"])
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    report = sup.recover()
    assert "a1" * 8 in report["aborted"]
    assert _events(sup, "a1" * 8)[-1]["to_state"] == "ABORTED"


def test_recovery_flags_committed_pointer_for_restoration(tmp_path):
    """SEC-202: a durable committed pointer means RECOVERY_REQUIRED —
    historical evidence, not liveness. The service is never SERVING on
    boot until a live model is re-verified."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    aid = "b2" * 8
    _write_history(store, aid,
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED",
                    "READY"])
    store.commit_with_pointer(
        activation_id=aid, at=TS, artifact_root_digest=digest({"r": 1}),
        backend_id="hf-peft", detail={"kind": "activation_intent"})
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    # boot state: durable commit exists but nothing is live
    assert sup.serving_state is ServingState.RECOVERY_REQUIRED
    report = sup.recover()
    assert report["serving_state"] == "RECOVERY_REQUIRED"
    assert report["requires_restoration"] == aid
    assert sup.active_id is None  # nothing routed


def test_recovery_aborts_stalled_before_commit(tmp_path):
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    _write_history(store, "c3" * 8,
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED"])
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    report = sup.recover()
    assert report["aborted"] == ["c3" * 8]


def test_recovery_clears_phantom_pointer(tmp_path):
    """A pointer naming an activation with no event history is
    corrupt — it is cleared rather than served."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    store._db.execute(
        "INSERT INTO serving_pointer(id, activation_id, "
        "artifact_root_digest, backend_id, committed_at, "
        "event_sequence) VALUES(1,?,?,?,?,?)",
        ("ghost" + "0" * 27, "", "", TS, 0))
    store._db.commit()
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    report = sup.recover()
    assert report["cleared_pointer"] is True
    assert report["serving_state"] == "UNAVAILABLE"


def test_recovery_refuses_tampered_log(tmp_path):
    """SEC-205: a modified event digest is detected — recovery refuses
    to trust the altered history."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    _write_history(store, "d4" * 8, ["REQUESTED", "AUTHORIZED"])
    store._db.execute(
        "UPDATE runtime_events SET to_state = 'COMMITTED' "
        "WHERE sequence = 2")
    store._db.commit()
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    from minagi.runtime.authority_store import StoreCorrupt
    with pytest.raises(StoreCorrupt):
        sup.recover()


def test_construction_rejects_unsigned_supervisor(tmp_path):
    registry, _ = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "j" / "authority.sqlite")
    with pytest.raises(ActivationRefused, match="signing identity"):
        ServingSupervisor(store, runtime_signer=None,
                          registry=registry)
