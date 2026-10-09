"""v16.4.2 — ServingSupervisor transactional activation lifecycle.

UPGRADE_PLAN §3.5/§3.6: intent is journaled before the pointer moves,
the pointer swap is atomic, the signed completion lands after, and any
failure aborts and unloads the candidate. Recovery reconciles the
journal against the pointer deterministically.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.durable_journal import (  # noqa: E402
    DurableJournal, JournalError)
from minagi.runtime.supervisor import (  # noqa: E402
    ActivationError, ActivationRefused, ServingSupervisor)
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
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "gpt2"}')
    adir = tmp_path / f"adapter-{tag}"
    adir.mkdir()
    (adir / "adapter_model.safetensors").write_bytes(b"w-" + tag.encode())
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


def _supervisor(tmp_path, registry, signers, *, journal="journal"):
    return ServingSupervisor(
        tmp_path / journal, runtime_signer=signers["runtime"],
        registry=registry, now=NOW)


def _drive(sup, signers, tmp_path, act_id, backend, *, tag="a"):
    digests, paths = _artifacts(tmp_path, tag)
    grant = _grant(signers, digest(digests))
    sup.request(act_id)
    sup.authorize(act_id, grant)
    sup.stage(act_id, _stage(tmp_path, tag, digests, paths))
    sup.prepare(act_id, backend)
    sup.health_check(act_id)
    sup.commit_activation(act_id)
    return grant


# ---------- happy path ---------------------------------------------------

def test_full_lifecycle_commits_atomically(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    backend = FakeBackend()
    _drive(sup, signers, tmp_path, "act-1", backend)

    assert sup.active_id == "act-1"
    pointer = sup.active_pointer()
    assert pointer["activation_id"] == "act-1"
    # intent is durable before the pointer, completion signed after
    hist = sup.journal.history("act-1")
    intent = [r for r in hist if r.detail.get("kind")
              == "activation_intent"]
    completion = [r for r in hist if r.detail.get("kind")
                  == "activation_completion"]
    assert intent and completion
    assert completion[0].signed()
    states = [r.to_state for r in hist]
    assert states.index("COMMITTED") < states.index("ACTIVE")


def test_activation_id_write_once(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    sup.request("act-1")
    with pytest.raises(ActivationRefused, match="already exists"):
        sup.request("act-1")


# ---------- stage / grant gates ------------------------------------------

def test_arbitrary_object_with_path_refused(tmp_path):
    """An object that merely quacks like a snapshot is not measured
    evidence — spec row 2."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    sup.request("act-1")
    sup.authorize("act-1", grant)

    class Quacks:
        def path(self, name):
            return tmp_path

        @property
        def artifact_digests(self):
            return tuple(sorted(digests.items()))

    with pytest.raises(ActivationRefused, match="MeasuredSnapshot"):
        sup.stage("act-1", Quacks())
    assert sup.journal.history("act-1")[-1].to_state == "ABORTED"


def test_grant_artifact_mismatch_aborts(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest({"adapter": "sha256:" + "0" * 64,
                                    "model": "sha256:" + "0" * 64}))
    sup.request("act-1")
    sup.authorize("act-1", grant)
    with pytest.raises(ActivationRefused, match="does not match"):
        sup.stage("act-1", _stage(tmp_path, "a", digests, paths))


def test_wrong_backend_aborts_at_prepare(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests), backend="hf-peft")
    sup.request("act-1")
    sup.authorize("act-1", grant)
    sup.stage("act-1", _stage(tmp_path, "a", digests, paths))

    class OtherBackend(FakeBackend):
        backend_id = "qwen-native-cuda"

    with pytest.raises(ActivationRefused, match="authorized backend"):
        sup.prepare("act-1", OtherBackend())


def test_grant_replay_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    grant = _grant(signers, digest({"x": 1}))
    sup.request("act-1")
    sup.authorize("act-1", grant)
    sup.request("act-2")
    with pytest.raises(ActivationRefused, match="one activation"):
        sup.authorize("act-2", grant)


# ---------- failure injection --------------------------------------------

def test_backend_load_failure_aborts_without_pointer(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    sup.request("act-1")
    sup.authorize("act-1", grant)
    sup.stage("act-1", _stage(tmp_path, "a", digests, paths))
    with pytest.raises(ActivationError, match="failed to load"):
        sup.prepare("act-1", FakeBackend(fail_load=True))
    assert sup.active_id is None
    assert sup.journal.history("act-1")[-1].to_state == "ABORTED"


def test_failed_health_probe_unloads_and_keeps_previous(tmp_path):
    """Spec row: fail the new model health check → previous healthy
    version remains active."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    _drive(sup, signers, tmp_path, "act-1", FakeBackend(), tag="a")
    assert sup.active_id == "act-1"

    bad = FakeBackend(fail_probe=True)
    digests, paths = _artifacts(tmp_path, "b")
    grant = _grant(signers, digest(digests))
    sup.request("act-2")
    sup.authorize("act-2", grant)
    sup.stage("act-2", _stage(tmp_path, "b", digests, paths))
    sup.prepare("act-2", bad)
    with pytest.raises(ActivationError, match="health probe failed"):
        sup.health_check("act-2")
    assert bad.unloaded, "failed candidate must be unloaded"
    assert sup.active_id == "act-1"
    assert sup.active_pointer()["activation_id"] == "act-1"


def test_pointer_write_failure_aborts(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    digests, paths = _artifacts(tmp_path, "a")
    grant = _grant(signers, digest(digests))
    sup.request("act-1")
    sup.authorize("act-1", grant)
    sup.stage("act-1", _stage(tmp_path, "a", digests, paths))
    sup.prepare("act-1", FakeBackend())
    sup.health_check("act-1")

    def boom(doc):
        raise JournalError("disk full")
    sup.journal.write_pointer = boom
    with pytest.raises(ActivationError, match="pointer"):
        sup.commit_activation("act-1")
    assert sup.journal.history("act-1")[-1].to_state == "ABORTED"
    assert sup.active_id is None


def test_completion_record_failure_rolls_back_pointer(tmp_path):
    """If the signed completion cannot be persisted after the pointer
    swap, the candidate is aborted and the pointer restored to the
    previous live runtime — nothing unevidenced stays routable."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    _drive(sup, signers, tmp_path, "act-1", FakeBackend(), tag="a")
    assert sup.active_id == "act-1"

    digests, paths = _artifacts(tmp_path, "b")
    grant = _grant(signers, digest(digests))
    sup.request("act-2")
    sup.authorize("act-2", grant)
    sup.stage("act-2", _stage(tmp_path, "b", digests, paths))
    sup.prepare("act-2", FakeBackend())
    sup.health_check("act-2")

    orig_append = sup.journal.append
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if kw.get("detail", {}).get("kind") == "activation_completion":
            raise JournalError("simulated persistence failure")
        return orig_append(**kw)
    sup.journal.append = flaky

    with pytest.raises(ActivationError, match="completion record"):
        sup.commit_activation("act-2")
    # pointer restored to the previous live runtime
    assert sup.active_id == "act-1"
    assert sup.active_pointer()["activation_id"] == "act-1"


# ---------- quarantine + rollback ----------------------------------------

def test_revoked_active_version_quarantined_and_rolled_back(tmp_path):
    """Spec row: revoke the currently active version → quarantine and
    restore the retained committed predecessor."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    _drive(sup, signers, tmp_path, "act-1", FakeBackend(), tag="a")
    _drive(sup, signers, tmp_path, "act-2", FakeBackend(), tag="b")
    assert sup.active_id == "act-2"

    sup.quarantine_active(reason="promotion revoked")
    assert sup.active_id == "act-1"
    assert sup.active_pointer()["activation_id"] == "act-1"
    hist = sup.journal.history("act-2")
    assert hist[-1].to_state == "QUARANTINED"


def test_rollback_restores_retained_live_runtime(tmp_path):
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    _drive(sup, signers, tmp_path, "act-1", FakeBackend(), tag="a")
    _drive(sup, signers, tmp_path, "act-2", FakeBackend(), tag="b")
    target = sup.rollback()
    assert target == "act-1"
    assert sup.active_id == "act-1"
    hist = sup.journal.history("act-1")
    assert hist[-1].detail.get("kind") == "rollback"
    assert hist[-1].signed()


def test_rollback_refuses_dead_runtime(tmp_path):
    """The pointer must never be restored to a model that is not
    actually loaded — rollback requires a live handle."""
    registry, signers = _chain(tmp_path)
    sup = _supervisor(tmp_path, registry, signers)
    b1 = FakeBackend()
    _drive(sup, signers, tmp_path, "act-1", b1, tag="a")
    _drive(sup, signers, tmp_path, "act-2", FakeBackend(), tag="b")
    # kill the retained runtime's handle out from under the supervisor
    pair = sup._live_handles.pop("act-1")
    b1.unload(pair[1])
    with pytest.raises(ActivationRefused, match="re-admit"):
        sup.rollback()


# ---------- crash recovery -----------------------------------------------

def _journal_write(journal, aid, states, signers=None):
    src = ""
    for st in states:
        journal.append(activation_id=aid, from_state=src,
                       to_state=st, at=TS)
        src = st


def test_recovery_aborts_committed_but_unrouted(tmp_path):
    """Crash between durable intent and pointer swap: the candidate
    never routed → abort on recovery."""
    registry, signers = _chain(tmp_path)
    journal = DurableJournal(tmp_path / "journal")
    _journal_write(journal, "act-1",
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED",
                    "READY", "COMMITTED"])
    journal.write_pointer({"activation_id": "act-0"})
    sup = _supervisor(tmp_path, registry, signers)
    report = sup.recover_from_journal()
    assert "act-1" in report["aborted"]
    assert sup.journal.history("act-1")[-1].to_state == "ABORTED"


def test_recovery_completes_committed_and_routed(tmp_path):
    """Crash after the pointer swap but before the completion record:
    the candidate DID go live → write the reconciled completion."""
    registry, signers = _chain(tmp_path)
    journal = DurableJournal(tmp_path / "journal")
    _journal_write(journal, "act-1",
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED",
                    "READY", "COMMITTED"])
    journal.write_pointer({"activation_id": "act-1"})
    sup = _supervisor(tmp_path, registry, signers)
    report = sup.recover_from_journal()
    assert report["reconciled_completions"] == ["act-1"]
    assert report["active"] == "act-1"
    completion = sup.journal.history("act-1")[-1]
    assert completion.detail["reconciled"] is True
    assert completion.signed()


def test_recovery_aborts_stalled_before_commit(tmp_path):
    registry, signers = _chain(tmp_path)
    journal = DurableJournal(tmp_path / "journal")
    _journal_write(journal, "act-1",
                   ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED"])
    sup = _supervisor(tmp_path, registry, signers)
    report = sup.recover_from_journal()
    assert report["aborted"] == ["act-1"]


def test_recovery_clears_phantom_pointer(tmp_path):
    """A pointer naming an activation with no journal history is
    corrupt — it is cleared rather than served."""
    registry, signers = _chain(tmp_path)
    journal = DurableJournal(tmp_path / "journal")
    journal.write_pointer({"activation_id": "ghost"})
    sup = _supervisor(tmp_path, registry, signers)
    report = sup.recover_from_journal()
    assert report["cleared_pointer"] is True
    assert report["active"] is None


def test_construction_rejects_unsigned_supervisor(tmp_path):
    registry, _ = _chain(tmp_path)
    with pytest.raises(ActivationRefused, match="signing identity"):
        ServingSupervisor(tmp_path / "j", runtime_signer=None,
                          registry=registry)
