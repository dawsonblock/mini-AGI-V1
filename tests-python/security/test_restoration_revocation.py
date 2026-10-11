"""v16.4.5 G1 — revoked promotions cannot restore (SEC-403).

End-to-end: real authority chain -> activate -> revoke -> crash ->
recovery. Every revoked path must fail closed; never a phantom
active pointer, never authority synthesized from journaled digests.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

import authority_chain as ac  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.recovery_manager import RecoveryManager  # noqa: E402
from minagi.runtime.supervisor import (  # noqa: E402
    ServingState, ServingSupervisor)
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationRefused, RevocationSnapshotV2)


def _supervisor(tmp_path, registry, signers, rstore):
    store = AuthorityStore(Path(tmp_path) / "journal" / "authority.sqlite")
    return ServingSupervisor(
        store, runtime_signer=signers["runtime"], registry=registry,
        now=ac.NOW, revocation_snapshot_provider=ac.snapshot_provider(
            rstore, registry))


def _manager(sup2, signers, registry, snaproot, rstore):
    return RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={"hf-peft": ac.FakeBackend},
        revocation_store=rstore, now=ac.NOW)


def _activate(tmp_path, registry, signers, rstore, tag, snaproot):
    sup = _supervisor(tmp_path, registry, signers, rstore)
    return ac.drive(sup, signers, tmp_path, ac.FakeBackend(), tag=tag,
                    snapshot_root=snaproot)


def _revoke(signers, rstore, *, decision_digests, epoch, keys=()):
    rstore.publish(RevocationSnapshotV2(
        epoch=epoch, issued_at=ac.TS, valid_until=ac.TS + 86400,
        revoked_decision_digests=tuple(decision_digests),
        revoked_key_ids=tuple(keys)).to_doc(
            signer=signers["revocation"]))


def test_revoked_promotion_cannot_restore(tmp_path):
    """The original promotion decision is revoked before restart —
    restoration must refuse, permanently, and the deployment goes
    durably UNAVAILABLE."""
    registry, signers = ac.trust_chain(tmp_path)
    snaproot = tmp_path / "snaps"
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    aid, chain = _activate(tmp_path, registry, signers, rstore, "a",
                           snaproot)
    assert aid
    _revoke(signers, rstore, epoch=1,
            decision_digests=[chain["decision"]["digest"]])

    sup2 = _supervisor(tmp_path, registry, signers, rstore)
    report = _manager(sup2, signers, registry, snaproot,
                      rstore).restore()
    assert report["restoration"]["status"] == "unavailable"
    attempt = report["restoration"]["attempts"][0]
    assert attempt["permanent"] is True
    assert "REVOKED" in attempt["reason"]
    assert sup2.serving_state is ServingState.UNAVAILABLE
    # no phantom pointer: nothing may name a nonexistent active model
    dep = sup2.store.deployment()
    assert dep["transition_phase"] == "UNAVAILABLE"


def test_revoked_signing_key_invalidates_chain(tmp_path):
    """The promotion signer's key is revoked — evidence it signed no
    longer authorizes anything, including restoration."""
    registry, signers = ac.trust_chain(tmp_path)
    snaproot = tmp_path / "snaps"
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    aid, chain = _activate(tmp_path, registry, signers, rstore, "a",
                           snaproot)
    _revoke(signers, rstore, epoch=1, decision_digests=[],
            keys=[chain["decision"]["signer_key_id"]])

    sup2 = _supervisor(tmp_path, registry, signers, rstore)
    report = _manager(sup2, signers, registry, snaproot,
                      rstore).restore()
    assert report["restoration"]["status"] == "unavailable"
    assert "key" in report["restoration"]["attempts"][0]["reason"]


def test_revoked_current_restores_earlier_valid_model(tmp_path):
    """Current model's decision revoked; an earlier completed model's
    chain is still valid — recovery restores the EARLIER one only."""
    registry, signers = ac.trust_chain(tmp_path)
    snaproot = tmp_path / "snaps"
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    first_id, first_chain = _activate(
        tmp_path, registry, signers, rstore, "a", snaproot)
    desired_id, desired_chain = _activate(
        tmp_path, registry, signers, rstore, "b", snaproot)
    _revoke(signers, rstore, epoch=1,
            decision_digests=[desired_chain["decision"]["digest"]])

    sup2 = _supervisor(tmp_path, registry, signers, rstore)
    report = _manager(sup2, signers, registry, snaproot,
                      rstore).restore()
    assert report["restoration"]["status"] == "restored"
    assert report["restoration"]["restored_from"] == first_id
    # the revoked attempt is recorded as a permanent refusal
    attempts = {a["candidate"]: a
                for a in report["restoration"]["attempts"]}
    assert attempts[desired_id]["permanent"] is True
    assert "REVOKED" in attempts[desired_id]["reason"]


def test_every_candidate_revoked_is_unavailable(tmp_path):
    registry, signers = ac.trust_chain(tmp_path)
    snaproot = tmp_path / "snaps"
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    _, ca = _activate(tmp_path, registry, signers, rstore, "a", snaproot)
    _, cb = _activate(tmp_path, registry, signers, rstore, "b", snaproot)
    _revoke(signers, rstore, epoch=1,
            decision_digests=[ca["decision"]["digest"],
                              cb["decision"]["digest"]])

    sup2 = _supervisor(tmp_path, registry, signers, rstore)
    report = _manager(sup2, signers, registry, snaproot,
                      rstore).restore()
    assert report["restoration"]["status"] == "unavailable"
    assert sup2.serving_state is ServingState.UNAVAILABLE
    dep = sup2.store.deployment()
    assert dep["transition_phase"] == "UNAVAILABLE"


def test_epoch_regression_refused_by_store(tmp_path):
    """Publishing an older-epoch snapshot over newer evidence is
    refused — revocation state never moves backward."""
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=2)
    with pytest.raises(RevocationRefused, match="regress"):
        rstore.publish(RevocationSnapshotV2(
            epoch=1, issued_at=ac.TS, valid_until=ac.TS + 86400).to_doc(
                signer=signers["revocation"]))


def test_absent_revocation_evidence_fails_closed(tmp_path):
    """A recovery manager with no revocation store must NOT restore —
    missing revocation evidence is refusal, not 'nothing revoked'."""
    registry, signers = ac.trust_chain(tmp_path)
    snaproot = tmp_path / "snaps"
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    aid, _ = _activate(tmp_path, registry, signers, rstore, "a",
                       snaproot)
    assert aid
    sup2 = _supervisor(tmp_path, registry, signers, rstore)
    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={"hf-peft": ac.FakeBackend},
        revocation_store=None, now=ac.NOW)
    report = mgr.restore()
    assert report["restoration"]["status"] == "unavailable"
    assert "revocation" in \
        report["restoration"]["attempts"][0]["reason"]
