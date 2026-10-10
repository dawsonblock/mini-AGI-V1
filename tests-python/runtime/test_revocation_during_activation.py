"""v16.4.5 SEC-403 race tests — revocation may advance between
authorization and traffic. The supervisor re-reads operative evidence
immediately before committing, and already-serving activations are
withdrawn when their promotion decision is revoked.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

import authority_chain as ac  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.serving_router import ServingRouter  # noqa: E402
from minagi.runtime.supervisor import (  # noqa: E402
    ActivationRefused, ActivationState, ServingState,
    ServingSupervisor)
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2)
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402


def _supervisor(tmp_path, registry, signers, rstore, *, router=None):
    store = AuthorityStore(Path(tmp_path) / "journal" / "authority.sqlite")
    sup = ServingSupervisor(
        store, runtime_signer=signers["runtime"], registry=registry,
        now=ac.NOW,
        revocation_snapshot_provider=ac.snapshot_provider(
            rstore, registry))
    sup.router = router or ServingRouter()
    return sup


def _publish(signers, rstore, *, epoch, revoked=(), keys=()):
    rstore.publish(RevocationSnapshotV2(
        epoch=epoch, issued_at=ac.TS, valid_until=ac.TS + 86400,
        revoked_decision_digests=tuple(revoked),
        revoked_key_ids=tuple(keys)).to_doc(
            signer=signers["revocation"]))


def test_epoch_advance_between_authorize_and_commit_refuses(tmp_path):
    """Authorize at epoch 0; epoch advances to 1 during preparation;
    the commit re-check refuses — re-authorize under new evidence."""
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    sup = _supervisor(tmp_path, registry, signers, rstore)

    chain = ac.docs(signers, tmp_path, "a")
    grant = ac.grant_for(signers, chain, epoch=0)
    aid = sup.request().activation_id
    sup.authorize(aid, grant, authority_docs=ac.authority_docs(chain))
    sup.stage(aid, stage_snapshot(
        tmp_path / "snap" / aid, chain["paths"],
        expected_digests=chain["digests"],
        manifest_digest=chain["runtime_manifest"]["digest"]))
    sup.prepare(aid, ac.FakeBackend())
    sup.health_check(aid)

    # revocation state moves forward while the candidate sat READY
    _publish(signers, rstore, epoch=1)
    with pytest.raises(ActivationRefused, match="re-authorized|epoch"):
        sup.commit_activation(aid)
    # nothing routed: the pointer never named the stale candidate
    ptr = sup.store.read_pointer()
    assert ptr is None or ptr["activation_id"] != aid


def test_decision_revoked_during_preparation_refuses_commit(tmp_path):
    """A snapshot listing THIS decision refuses the commit. (Any new
    revoking snapshot also advances the epoch — the epoch-floor check
    fires first and is sufficient; the decision check is the
    defense-in-depth underneath it.)"""
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    sup = _supervisor(tmp_path, registry, signers, rstore)

    chain = ac.docs(signers, tmp_path, "a")
    grant = ac.grant_for(signers, chain, epoch=0)
    aid = sup.request().activation_id
    sup.authorize(aid, grant, authority_docs=ac.authority_docs(chain))
    sup.stage(aid, stage_snapshot(
        tmp_path / "snap" / aid, chain["paths"],
        expected_digests=chain["digests"],
        manifest_digest=chain["runtime_manifest"]["digest"]))
    sup.prepare(aid, ac.FakeBackend())
    sup.health_check(aid)

    _publish(signers, rstore, epoch=1,
             revoked=[chain["decision"]["digest"]])
    with pytest.raises(ActivationRefused):
        sup.commit_activation(aid)
    ptr = sup.store.read_pointer()
    assert ptr is None or ptr["activation_id"] != aid


def test_apply_snapshot_quarantines_active_keeps_valid(tmp_path):
    """Revocation response for serving models: the active model whose
    decision is revoked is withdrawn; a still-valid retained
    predecessor is republished — never a revoked fallback."""
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    router = ServingRouter()
    sup = _supervisor(tmp_path, registry, signers, rstore,
                      router=router)

    first_id, first_chain = ac.drive(
        sup, signers, tmp_path, ac.FakeBackend(), tag="a",
        snapshot_root=tmp_path / "snaps")
    second_id, second_chain = ac.drive(
        sup, signers, tmp_path, ac.FakeBackend(), tag="b",
        snapshot_root=tmp_path / "snaps")
    assert sup.active_id == second_id

    snap = rstore.latest_valid(registry, now=ac.NOW)
    _publish(signers, rstore, epoch=1,
             revoked=[second_chain["decision"]["digest"]])
    snap = rstore.latest_valid(registry, now=ac.NOW)
    actions = sup.apply_revocation_snapshot(snap)
    assert second_id in actions["quarantined"]
    # the still-valid predecessor republished — service survives
    assert sup.active_id == first_id
    assert sup.serving_state is ServingState.SERVING


def test_apply_snapshot_everything_revoked_is_unavailable(tmp_path):
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    sup = _supervisor(tmp_path, registry, signers, rstore)
    _, ca = ac.drive(sup, signers, tmp_path, ac.FakeBackend(),
                     tag="a", snapshot_root=tmp_path / "snaps")
    _, cb = ac.drive(sup, signers, tmp_path, ac.FakeBackend(),
                     tag="b", snapshot_root=tmp_path / "snaps")

    _publish(signers, rstore, epoch=1,
             revoked=[ca["decision"]["digest"],
                      cb["decision"]["digest"]])
    snap = rstore.latest_valid(registry, now=ac.NOW)
    actions = sup.apply_revocation_snapshot(snap)
    assert actions["quarantined"]
    assert sup.serving_state is ServingState.UNAVAILABLE
    dep = sup.store.deployment()
    assert dep["transition_phase"] == "UNAVAILABLE"


def test_rollback_to_revoked_predecessor_refused(tmp_path):
    """Rollback is not a revocation bypass: a retained predecessor
    whose decision is now revoked is refused as a target."""
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path, epoch=0)
    sup = _supervisor(tmp_path, registry, signers, rstore)
    first_id, first_chain = ac.drive(
        sup, signers, tmp_path, ac.FakeBackend(), tag="a",
        snapshot_root=tmp_path / "snaps")
    ac.drive(sup, signers, tmp_path, ac.FakeBackend(), tag="b",
             snapshot_root=tmp_path / "snaps")
    _publish(signers, rstore, epoch=1,
             revoked=[first_chain["decision"]["digest"]])
    with pytest.raises(ActivationRefused):
        sup.rollback(first_id)
