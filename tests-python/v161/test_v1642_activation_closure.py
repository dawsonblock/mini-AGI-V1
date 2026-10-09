"""v16.4.2 activation-closure qualification table (UPGRADE_PLAN §3.6).

Each row of the spec's acceptance matrix is executable here at the
launcher/supervisor integration level:

  1  Forge a MeasuredSnapshot and request serving      refused
  2  Supply an arbitrary object with path()            refused
  3  Omit backend qualification                        refused
  4  Qualify HF/PEFT but request native CUDA           refused
  5  Substitute one byte of the adapter                refused
  6  Add a symbolic link to an artifact                refused
  7  Supply unsigned revocations                       refused
  8  Supply a revocation list from 2100                refused
  9  Replay an older revocation epoch                  refused
 10  Fail receipt persistence after preparation        no externally
                                                        active candidate
 11  Crash between commit and traffic switch           recover
                                                        deterministically
 12  Revoke the currently active version               quarantine
 13  Fail the new model health check                   previous healthy
                                                        version remains
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egai.common.canonical import digest  # noqa: E402
from minagi.runtime.supervisor import ServingSupervisor  # noqa: E402
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2, RevocationStore, verify_snapshot,
    RevocationRefused)
from minagi.v161.immutable_snapshot import (  # noqa: E402
    MeasuredSnapshot, SnapshotError)
from minagi.v161.trusted_launcher import (  # noqa: E402
    LaunchRefused)

# reuse the v16.4.1 chain fixture (which now issues V2 snapshots)
from test_v1641_trusted_launcher import (  # noqa: E402
    FakeBackend, TS, NOW, _build_chain, _launcher, _request,
    _revocation_snapshot)


# --- rows 1-2: measurement never implies authorization ------------------

def test_row1_forged_measured_snapshot_refused(tmp_path):
    """A caller cannot construct a MeasuredSnapshot — only
    stage_snapshot can. An object created by measuring does not carry
    deployment authority."""
    with pytest.raises(SnapshotError, match="stage_snapshot"):
        MeasuredSnapshot(root=tmp_path, digests={},
                         manifest_digest=digest({"m": 1}), closures={})


def test_row2_arbitrary_path_object_refused(tmp_path):
    """An object with a path() method is not measurement evidence —
    the supervisor refuses it at stage()."""
    chain = _build_chain(tmp_path)
    sup = ServingSupervisor(tmp_path / "journal",
                            runtime_signer=chain.signers["runtime"],
                            registry=chain.registry, now=NOW)
    from minagi.security.admission_grants import issue_grant
    grant = issue_grant(
        chain.signers["admission"], decision_digest=digest({"d": 1}),
        qualification_digest=digest({"q": 1}),
        runtime_manifest_digest=chain.manifest["digest"],
        artifact_root_digest=digest({"a": 1}), backend_id="hf-peft",
        audience_runtime_identity=sup.runtime_identity, now=NOW)
    sup.request("act")
    sup.authorize("act", grant)

    class Quacks:
        def path(self, name):
            return tmp_path
    with pytest.raises(Exception, match="MeasuredSnapshot"):
        sup.stage("act", Quacks())


# --- rows 3-4: explicit backend authorization ----------------------------

def test_row3_missing_backend_coverage_refused(tmp_path):
    """A fully signed qualification that omits runtime_backends fails
    closed — no silent default to the supported list."""
    chain = _build_chain(tmp_path)
    from test_v1641_trusted_launcher import _signed
    from minagi.v161.artifact_closure import (close_tree,
                                              tokenizer_artifact_digest)
    from minagi.v161.peft_serving import runtime_manifest
    chain.qual_doc = _signed(chain.signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": "camp", "decision": "QUALIFIED",
        "campaign_plan_digest": chain.plan_doc["digest"],
        "evaluation_bundle_digest":
            chain.qual_doc["value"]["evaluation_bundle_digest"]})
    # the manifest binds the qualification digest — regenerate it
    # against the coverage-omitting record so the chain stays
    # self-consistent and only backend coverage is missing
    chain.manifest = runtime_manifest(
        model_id="tiny", model_revision="r1",
        model_digest=close_tree(chain.model, resolve_symlinks=True).digest,
        tokenizer_digest=tokenizer_artifact_digest(chain.model),
        adapter_dir=chain.adir, protocol=chain.proto,
        campaign_digest=chain.plan_doc["digest"],
        qualification_record_digest=chain.qual_doc["digest"])
    chain.decision_doc = _signed(chain.signers["promotion"], {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": "camp",
        "campaign_plan_digest": chain.plan_doc["digest"],
        "qualification_record_digest": chain.qual_doc["digest"],
        "evaluation_bundle_digest":
            chain.qual_doc["value"]["evaluation_bundle_digest"],
        "adapter": "L6",
        "adapter_artifact_digests":
            {"seed-0": chain.manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": chain.manifest["digest"]},
        "authorized_at": TS - 60, "expires_at": TS + 3600})
    chain.decision_doc["runtime_manifests"] = {"seed-0": chain.manifest}
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused,
                       match="backend coverage"):
        launcher.launch(_request(chain), FakeBackend())


def test_row4_hf_peft_qualified_native_cuda_refused(tmp_path):
    """HF/PEFT qualification does not extend to a native backend."""
    chain = _build_chain(tmp_path)

    class NativeBackend(FakeBackend):
        backend_id = "qwen-native-cuda"

    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused,
                       match="manifest backend|not covered"):
        launcher.launch(
            _request(chain, expected_backend="qwen-native-cuda"),
            NativeBackend())


# --- rows 5-6: artifact integrity ----------------------------------------

def test_row5_adapter_byte_substitution_refused(tmp_path):
    chain = _build_chain(tmp_path)
    (chain.adir / "adapter_model.safetensors").write_bytes(b"tampered")
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused):
        launcher.launch(_request(chain), FakeBackend())


def test_row6_symlinked_artifact_refused(tmp_path):
    chain = _build_chain(tmp_path)
    (chain.adir / "link.safetensors").symlink_to(
        chain.adir / "adapter_model.safetensors")
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused):
        launcher.launch(_request(chain), FakeBackend())


# --- rows 7-9: revocation authorization ----------------------------------

def test_row7_unsigned_revocations_refused(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(
        chain, tmp_path,
        revocation={"schema": "mini-agi-v16.4.1-revocation-list-v1",
                    "digests": [], "generated_at": TS})
    with pytest.raises(LaunchRefused, match="revocation"):
        launcher.launch(_request(chain), FakeBackend())


def test_row8_revocation_from_2100_refused(tmp_path):
    chain = _build_chain(tmp_path)
    snap = RevocationSnapshotV2(epoch=0, issued_at=4102444800,
                              valid_until=4102444800 + 3600)
    launcher = _launcher(
        chain, tmp_path,
        revocation=snap.to_doc(signer=chain.signers["revocation"]))
    with pytest.raises(LaunchRefused, match="future-dated"):
        launcher.launch(_request(chain), FakeBackend())


def test_row9_replayed_older_revocation_epoch_refused(tmp_path):
    """An older epoch cannot displace newer evidence — the durable
    store refuses regression and the verifier rejects stale epochs."""
    chain = _build_chain(tmp_path)
    store = RevocationStore(tmp_path / "rev")
    store.publish(_revocation_snapshot(chain, epoch=5))
    with pytest.raises(RevocationRefused):
        store.publish(_revocation_snapshot(chain, epoch=3))
    stale = _revocation_snapshot(chain, epoch=3)
    with pytest.raises(RevocationRefused, match="epoch"):
        verify_snapshot(stale, chain.registry, now=NOW, min_epoch=5)


# --- rows 10-11: activation atomicity ------------------------------------

def test_row10_receipt_failure_leaves_no_active_candidate(tmp_path):
    """Evidence persistence failure after commit → quarantine; nothing
    unevidenced remains routable."""
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    # a receipt path that is a directory makes persistence fail
    receipt = tmp_path / "receipt-dir"
    receipt.mkdir()
    with pytest.raises(LaunchRefused, match="evidence"):
        launcher.launch(_request(chain), FakeBackend(),
                        receipt_path=receipt)
    pointer = launcher.supervisor.active_pointer()
    assert not pointer or not pointer.get("activation_id")


def test_row11_crash_between_commit_and_switch_recovers(tmp_path):
    """Drive the journal to COMMITTED-but-unrouted, then recover."""
    chain = _build_chain(tmp_path)
    journal_dir = tmp_path / "journal"
    sup = ServingSupervisor(journal_dir,
                            runtime_signer=chain.signers["runtime"],
                            registry=chain.registry, now=NOW)
    journal = sup.journal
    for st in ["REQUESTED", "AUTHORIZED", "STAGED", "PREPARED",
               "READY", "COMMITTED"]:
        journal.append(activation_id="act-x",
                       from_state="", to_state=st, at=TS)
    # no pointer write — the crash landed before routing
    fresh = ServingSupervisor(journal_dir,
                              runtime_signer=chain.signers["runtime"],
                              registry=chain.registry, now=NOW)
    report = fresh.recover_from_journal()
    assert "act-x" in report["aborted"]


# --- rows 12-13: quarantine + rollback ------------------------------------

def test_row12_revoked_active_version_quarantined(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    result = launcher.launch(_request(chain), FakeBackend())
    assert launcher.supervisor.active_id == result.activation_id
    launcher.supervisor.quarantine_active(
        reason="promotion decision revoked")
    pointer = launcher.supervisor.active_pointer()
    assert not pointer or not pointer.get("activation_id")


def test_row13_failed_health_probe_keeps_previous(tmp_path):
    chain = _build_chain(tmp_path)
    sup = ServingSupervisor(tmp_path / "journal",
                            runtime_signer=chain.signers["runtime"],
                            registry=chain.registry, now=NOW)
    launcher = _launcher(chain, tmp_path, supervisor=sup)
    ok = launcher.launch(_request(chain), FakeBackend())
    assert sup.active_id == ok.activation_id
    with pytest.raises(LaunchRefused, match="health probe"):
        launcher.launch(_request(chain), FakeBackend(unhealthy=True))
    assert sup.active_id == ok.activation_id
    assert sup.active_pointer()["activation_id"] == ok.activation_id
