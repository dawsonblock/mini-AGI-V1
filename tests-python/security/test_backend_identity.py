"""v16.4.3 — backend implementation + policy-epoch binding (SEC-206).

A grant naming hf-peft cannot authorize an arbitrary implementation
claiming to be hf-peft: the signed RuntimeBackendManifestV1 binds the
measured module digests and dependency closure, and the supervisor
re-measures before PREPARED.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.backend_manifest import (  # noqa: E402
    BackendRefused, measure_backend, verify_backend_manifest,
    verify_installed_backend)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)

#: a real, importable module for measurement
MODULE = "minagi.v161.strict_schema"


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _manifest_doc(signers, *, modules=(MODULE,), epoch=1):
    m = measure_backend(modules, backend_id="hf-peft", policy_epoch=epoch)
    return m.to_doc(signer=signers["admission"])


def test_measure_backend_produces_stable_digests(tmp_path):
    a = measure_backend((MODULE,), backend_id="hf-peft")
    b = measure_backend((MODULE,), backend_id="hf-peft")
    assert a.implementation_digest == b.implementation_digest
    assert a.dependency_lock_digest == b.dependency_lock_digest


def test_unmeasurable_module_refused(tmp_path):
    with pytest.raises(BackendRefused, match="cannot be resolved"):
        measure_backend(("minagi.does_not_exist",), backend_id="x")


def test_signed_manifest_verifies(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _manifest_doc(signers)
    m = verify_backend_manifest(doc, registry, now=NOW)
    assert m.backend_id == "hf-peft"
    verify_installed_backend(m, module_names=(MODULE,))


def test_unsigned_manifest_refused(tmp_path):
    registry, _signers = _chain(tmp_path)
    m = measure_backend((MODULE,), backend_id="hf-peft")
    doc = {"value": m.to_body(), "digest": digest(m.to_body())}
    with pytest.raises(BackendRefused, match="unsigned"):
        verify_backend_manifest(doc, registry, now=NOW)


def test_wrong_role_signer_refused(tmp_path):
    """Only the admission/runtime role may sign backend manifests."""
    registry, signers = _chain(tmp_path)
    m = measure_backend((MODULE,), backend_id="hf-peft")
    doc = m.to_doc(signer=signers["promotion"])  # wrong role
    with pytest.raises(BackendRefused, match="not an authorized"):
        verify_backend_manifest(doc, registry, now=NOW)


def test_impl_digest_mismatch_refused(tmp_path):
    """The backend changed after admission — the authorized digest no
    longer matches the installed implementation."""
    registry, signers = _chain(tmp_path)
    doc = _manifest_doc(signers, modules=("minagi.v161.strict_schema",))
    m = verify_backend_manifest(doc, registry, now=NOW)
    with pytest.raises(BackendRefused,
                       match="implementation digest"):
        verify_installed_backend(
            m, module_names=("minagi.v161.authority",))


def test_dependency_closure_mismatch_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    m = measure_backend((MODULE,), backend_id="hf-peft",
                        dependency_packages=("pytest",))
    doc = m.to_doc(signer=signers["admission"])
    verified = verify_backend_manifest(doc, registry, now=NOW)
    with pytest.raises(BackendRefused, match="dependency closure"):
        verify_installed_backend(
            verified, module_names=(MODULE,),
            dependency_packages=("nonexistent-pkg-xyz",))


def test_superseded_policy_epoch_refused(tmp_path):
    registry, signers = _chain(tmp_path)
    doc = _manifest_doc(signers, epoch=1)
    m = verify_backend_manifest(doc, registry, now=NOW)
    with pytest.raises(BackendRefused, match="policy epoch"):
        verify_installed_backend(m, module_names=(MODULE,),
                                 min_policy_epoch=2)


def test_supervisor_prepare_enforces_backend_manifest(tmp_path):
    """Integration: when the service is configured with a signed
    backend manifest, prepare() re-measures the installed backend."""
    from minagi.runtime.authority_store import AuthorityStore
    from minagi.runtime.supervisor import (ActivationRefused,
                                           ServingSupervisor)
    from minagi.security.admission_grants import issue_grant
    from minagi.v161.immutable_snapshot import stage_snapshot
    from minagi.v161.artifact_closure import close_tree

    registry, signers = _chain(tmp_path)
    doc = _manifest_doc(signers, modules=("minagi.v161.strict_schema",))
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    sup = ServingSupervisor(
        store, runtime_signer=signers["runtime"], registry=registry,
        backend_manifest_doc=doc,
        backend_modules=("minagi.v161.strict_schema",),
        backend_deps=(), min_policy_epoch=1, now=NOW)

    # stage a measured artifact set
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / "w.bin").write_bytes(b"w")
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    root = digest(digests)
    grant = issue_grant(
        signers["admission"], decision_digest=digest({"d": 1}),
        qualification_digest=digest({"q": 1}),
        runtime_manifest_digest=digest({"m": 1}),
        artifact_root_digest=root, backend_id="hf-peft",
        backend_binary_digest=str(doc["digest"]), policy_epoch=1,
        audience_runtime_identity="local-supervisor", now=NOW)

    class B:
        backend_id = "hf-peft"

        def load(self, snapshot):
            return {"h": 1}

        def health_probe(self, h):
            return None

        def unload(self, h):
            return None

    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, stage_snapshot(
        tmp_path / "snap" / "a",
        {"model": str(model), "adapter": str(adir)},
        expected_digests=digests,
        manifest_digest=digest({"m": 1})))
    act = sup.prepare(aid, B())
    assert act.state.value == "PREPARED"

    # a grant binding a DIFFERENT backend manifest digest is refused
    grant2 = issue_grant(
        signers["admission"], decision_digest=digest({"d": 2}),
        qualification_digest=digest({"q": 2}),
        runtime_manifest_digest=digest({"m": 2}),
        artifact_root_digest=root, backend_id="hf-peft",
        backend_binary_digest=digest({"other": "manifest"}),
        policy_epoch=1, audience_runtime_identity="local-supervisor",
        now=NOW)
    aid2 = sup.request().activation_id
    sup.authorize(aid2, grant2)
    sup.stage(aid2, stage_snapshot(
        tmp_path / "snap" / "b",
        {"model": str(model), "adapter": str(adir)},
        expected_digests=digests,
        manifest_digest=digest({"m": 2})))
    with pytest.raises(ActivationRefused,
                       match="backend manifest digest"):
        sup.prepare(aid2, B())
