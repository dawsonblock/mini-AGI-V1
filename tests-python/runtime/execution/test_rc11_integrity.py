import numpy as np
import pytest

from kvcontinual.execution.attention.relocation import RopeSpec, rope_apply, relocate_rope_keys
from kvcontinual.execution.authority import Ed25519ReceiptSigner
from kvcontinual.execution.cache.block import ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.qualification import QualificationPolicy
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.registry import AdapterRegistry
from kvcontinual.execution.replay import ReplayItem, ReplayStore
from kvcontinual.execution.source import SourceSegment, SourceSegmentStore
from kvcontinual.execution.types import AssemblyTopology, ExecutionIdentity, ModelIdentity


def _identity():
    return ExecutionIdentity(ModelIdentity("base", "adapter", "tok", "qwen", "rope"), "abi", "gdn", "layout", "fp16", "fp16")


def test_source_segment_id_is_append_only():
    store = SourceSegmentStore(":memory:")
    s = SourceSegment([1, 2, 3], "doc", id="stable", tokenizer_digest="sha256:tok")
    store.put(s)
    store.put(s)  # idempotent retry
    with pytest.raises(ValueError, match="cannot be rewritten"):
        store.put(SourceSegment([9], "doc", id="stable", tokenizer_digest="sha256:tok"))


def test_source_content_digest_binds_tokenizer_identity():
    a = SourceSegment([7, 8], "doc", tokenizer_digest="sha256:a")
    b = SourceSegment([7, 8], "doc", tokenizer_digest="sha256:b")
    assert a.token_digest == b.token_digest
    assert a.content_digest != b.content_digest


def test_contiguous_interior_slice_is_not_exact_prefix():
    store = SourceSegmentStore(":memory:")
    a = SourceSegment([1], "doc", canonical_stream="s", canonical_start=10, canonical_end=11)
    b = SourceSegment([2], "doc", canonical_stream="s", canonical_start=11, canonical_end=12)
    store.put(a); store.put(b)
    assert store.classify_topology([a.id, b.id]) == AssemblyTopology.CANONICAL


def test_execution_artifact_source_digest_cannot_alias():
    identity = _identity()
    tail = RecurrentTailArtifact(0, 0, 8, AffineSummary(np.eye(2), np.zeros((2, 2))))
    store = ExecutionArtifactStore()
    a = ExecutionArtifact("s", identity, 8, {(0, 0): tail}, source_content_digest="sha256:a")
    store.put(a)
    assert store.get("s", identity, "sha256:a") is a
    assert store.get("s", identity, "sha256:b") is None
    b = ExecutionArtifact("s", identity, 8, {(0, 0): tail}, source_content_digest="sha256:b")
    with pytest.raises(ValueError, match="different source content"):
        store.put(b)


def test_qwen_style_partial_split_half_rope_relocation():
    rng = np.random.default_rng(11)
    t, h, head_dim, rope_dim = 7, 2, 12, 8
    raw = rng.normal(size=(t, h, head_dim)).astype(np.float64)
    spec = RopeSpec(head_dim=head_dim, rope_dim=rope_dim, pairing_layout="split_half")
    inv = 1.0 / (10000.0 ** (np.arange(0, rope_dim, 2) / rope_dim))
    old = np.arange(t)
    new = np.array([0, 2, 5, 9, 10, 20, 21])
    old_rot = rope_apply(raw, old, inv, spec)
    relocated = relocate_rope_keys(old_rot, old, new, inv, spec)
    fresh = rope_apply(raw, new, inv, spec)
    assert np.allclose(relocated, fresh, rtol=1e-12, atol=1e-12)
    assert np.array_equal(relocated[..., rope_dim:], raw[..., rope_dim:])


def test_adapter_mutation_after_qualification_is_rejected(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "adapter.safetensors"
    f.write_bytes(b"qualified")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q)
    reg.stage_cache_namespace(c.candidate_id, "sha256:ns")
    candidate_file = tmp_path / "registry" / "candidates" / c.candidate_id / "adapter.safetensors"
    candidate_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed after registration"):
        reg.promote(c.candidate_id)


def test_signed_promotion_receipt_is_verified(tmp_path):
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="test-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)
    f = tmp_path / "adapter.safetensors"
    f.write_bytes(b"candidate")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q, signer=signer)
    reg.stage_cache_namespace(c.candidate_id, "sha256:ns")
    with pytest.raises(RuntimeError, match="required"):
        reg.promote(c.candidate_id)
    promoted = reg.promote(c.candidate_id, verifier=verifier, signer=signer)
    assert promoted.exists()


def test_replay_sampling_defaults_to_unique_items():
    replay = ReplayStore()
    for i in range(8):
        replay.add(ReplayItem({"i": i}, importance=1.0 if i == 0 else .2))
    out = replay.sample(8, seed=7)
    assert len(out) == 8
    assert len({id(x) for x in out}) == 8


def test_signed_execution_artifact_receipt_binds_source_and_identity():
    pytest.importorskip("cryptography")
    from kvcontinual.execution.artifact_receipts import body_for_artifact, issue_artifact_receipt, verify_artifact_receipt
    identity = _identity()
    tail = RecurrentTailArtifact(0, 0, 8, AffineSummary(np.eye(2), np.zeros((2, 2))))
    artifact = ExecutionArtifact(
        "s", identity, 8, {(0, 0): tail},
        capture_manifest_digest="sha256:capture",
        source_content_digest="sha256:source",
    )
    signer = Ed25519ReceiptSigner.generate(key_id="artifact-key")
    body = body_for_artifact(
        artifact,
        artifact_digest="sha256:artifact",
        backend_identity="transformers-mps",
        hardware_fingerprint="apple-silicon-test",
        kernel_build_digest="sha256:kernel",
        oracle_result_digest="sha256:oracle",
        policy_generation="rc11-policy-1",
        numerical_tolerance=1e-4,
    )
    receipt = issue_artifact_receipt(signer, body)
    assert verify_artifact_receipt(signer.verifier(), receipt, body)
    tampered = body.__class__(**{**body.__dict__, "source_content_digest": "sha256:other"})
    assert not verify_artifact_receipt(signer.verifier(), receipt, tampered)


def test_dense_affine_storage_is_guarded_for_large_models():
    from kvcontinual.execution.recurrent.storage_policy import dense_affine_storage_estimate, production_storage_allowed
    estimate = dense_affine_storage_estimate(heads=32, state_dim=128, layers=30, dtype="fp16")
    assert estimate.mib_per_segment == pytest.approx(60.0)
    assert not production_storage_allowed(estimate)


def test_signed_json_rejects_non_finite_values():
    pytest.importorskip("cryptography")
    from kvcontinual.execution.authority import Ed25519ReceiptSigner
    signer = Ed25519ReceiptSigner.generate(key_id="finite-only")
    with pytest.raises(ValueError, match="non-finite"):
        signer.issue({"score": float("nan")})


def test_registry_rejects_candidate_id_path_traversal(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    with pytest.raises(ValueError, match="unsafe candidate_id"):
        reg.stage_cache_namespace("../escape", "sha256:any")


def test_manifest_artifact_path_cannot_escape_candidate(tmp_path):
    import json
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "adapter.safetensors"
    f.write_bytes(b"candidate")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    mpath = tmp_path / "registry" / "candidates" / c.candidate_id / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["artifact_relpath"] = "../../outside.bin"
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe artifact_relpath|escapes"):
        reg._verified_candidate_binding(c.candidate_id)


def test_signed_rollback_reverifies_prior_authority_and_namespace(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="rollback-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)

    def add(payload: bytes):
        f = tmp_path / (payload.decode() + ".safetensors")
        f.write_bytes(payload)
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q, signer=signer)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + payload.hex())
        reg.promote(c.candidate_id, verifier=verifier, signer=signer)
        return c

    c1 = add(b"one")
    c2 = add(b"two")
    assert reg.verify_current(verifier=verifier)["current"] == c2.candidate_id

    ns = tmp_path / "registry" / "cache_namespaces" / c1.candidate_id / "namespace.json"
    data = json.loads(ns.read_text())
    data["namespace_digest"] = "sha256:tampered"
    ns.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="production binding receipt is invalid"):
        reg.rollback(verifier=verifier, signer=signer)


def test_signed_current_state_detects_metadata_tampering(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="current-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)
    f = tmp_path / "adapter.safetensors"; f.write_bytes(b"candidate")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q, signer=signer)
    reg.stage_cache_namespace(c.candidate_id, "sha256:ns")
    reg.promote(c.candidate_id, verifier=verifier, signer=signer)
    state_path = tmp_path / "registry" / "meta" / "production.json"
    state = json.loads(state_path.read_text())
    state["adapter_digest"] = "sha256:forged"
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="production state adapter digest mismatch"):
        reg.verify_current(verifier=verifier)


def test_manifest_artifact_path_rejects_symlink(tmp_path):
    import json, os
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "adapter.safetensors"; f.write_bytes(b"candidate")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    cdir = tmp_path / "registry" / "candidates" / c.candidate_id
    target = cdir / "inside.bin"; target.write_bytes(b"candidate")
    link = cdir / "link.bin"
    try:
        os.symlink(target.name, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    mpath = cdir / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["artifact_relpath"] = "link.bin"
    manifest["adapter_digest"] = "sha256:" + __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="symlink"):
        reg._verified_candidate_binding(c.candidate_id)

def test_artifact_receipt_rejects_nonfinite_tolerance():
    from kvcontinual.execution.artifact_receipts import ArtifactReceiptBody
    body = ArtifactReceiptBody(
        source_content_digest="sha256:s", execution_identity_digest="sha256:e",
        model_weights_digest="sha256:m", tokenizer_digest="sha256:t",
        capture_manifest_digest="sha256:c", artifact_digest="sha256:a",
        algorithm="HYPIC_SEAM8", seam_width=8, numerical_tolerance=float("nan"),
        backend_identity="mps", hardware_fingerprint="apple", kernel_build_digest="sha256:k",
        oracle_result_digest="sha256:o", policy_generation="p1",
    )
    with pytest.raises(ValueError, match="invalid seam/tolerance"):
        body.validate()


def test_signed_rollback_requires_fresh_signing_authority(tmp_path):
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="transition-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)

    def add(name: str):
        f = tmp_path / f"{name}.safetensors"
        f.write_bytes(name.encode())
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"name": name})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q, signer=signer)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + name)
        reg.promote(c.candidate_id, verifier=verifier, signer=signer)
        return c

    c1 = add("one")
    add("two")
    with pytest.raises(RuntimeError, match="signed rollback authority is required"):
        reg.rollback(verifier=verifier)
    assert reg.rollback(verifier=verifier, signer=signer) == c1.candidate_id
    state = reg.verify_current(verifier=verifier)
    assert state["generation"] == 3
    assert state["transition_receipt"]["body"]["action"] == "rollback"


def test_replayed_old_production_state_is_rejected_by_transition_head(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="replay-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)

    def add(name: str):
        f = tmp_path / f"{name}.safetensors"
        f.write_bytes(name.encode())
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"name": name})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q, signer=signer)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + name)
        reg.promote(c.candidate_id, verifier=verifier, signer=signer)
        return c

    add("one")
    state_path = tmp_path / "registry" / "meta" / "production.json"
    stale = json.loads(state_path.read_text())
    add("two")
    state_path.write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="generation does not match transition ledger|transition digest mismatch"):
        reg.verify_current(verifier=verifier)


def test_transition_ledger_tampering_is_detected(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="ledger-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)
    f = tmp_path / "a.safetensors"
    f.write_bytes(b"a")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"lr": 1e-5})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q, signer=signer)
    reg.stage_cache_namespace(c.candidate_id, "sha256:ns")
    reg.promote(c.candidate_id, verifier=verifier, signer=signer)
    ledger = tmp_path / "registry" / "meta" / "production_transitions.jsonl"
    record = json.loads(ledger.read_text().strip())
    record["body"]["generation"] = 99
    ledger.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="generation is not monotonic|transition receipt is invalid"):
        reg.verify_current(verifier=verifier)


def _sha_text(text: str) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _passing_bundle(*, model: str, tokenizer: str, execution: str):
    import hashlib, json
    from kvcontinual.execution.qualification_bundle import build_qualification_bundle
    from kvcontinual.execution.qualification_harness import MacQualificationThresholds, QualificationObservation
    probe = {
        "schema_version": 3,
        "platform": "darwin",
        "machine": "arm64",
        "apple_silicon": True,
        "metal_available": True,
        "mps_available": True,
        "xcodebuild": "Xcode test",
    }
    raw = json.dumps(probe, sort_keys=True, separators=(",", ":")).encode()
    probe["hardware_toolchain_fingerprint"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return build_qualification_bundle(
        hardware_probe=probe,
        capture_manifest={"layers": [0]},
        observations=[QualificationObservation("c", 0.0, 1.0, True)],
        model_weights_digest=model,
        tokenizer_digest=tokenizer,
        kernel_build_digest=_sha_text("kernel"),
        execution_identity_digest=execution,
        runtime_build_digest=_sha_text("runtime"),
        thresholds=MacQualificationThresholds(min_cases=1),
    ).__dict__


def test_crash_window_production_snapshot_is_recoverable_from_ledger(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="recover-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)

    def add(name: str):
        f = tmp_path / f"{name}.safetensors"; f.write_bytes(name.encode())
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"name": name})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q, signer=signer)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + name)
        reg.promote(c.candidate_id, verifier=verifier, signer=signer)
        return c

    add("one")
    state_path = tmp_path / "registry" / "meta" / "production.json"
    stale = json.loads(state_path.read_text())
    second = add("two")
    promoted = tmp_path / "registry" / "promoted" / second.candidate_id
    (promoted / "promotion_receipt.json").unlink()
    import shutil
    shutil.rmtree(promoted / "promotion_receipts")
    state_path.write_text(json.dumps(stale))
    with pytest.raises(RuntimeError, match="generation does not match transition ledger|transition digest mismatch"):
        reg.verify_current(verifier=verifier)
    recovered = reg.recover_current(verifier=verifier)
    assert recovered["current"] == second.candidate_id
    assert recovered["generation"] == 2
    assert (promoted / "promotion_receipt.json").exists()


def test_external_transition_anchor_enforces_monotonic_floor_and_tail(tmp_path):
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="anchor-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"x": 1})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q, signer=signer)
    reg.stage_cache_namespace(c.candidate_id, "sha256:a")
    reg.promote(c.candidate_id, verifier=verifier, signer=signer)
    anchor = reg.transition_anchor(verifier=verifier)
    assert reg.verify_current(
        verifier=verifier,
        minimum_generation=anchor["generation"],
        expected_tail_digest=anchor["transition_record_digest"],
    )["current"] == c.candidate_id
    with pytest.raises(RuntimeError, match="below the external monotonic floor"):
        reg.verify_current(verifier=verifier, minimum_generation=2)
    with pytest.raises(RuntimeError, match="external anchor"):
        reg.verify_current(verifier=verifier, expected_tail_digest="sha256:stale")


def test_unsigned_transition_chain_checks_candidate_continuity(tmp_path):
    import json
    reg = AdapterRegistry(str(tmp_path / "registry"))
    for name in ("one", "two"):
        f = tmp_path / f"{name}.safetensors"; f.write_bytes(name.encode())
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"name": name})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + name)
        reg.promote(c.candidate_id)
    ledger = tmp_path / "registry" / "meta" / "production_transitions.jsonl"
    records = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    records[1]["body"]["previous_candidate_id"] = "forged"
    ledger.write_text("\n".join(json.dumps(x) for x in records) + "\n")
    with pytest.raises(RuntimeError, match="candidate continuity"):
        reg.verify_transition_chain()


def test_transition_registry_identity_is_bound_even_without_signatures(tmp_path):
    import json
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"x": 1})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q)
    reg.stage_cache_namespace(c.candidate_id, "sha256:a")
    reg.promote(c.candidate_id)
    ledger = tmp_path / "registry" / "meta" / "production_transitions.jsonl"
    record = json.loads(ledger.read_text().strip())
    record["body"]["registry_id"] = "different-registry"
    ledger.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="registry identity mismatch"):
        reg.verify_transition_chain()


def test_qualification_record_is_immutable(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"x": 1})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q)
    with pytest.raises(RuntimeError, match="immutable"):
        reg.write_qualification(q)


def test_promoted_cache_namespace_is_frozen(tmp_path):
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"x": 1})
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    reg.write_qualification(q)
    reg.stage_cache_namespace(c.candidate_id, "sha256:a")
    reg.promote(c.candidate_id)
    reg.stage_cache_namespace(c.candidate_id, "sha256:a")  # idempotent
    with pytest.raises(RuntimeError, match="immutable"):
        reg.stage_cache_namespace(c.candidate_id, "sha256:changed")


def test_required_qualification_bundle_binds_model_tokenizer_and_execution_identity(tmp_path):
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="bundle-key")
    verifier = signer.verifier()
    model = _sha_text("model"); tok = _sha_text("tok"); execution = _sha_text("exec")
    reg = AdapterRegistry(
        str(tmp_path / "registry"),
        require_signed_promotions=True,
        require_qualification_bundle=True,
    )
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(
        str(f), model, _sha_text("data"), {"x": 1},
        tokenizer_digest=tok, execution_identity_digest=execution,
    )
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    with pytest.raises(RuntimeError, match="bundle is required"):
        reg.write_qualification(q, signer=signer)
    bundle = _passing_bundle(model=model, tokenizer=tok, execution=execution)
    reg.write_qualification(q, signer=signer, qualification_bundle=bundle)
    reg.stage_cache_namespace(c.candidate_id, _sha_text("namespace"))
    reg.promote(c.candidate_id, verifier=verifier, signer=signer)
    state = reg.verify_current(verifier=verifier)
    assert state["qualification_bundle_digest"] == bundle["bundle_digest"]


def test_qualification_bundle_identity_mismatch_is_rejected(tmp_path):
    model = _sha_text("model"); tok = _sha_text("tok"); execution = _sha_text("exec")
    reg = AdapterRegistry(str(tmp_path / "registry"))
    f = tmp_path / "a.safetensors"; f.write_bytes(b"a")
    c = reg.register_candidate(
        str(f), model, _sha_text("data"), {"x": 1},
        tokenizer_digest=tok, execution_identity_digest=execution,
    )
    q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
    bad = _passing_bundle(model=model, tokenizer=tok, execution=_sha_text("other-exec"))
    with pytest.raises(RuntimeError, match="execution identity"):
        reg.write_qualification(q, qualification_bundle=bad)


def test_promotion_uses_ledger_predecessor_when_snapshot_is_stale(tmp_path):
    import json
    pytest.importorskip("cryptography")
    signer = Ed25519ReceiptSigner.generate(key_id="stale-promote-key")
    verifier = signer.verifier()
    reg = AdapterRegistry(str(tmp_path / "registry"), require_signed_promotions=True)

    def prepare(name: str):
        f = tmp_path / f"{name}.safetensors"; f.write_bytes(name.encode())
        c = reg.register_candidate(str(f), "sha256:base", "sha256:data", {"name": name})
        q = QualificationPolicy().evaluate(c.candidate_id, .04, .005, True, True)
        reg.write_qualification(q, signer=signer)
        reg.stage_cache_namespace(c.candidate_id, "sha256:" + name)
        return c

    one = prepare("one"); reg.promote(one.candidate_id, verifier=verifier, signer=signer)
    state_path = tmp_path / "registry" / "meta" / "production.json"
    stale = json.loads(state_path.read_text())
    two = prepare("two"); reg.promote(two.candidate_id, verifier=verifier, signer=signer)
    state_path.write_text(json.dumps(stale))
    three = prepare("three"); reg.promote(three.candidate_id, verifier=verifier, signer=signer)
    state = reg.verify_current(verifier=verifier)
    assert state["current"] == three.candidate_id
    assert state["previous"] == two.candidate_id
    assert state["generation"] == 3
