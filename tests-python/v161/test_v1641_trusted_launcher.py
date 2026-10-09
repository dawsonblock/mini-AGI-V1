"""v16.4.1 — TrustedRuntimeLauncher: admission is mandatory.

The v16.4.0 audit could not establish that any model-loading path
passed through admission. These tests exercise the enforcement point:
every launch verifies the signed chain, physically measures the
artifacts, stages an immutable snapshot, loads only from that snapshot,
and emits a signed production receipt only after a successful load.
Refusals cover substituted bytes, symlinks, stale revocation evidence,
backend mismatch, load failure, and forged receipts.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2)
from minagi.v161.artifact_closure import (close_tree,  # noqa: E402
                                          tokenizer_artifact_digest)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityLedger,  # noqa: E402
                                   AuthorityRegistry, write_trust_root)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import (PeftServingBackend,  # noqa: E402
                                      runtime_manifest)
from minagi.v161.runtime_admission import (  # noqa: E402
    AdmissionRefused, RevocationList, RuntimeAdmissionController,
    check_activation_receipt)
from minagi.v161.trusted_launcher import (  # noqa: E402
    LaunchRefused, LaunchRequest, TrustedRuntimeLauncher)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())


def _signed(signer, value):
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


class FakeBackend:
    """Test double for a serving backend. `mutate` runs inside load(),
    modelling a writer racing the launcher after verification. The
    v16.4.2 supervised protocol adds health_probe + unload."""

    backend_id = "hf-peft"

    def __init__(self, *, fail=False, mutate=None, unhealthy=False):
        self.fail = fail
        self.mutate = mutate
        self.unhealthy = unhealthy
        self.seen = None
        self.unloaded = []

    def load(self, snapshot):
        if self.fail:
            raise RuntimeError("backend exploded")
        if self.mutate is not None:
            self.mutate()
        self.seen = snapshot
        cfg = (snapshot.path("adapter") / "adapter_config.json").read_text()
        handle = {"config": json.loads(cfg),
                  "loaded_from": str(snapshot.root)}
        self._handle = handle
        return handle

    def health_probe(self, handle):
        if self.unhealthy:
            raise RuntimeError("probe failed: model did not answer")

    def unload(self, handle):
        self.unloaded.append(handle)


def _build_chain(tmp_path, *, expires_at=None, generated_at=None):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}

    adir = storage / "adapters" / "camp" / "L6" / "seed-0"
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4,
        "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": ["c_attn"]}))
    (adir / "adapter_model.safetensors").write_bytes(b"adapter-weights")

    model = storage / "models" / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type": "gpt2"}')
    (model / "model.safetensors").write_bytes(b"base-model-weights")
    # the signed plan binds the tokenizer as the tokenizer-named files
    # inside the model snapshot (physical_identity_digests convention)
    (model / "tokenizer.json").write_text('{"vocab": []}')
    (model / "tokenizer_config.json").write_text('{"chat_template": "x"}')
    # an operator's tokenizer-only cross-check root (same files)
    tokenizer = storage / "models" / "tok"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text('{"vocab": []}')
    (tokenizer / "tokenizer_config.json").write_text(
        '{"chat_template": "x"}')

    proto = ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48},
    })
    plan_doc = _signed(signers["plan"], {
        "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
        "campaign_id": "camp",
        "experiment_protocol_digest": proto.digest})
    bundle_d = digest({"bundle": 1})
    qual_doc = _signed(signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": "camp", "decision": "QUALIFIED",
        "campaign_plan_digest": plan_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "runtime_backends": ["hf-peft"]})

    manifest = runtime_manifest(
        model_id="tiny", model_revision="r1",
        model_digest=close_tree(model, resolve_symlinks=True).digest,
        tokenizer_digest=tokenizer_artifact_digest(model),
        adapter_dir=adir, protocol=proto,
        campaign_digest=plan_doc["digest"],
        qualification_record_digest=qual_doc["digest"])

    decision = _signed(signers["promotion"], {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": "camp",
        "campaign_plan_digest": plan_doc["digest"],
        "qualification_record_digest": qual_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "adapter": "L6",
        "adapter_artifact_digests": {"seed-0": manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": manifest["digest"]},
        "authorized_at": TS - 60,
        "expires_at": TS + 3600 if expires_at is None else expires_at})
    decision["runtime_manifests"] = {"seed-0": manifest}

    return SimpleNamespace(
        storage=storage, registry=registry, signers=signers, adir=adir,
        model=model, tokenizer=tokenizer, proto=proto, plan_doc=plan_doc,
        qual_doc=qual_doc, manifest=manifest, decision_doc=decision)


def _revocation(chain, *, generated_at=None):
    return RevocationList(digests=(), generated_at=(
        TS if generated_at is None else generated_at))


def _revocation_snapshot(chain, *, revoked=(), epoch=0,
                         issued_at=None, valid_until=None):
    """A signed RevocationSnapshotV2 — the authenticated revocation
    evidence the v16.4.2 production path requires."""
    snap = RevocationSnapshotV2(
        epoch=epoch, issued_at=TS - 60 if issued_at is None else issued_at,
        valid_until=TS + 86400 if valid_until is None else valid_until,
        revoked_decision_digests=tuple(str(d) for d in revoked))
    return snap.to_doc(signer=chain.signers["revocation"])


def _launcher(chain, tmp_path, *, revocation=None, signer=None, **kw):
    from minagi.runtime.authority_store import AuthorityStore
    return TrustedRuntimeLauncher(
        chain.registry,
        revocation_snapshot=revocation if revocation is not None
        else _revocation_snapshot(chain),
        runtime_signer=signer or chain.signers["runtime"],
        admission_signer=chain.signers["admission"],
        snapshot_root=tmp_path / "snapshots",
        receipts_dir=tmp_path / "receipts",
        authority_store=kw.pop("authority_store", None) or
        AuthorityStore(tmp_path / "state" / "authority.sqlite"),
        nonce_journal=tmp_path / "nonces.jsonl",
        ledger_path=kw.pop("ledger_path", tmp_path / "ledger.jsonl"),
        now=NOW, **kw)


def _request(chain, **overrides):
    kw = dict(campaign_id="camp", seed="seed-0",
              decision_doc=chain.decision_doc,
              qualification_doc=chain.qual_doc,
              plan_doc=chain.plan_doc, runtime_manifest=chain.manifest,
              adapter_dir=str(chain.adir), model_path=str(chain.model),
              tokenizer_path=str(chain.tokenizer),
              expected_backend="hf-peft")
    kw.update(overrides)
    return LaunchRequest(**kw)


def _ledger(chain, tmp_path):
    ledger = AuthorityLedger(tmp_path / "ledger.jsonl")
    ledger.append(chain.signers["plan"], "experiment_preregistration",
                  {"plan": chain.plan_doc["digest"]})
    return ledger


# ---------- happy path ---------------------------------------------------

def test_launch_admits_stages_loads_and_emits_production_receipt(tmp_path):
    chain = _build_chain(tmp_path)
    ledger = _ledger(chain, tmp_path)
    launcher = _launcher(chain, tmp_path, ledger_path=ledger.path)
    backend = FakeBackend()
    result = launcher.launch(_request(chain), backend)
    out = Path(result.receipt_path)

    assert result.backend_id == "hf-peft"
    assert backend.seen is result.snapshot
    receipt = result.receipt
    assert receipt.production
    assert receipt.candidate_digest == chain.manifest["adapter_digest"]
    assert dict(receipt.loaded_artifact_digests) == {
        "model": chain.manifest["model_digest"],
        "tokenizer": chain.manifest["tokenizer_digest"],
        "adapter": chain.manifest["adapter_digest"]}
    assert receipt.activation_nonce and len(receipt.activation_nonce) == 32

    doc = json.loads(out.read_text())
    assert check_activation_receipt(doc, chain.registry, now=NOW,
                                    require_production=True) == []
    assert ledger.verify(chain.registry, now=NOW) == []
    assert (tmp_path / "nonces.jsonl").read_text().count(
        receipt.activation_nonce) == 1

    # the snapshot is immutable: the backend cannot be handed a mutable
    # source path, and the staged bytes survive a source mutation
    assert backend.seen.root != chain.adir


def test_snapshot_survives_source_mutation_during_load(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)

    def mutate():
        (chain.adir / "adapter_model.safetensors").write_bytes(b"swapped")

    backend = FakeBackend(mutate=mutate)
    result = launcher.launch(_request(chain), backend)
    # the loaded artifact was the verified copy, not the mutated source
    assert result.receipt.adapter_digest == chain.manifest["adapter_digest"]
    staged = (result.snapshot.path("adapter")
              / "adapter_model.safetensors").read_bytes()
    assert staged == b"adapter-weights"
    # ... and the mutated source cannot be launched again
    with pytest.raises(LaunchRefused, match="adapter bytes differ"):
        launcher.launch(_request(chain), FakeBackend())


# ---------- refusals -----------------------------------------------------

def test_backend_identity_mismatch_refused(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    backend = FakeBackend()
    backend.backend_id = "qw3-native"
    with pytest.raises(LaunchRefused, match="does not match"):
        launcher.launch(_request(chain), backend)
    assert not list((tmp_path / "receipts").glob("*.json"))
    assert backend.seen is None


def test_backend_load_failure_emits_no_receipt(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="failed to load"):
        launcher.launch(_request(chain), FakeBackend(fail=True))
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_symlink_added_to_adapter_refuses_launch(tmp_path):
    chain = _build_chain(tmp_path)
    (chain.adir / "smuggled.bin").symlink_to(
        chain.adir / "adapter_model.safetensors")
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="symbolic link"):
        launcher.launch(_request(chain), FakeBackend())


def test_substituted_model_refuses_launch(tmp_path):
    chain = _build_chain(tmp_path)
    (chain.model / "model.safetensors").write_bytes(b"evil-weights")
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="base model digest differs"):
        launcher.launch(_request(chain), FakeBackend())


def test_manifest_tokenizer_binding_must_match_model_snapshot(tmp_path):
    """The plan binds the tokenizer inside the model snapshot; a
    manifest whose tokenizer digest does not re-derive from the model
    bytes is refused — there is no separately substitutable tokenizer."""
    chain = _build_chain(tmp_path)
    manifest = dict(chain.manifest)
    manifest["tokenizer_digest"] = "sha256:" + "5" * 64
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    decision = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": manifest["digest"]}})
    decision["runtime_manifests"] = {"seed-0": manifest}
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="tokenizer digest differs"):
        launcher.launch(_request(chain, runtime_manifest=manifest,
                                 decision_doc=decision), FakeBackend())


def test_tokenizer_cross_check_path_mismatch_refused(tmp_path):
    """An explicit tokenizer root is a cross-check: one whose tokenizer
    files differ from the model snapshot's is refused."""
    chain = _build_chain(tmp_path)
    (chain.tokenizer / "tokenizer.json").write_text('{"vocab": ["evil"]}')
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="tokenizer artifact differs"):
        launcher.launch(_request(chain), FakeBackend())


def test_model_snapshot_without_tokenizer_refused(tmp_path):
    """A model snapshot carrying no tokenizer artifacts cannot satisfy
    the plan's tokenizer binding — refused, not served."""
    chain = _build_chain(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (chain.model / name).unlink()
    manifest = dict(chain.manifest)
    manifest["model_digest"] = close_tree(
        chain.model, resolve_symlinks=True).digest
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    decision = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": manifest["digest"]}})
    decision["runtime_manifests"] = {"seed-0": manifest}
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="no tokenizer artifact files"):
        launcher.launch(_request(chain, runtime_manifest=manifest,
                                 decision_doc=decision), FakeBackend())


def test_caller_supplied_digest_never_trusted(tmp_path):
    chain = _build_chain(tmp_path)
    controller = RuntimeAdmissionController(
        chain.registry, revocation_list=_revocation(chain),
        max_revocation_age_seconds=86400, now=NOW)
    with pytest.raises(AdmissionRefused, match="disagrees with the "
                                               "physically measured"):
        controller.admit(
            decision_doc=chain.decision_doc,
            qualification_doc=chain.qual_doc, plan_doc=chain.plan_doc,
            runtime_manifest=chain.manifest, adapter_dir=chain.adir,
            seed="seed-0", runtime_model_digest="sha256:" + "9" * 64,
            runtime_model_path=chain.model,
            runtime_tokenizer_path=chain.tokenizer)


def test_stale_revocation_snapshot_refused(tmp_path):
    chain = _build_chain(tmp_path)
    stale = _revocation_snapshot(chain, issued_at=TS - 10**7,
                                 valid_until=TS - 10**7 + 86400)
    with pytest.raises(LaunchRefused, match="revocation evidence refused"):
        TrustedRuntimeLauncher(
            chain.registry, revocation_snapshot=stale,
            runtime_signer=chain.signers["runtime"],
            admission_signer=chain.signers["admission"],
            snapshot_root=tmp_path / "snapshots",
            nonce_journal=tmp_path / "nonces.jsonl",
            receipts_dir=tmp_path / "receipts",
            ledger_path=tmp_path / "ledger.jsonl",
            now=NOW).launch(_request(chain), FakeBackend())


def test_launcher_requires_revocation_evidence(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(LaunchRefused,
                       match="authenticated revocation evidence"):
        TrustedRuntimeLauncher(chain.registry, revocation_snapshot=None,
                               revocation_store=None,
                               runtime_signer=chain.signers["runtime"],
                               admission_signer=chain.signers["admission"],
                               snapshot_root=tmp_path / "s",
                               nonce_journal=tmp_path / "n.jsonl")


def test_launcher_requires_runtime_signer(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(LaunchRefused, match="signing identity is required"):
        TrustedRuntimeLauncher(chain.registry,
                               revocation_snapshot=_revocation_snapshot(chain),
                               runtime_signer=None,
                               admission_signer=chain.signers["admission"],
                               snapshot_root=tmp_path / "s",
                               nonce_journal=tmp_path / "n.jsonl")


def test_launcher_requires_admission_signer(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(LaunchRefused,
                       match="admission signing identity"):
        TrustedRuntimeLauncher(chain.registry,
                               revocation_snapshot=_revocation_snapshot(chain),
                               runtime_signer=chain.signers["runtime"],
                               admission_signer=None,
                               snapshot_root=tmp_path / "s",
                               nonce_journal=tmp_path / "n.jsonl")


def test_unsigned_revocation_list_refused(tmp_path):
    """A bare RevocationList document is not authenticated revocation
    evidence — the v16.4.2 production path refuses it."""
    chain = _build_chain(tmp_path)
    unsigned = {"schema": "mini-agi-v16.4.1-revocation-list-v1",
                "digests": [], "generated_at": TS}
    with pytest.raises(LaunchRefused,
                       match="signed envelope required"):
        TrustedRuntimeLauncher(
            chain.registry, revocation_snapshot=unsigned,
            runtime_signer=chain.signers["runtime"],
            admission_signer=chain.signers["admission"],
            snapshot_root=tmp_path / "s", nonce_journal=tmp_path / "n.jsonl",
            receipts_dir=tmp_path / "r", now=NOW).launch(
                _request(chain), FakeBackend())


def test_revoked_decision_refused(tmp_path):
    chain = _build_chain(tmp_path)
    revoked = _revocation_snapshot(
        chain, revoked=(chain.decision_doc["digest"],))
    launcher = _launcher(chain, tmp_path, revocation=revoked)
    with pytest.raises(LaunchRefused, match="revoked"):
        launcher.launch(_request(chain), FakeBackend())


def test_expired_decision_refused(tmp_path):
    chain = _build_chain(tmp_path, expires_at=TS - 10)
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="expired"):
        launcher.launch(_request(chain), FakeBackend())


def test_manifest_without_file_listing_refused_for_serving(tmp_path):
    """A v16.4.0-style manifest (aggregate digest only) is not enough
    for serving: the launcher requires the explicit authorized listing."""
    chain = _build_chain(tmp_path)
    manifest = dict(chain.manifest)
    manifest["schema"] = "mini-agi-v16.5-peft-runtime-manifest-v1"
    manifest.pop("adapter_files")
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    decision = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": manifest["digest"]}})
    decision["runtime_manifests"] = {"seed-0": manifest}
    launcher = _launcher(chain, tmp_path)
    with pytest.raises(LaunchRefused, match="explicit"):
        launcher.launch(_request(chain, runtime_manifest=manifest,
                                 decision_doc=decision), FakeBackend())


def test_launcher_rollback_readmits_previous_release_with_lineage(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    current = launcher.launch(_request(chain), FakeBackend())
    rolled = launcher.launch(
        _request(chain, rollback_of=current.receipt.decision_digest),
        FakeBackend())
    assert rolled.receipt.rollback_of == current.receipt.decision_digest
    assert rolled.receipt.adapter_digest == current.receipt.adapter_digest
    # a tampered rollback target is refused, not silently activated
    (chain.adir / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(LaunchRefused, match="adapter bytes differ"):
        launcher.launch(
            _request(chain, rollback_of=current.receipt.decision_digest),
            FakeBackend())


def test_nonce_replay_is_refused_across_receipts(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    first = launcher.launch(_request(chain), FakeBackend())
    second = launcher.launch(_request(chain), FakeBackend())
    assert first.receipt.activation_nonce != second.receipt.activation_nonce
    journal = (tmp_path / "nonces.jsonl").read_text().splitlines()
    assert len(journal) == 2


# ---------- receipt verification -----------------------------------------

def test_unsigned_and_wrong_role_receipts_are_problems(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    result = launcher.launch(_request(chain), FakeBackend())

    unsigned = {"value": result.receipt_doc["value"],
                "digest": result.receipt_doc["digest"]}
    problems = check_activation_receipt(unsigned, chain.registry, now=NOW)
    assert any("unsigned" in p for p in problems)

    wrong = result.receipt.to_doc(signer=chain.signers["promotion"])
    problems = check_activation_receipt(wrong, chain.registry, now=NOW)
    assert any("not an authorized" in p for p in problems)


def test_admission_only_receipt_is_not_production_evidence(tmp_path):
    chain = _build_chain(tmp_path)
    controller = RuntimeAdmissionController(chain.registry, now=NOW)
    admission = controller.admit(
        decision_doc=chain.decision_doc, qualification_doc=chain.qual_doc,
        plan_doc=chain.plan_doc, runtime_manifest=chain.manifest,
        adapter_dir=chain.adir, seed="seed-0",
        runtime_model_path=chain.model,
        runtime_tokenizer_path=chain.tokenizer)
    doc = admission.to_doc(signer=chain.signers["runtime"])
    assert check_activation_receipt(doc, chain.registry, now=NOW) == []
    problems = check_activation_receipt(doc, chain.registry, now=NOW,
                                        require_production=True)
    assert any("measured at load" in p for p in problems)


def test_production_receipt_schema_required(tmp_path):
    chain = _build_chain(tmp_path)
    launcher = _launcher(chain, tmp_path)
    result = launcher.launch(_request(chain), FakeBackend())
    doc = json.loads(json.dumps(result.receipt_doc))
    doc["value"]["schema"] = "mini-agi-v16.9-activation-receipt-v9"
    doc["digest"] = digest(doc["value"])
    env = chain.signers["runtime"].sign(doc["value"])
    doc.update({"signer_key_id": env.key_id,
                "signature_b64": env.signature_b64})
    problems = check_activation_receipt(doc, chain.registry, now=NOW)
    assert any("unrecognized schema" in p for p in problems)


# ---------- no raw-path serving entry ------------------------------------

def test_peft_backend_refuses_raw_paths(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(PermissionError, match="MeasuredSnapshot"):
        PeftServingBackend().load(chain.adir)
    with pytest.raises(PermissionError, match="MeasuredSnapshot"):
        PeftServingBackend().load(str(chain.adir))


def test_cuda_loader_refuses_serving_a_raw_adapter_path(tmp_path):
    """The CUDA/HF loader is a second model-loading entry; serving
    through it with a raw adapter path (or no snapshot at all) must be
    refused, and the research plane must opt in explicitly."""
    from minagi.platforms.cuda.hf_runtime import HFLoadSpec, load_causal_lm
    spec = HFLoadSpec(model_id="tiny", revision="r1")
    with pytest.raises(PermissionError, match="MeasuredSnapshot"):
        load_causal_lm(spec, adapter_path=str(tmp_path / "adapter"))
    with pytest.raises(ValueError, match="purpose"):
        load_causal_lm(spec, purpose="whatever")
    # serving without a snapshot is a bare load of an unadmitted model
    with pytest.raises(PermissionError, match="snapshot"):
        load_causal_lm(spec)
