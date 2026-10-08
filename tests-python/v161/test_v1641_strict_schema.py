"""v16.4.1 — strict versioned schemas for authority-bearing artifacts.

v16.4.0 accepted a qualification record or campaign plan of any
version, compared several digest bindings only "when present on both
sides" (so omitting one side disabled the check), and left the
protocol link optional. These tests drive every one of those holes:
unknown versions, absent required digests, unsupported backends,
unbound evaluations, and an unbound protocol are refused.
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
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import runtime_manifest  # noqa: E402
from minagi.v161.runtime_admission import (  # noqa: E402
    AdmissionRefused, RuntimeAdmissionController)
from minagi.v161 import strict_schema  # noqa: E402

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())
D = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64


def _signed(signer, value):
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


# ---------- direct validator coverage ------------------------------------

def test_unknown_schema_versions_refused():
    with pytest.raises(strict_schema.SchemaRefused, match="unrecognized"):
        strict_schema.validate("campaign_plan", {
            "schema": "mini-agi-v16.99-colab-campaign-plan-v1",
            "campaign_id": "c", "experiment_protocol_digest": D})
    with pytest.raises(strict_schema.SchemaRefused, match="unrecognized"):
        strict_schema.validate("qualification_record", {
            "schema": "mini-agi-v16.99-qualification-record-v1",
            "campaign_id": "c", "decision": "QUALIFIED",
            "campaign_plan_digest": D})
    with pytest.raises(strict_schema.SchemaRefused, match="unrecognized"):
        strict_schema.validate("promotion_decision", {
            "schema": "mini-agi-v16.99-promotion-decision-v1"})
    with pytest.raises(strict_schema.SchemaRefused, match="unrecognized"):
        strict_schema.validate("runtime_manifest", {
            "schema": "mini-agi-v16.99-manifest-v1"})
    with pytest.raises(strict_schema.SchemaRefused, match="unrecognized"):
        strict_schema.validate("activation_receipt", {
            "schema": "mini-agi-v16.99-activation-receipt-v1"})


def test_absent_required_digests_refused():
    with pytest.raises(strict_schema.SchemaRefused,
                       match="experiment_protocol_digest"):
        strict_schema.validate("campaign_plan", {
            "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
            "campaign_id": "c"})
    with pytest.raises(strict_schema.SchemaRefused,
                       match="campaign_plan_digest"):
        strict_schema.validate("qualification_record", {
            "schema": "mini-agi-v16.5-qualification-record-v1",
            "campaign_id": "c", "decision": "QUALIFIED"})
    with pytest.raises(strict_schema.SchemaRefused, match="malformed"):
        strict_schema.validate("qualification_record", {
            "schema": "mini-agi-v16.5-qualification-record-v1",
            "campaign_id": "c", "decision": "QUALIFIED",
            "campaign_plan_digest": "not-a-digest"})
    with pytest.raises(strict_schema.SchemaRefused,
                       match="evaluation_bundle_digest"):
        strict_schema.validate("promotion_decision", {
            "schema": "mini-agi-v16.5-promotion-decision-v1",
            "campaign_id": "c", "campaign_plan_digest": D,
            "qualification_record_digest": D, "adapter": "L6",
            "adapter_artifact_digests": {"seed-0": D2},
            "runtime_manifest_digests": {"seed-0": D2},
            "authorized_at": 1, "expires_at": 2})
    with pytest.raises(strict_schema.SchemaRefused, match="protocol_digest"):
        strict_schema.validate("runtime_manifest", {
            "schema": "mini-agi-v16.5-peft-runtime-manifest-v2",
            "model_id": "m", "model_revision": "r", "model_digest": D,
            "tokenizer_digest": D, "adapter_digest": D,
            "campaign_digest": D, "qualification_record_digest": D,
            "serving_stack": "hf-peft",
            "lora": {"rank": 4, "target_modules": ["c_attn"]}})


def test_unsupported_backend_identifiers_refused():
    base = {
        "schema": "mini-agi-v16.5-peft-runtime-manifest-v2",
        "model_id": "m", "model_revision": "r", "model_digest": D,
        "tokenizer_digest": D, "adapter_digest": D, "protocol_digest": D,
        "campaign_digest": D, "qualification_record_digest": D,
        "lora": {"rank": 4, "target_modules": ["c_attn"]}}
    with pytest.raises(strict_schema.SchemaRefused, match="unsupported backend"):
        strict_schema.validate("runtime_manifest",
                               {**base, "serving_stack": "qw3-native"})
    with pytest.raises(strict_schema.SchemaRefused, match="unsupported backend"):
        strict_schema.validate("qualification_record", {
            "schema": "mini-agi-v16.5-qualification-record-v1",
            "campaign_id": "c", "decision": "QUALIFIED",
            "campaign_plan_digest": D,
            "runtime_backends": ["hf-peft", "qw3-native"]})


def test_decision_manifest_coverage_and_expiry_shape():
    base = {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": "c", "campaign_plan_digest": D,
        "qualification_record_digest": D, "evaluation_bundle_digest": D,
        "adapter": "L6", "adapter_artifact_digests": {"seed-0": D2},
        "runtime_manifest_digests": {"seed-0": D2},
        "authorized_at": 100, "expires_at": 200}
    strict_schema.validate("promotion_decision", base)
    with pytest.raises(strict_schema.SchemaRefused, match="expires_at"):
        strict_schema.validate("promotion_decision",
                               {**base, "expires_at": 100})
    with pytest.raises(strict_schema.SchemaRefused, match="cover exactly"):
        strict_schema.validate("promotion_decision",
                               {**base, "runtime_manifest_digests": {}})
    with pytest.raises(strict_schema.SchemaRefused, match="at least one seed"):
        strict_schema.validate("promotion_decision",
                               {**base, "adapter_artifact_digests": {}})


def test_manifest_listing_entries_validated():
    base = {
        "schema": "mini-agi-v16.5-peft-runtime-manifest-v2",
        "model_id": "m", "model_revision": "r", "model_digest": D,
        "tokenizer_digest": D, "adapter_digest": D, "protocol_digest": D,
        "campaign_digest": D, "qualification_record_digest": D,
        "serving_stack": "hf-peft",
        "lora": {"rank": 4, "target_modules": ["c_attn"]}}
    ok = [{"path": "adapter_config.json", "size": 1, "sha256": D}]
    strict_schema.validate("runtime_manifest", {**base, "adapter_files": ok})
    with pytest.raises(strict_schema.SchemaRefused, match="unsafe"):
        strict_schema.validate("runtime_manifest", {
            **base, "adapter_files": [
                {"path": "../escape", "size": 1, "sha256": D}]})
    with pytest.raises(strict_schema.SchemaRefused, match="duplicate"):
        strict_schema.validate("runtime_manifest", {
            **base, "adapter_files": [ok[0], dict(ok[0])]})
    with pytest.raises(strict_schema.SchemaRefused, match="integer size"):
        strict_schema.validate("runtime_manifest", {
            **base, "adapter_files": [
                {"path": "a.json", "size": -1, "sha256": D}]})
    with pytest.raises(strict_schema.SchemaRefused, match="content digest"):
        strict_schema.validate("runtime_manifest", {
            **base, "adapter_files": [
                {"path": "a.json", "size": 1, "sha256": "nope"}]})


def test_receipt_production_fields_validated():
    value = {"schema": "mini-agi-v16.4.1-activation-receipt-v2",
             "decision_digest": D, "qualification_record_digest": D,
             "runtime_manifest_digest": D, "adapter_digest": D,
             "backend": "hf-peft", "admitted_at": "2026-10-08T12:00:00Z",
             "candidate_digest": D, "activation_nonce": "ab" * 16,
             "loaded_artifact_digests": [["adapter", D]]}
    strict_schema.validate("activation_receipt", value,
                           require_production=True)
    with pytest.raises(strict_schema.SchemaRefused, match="activation_nonce"):
        strict_schema.validate("activation_receipt",
                               {**value, "activation_nonce": "short"},
                               require_production=True)
    with pytest.raises(strict_schema.SchemaRefused, match="measured at load"):
        strict_schema.validate("activation_receipt",
                               {**value, "loaded_artifact_digests": []},
                               require_production=True)
    with pytest.raises(strict_schema.SchemaRefused,
                       match="not a production"):
        strict_schema.validate("activation_receipt", {
            **value, "schema": "mini-agi-v16.4-activation-receipt-v1"},
            require_production=True)


# ---------- admission-level strictness -----------------------------------

def _chain(tmp_path, *, plan_protocol=True, qual_bundle=True):
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

    proto = ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48}})
    plan_value = {"schema": "mini-agi-v16.6-colab-campaign-plan-v1",
                  "campaign_id": "camp"}
    if plan_protocol:
        plan_value["experiment_protocol_digest"] = proto.digest
    plan_doc = _signed(signers["plan"], plan_value)

    bundle = digest({"bundle": 1})
    qual_value = {"schema": "mini-agi-v16.5-qualification-record-v1",
                  "campaign_id": "camp", "decision": "QUALIFIED",
                  "campaign_plan_digest": plan_doc["digest"],
                  "runtime_backends": ["hf-peft"]}
    if qual_bundle:
        qual_value["evaluation_bundle_digest"] = bundle
    qual_doc = _signed(signers["qualification"], qual_value)

    manifest = runtime_manifest(
        model_id="m", model_revision="r1", model_digest=D,
        tokenizer_digest=D2, adapter_dir=adir, protocol=proto,
        campaign_digest=plan_doc["digest"],
        qualification_record_digest=qual_doc["digest"])
    decision = _signed(signers["promotion"], {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": "camp",
        "campaign_plan_digest": plan_doc["digest"],
        "qualification_record_digest": qual_doc["digest"],
        "evaluation_bundle_digest": bundle,
        "adapter": "L6",
        "adapter_artifact_digests": {"seed-0": manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": manifest["digest"]},
        "authorized_at": TS - 60, "expires_at": TS + 3600})
    decision["runtime_manifests"] = {"seed-0": manifest}
    return SimpleNamespace(registry=registry, signers=signers, adir=adir,
                           plan_doc=plan_doc, qual_doc=qual_doc,
                           manifest=manifest, decision_doc=decision)


def _admit(chain, **overrides):
    kw = dict(decision_doc=chain.decision_doc,
              qualification_doc=chain.qual_doc, plan_doc=chain.plan_doc,
              runtime_manifest=chain.manifest, adapter_dir=chain.adir,
              seed="seed-0", runtime_model_digest=D,
              runtime_tokenizer_digest=D2, expected_backend="hf-peft")
    kw.update(overrides)
    return RuntimeAdmissionController(chain.registry, now=NOW).admit(**kw)


def test_plan_without_protocol_digest_refused(tmp_path):
    chain = _chain(tmp_path, plan_protocol=False)
    with pytest.raises(AdmissionRefused, match="experiment_protocol_digest"):
        _admit(chain)


def test_qualification_without_evaluation_bundle_refused(tmp_path):
    chain = _chain(tmp_path, qual_bundle=False)
    with pytest.raises(AdmissionRefused,
                       match="does not bind an evaluation bundle"):
        _admit(chain)


def test_manifest_protocol_must_match_signed_plan(tmp_path):
    chain = _chain(tmp_path)
    manifest = dict(chain.manifest)
    manifest["protocol_digest"] = "sha256:" + "8" * 64
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    decision = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": manifest["digest"]}})
    decision["runtime_manifests"] = {"seed-0": manifest}
    with pytest.raises(AdmissionRefused, match="protocol differs"):
        _admit(chain, runtime_manifest=manifest, decision_doc=decision)


def test_manifest_without_protocol_refused(tmp_path):
    chain = _chain(tmp_path)
    manifest = dict(chain.manifest)
    manifest["protocol_digest"] = None
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    decision = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": manifest["digest"]}})
    decision["runtime_manifests"] = {"seed-0": manifest}
    with pytest.raises(AdmissionRefused, match="protocol_digest"):
        _admit(chain, runtime_manifest=manifest, decision_doc=decision)


def test_stale_revocation_list_refused_at_admission(tmp_path):
    from minagi.v161.runtime_admission import RevocationList
    chain = _chain(tmp_path)
    controller = RuntimeAdmissionController(
        chain.registry,
        revocation_list=RevocationList(digests=(), generated_at=TS - 10**7),
        max_revocation_age_seconds=3600, now=NOW)
    with pytest.raises(AdmissionRefused, match="stale"):
        controller.admit(
            decision_doc=chain.decision_doc,
            qualification_doc=chain.qual_doc, plan_doc=chain.plan_doc,
            runtime_manifest=chain.manifest, adapter_dir=chain.adir,
            seed="seed-0", runtime_model_digest=D,
            runtime_tokenizer_digest=D2)
