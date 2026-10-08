"""v16.4.0 Phase 3 — runtime admission control.

The plan's Phase-3 acceptance gate, exercised adversarially:
alter one byte of the adapter, substitute an old qualification, replay
a revoked decision, and promote under a research-plane identity — every
attempt must fail without activating the candidate. Plus: expired
decisions, unsigned envelopes, backend substitution, model/tokenizer
substitution, tampered manifests, incomplete chains, and rollback.
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
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityLedger,  # noqa: E402
                                   AuthorityRegistry, write_trust_root)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import runtime_manifest  # noqa: E402
from minagi.v161.runtime_admission import (  # noqa: E402
    PROMOTION_DECISION_SCHEMA, ActivationReceipt, AdmissionRefused,
    RuntimeAdmissionController, check_activation_receipt,
    write_activation_receipt)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())
MODEL_D = "sha256:" + "1" * 64
TOKENIZER_D = "sha256:" + "2" * 64


def _signed(signer, value):
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


def _protocol():
    return ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48},
    })


def _build_chain(tmp_path, *, expires_at=None, authorized_at=None,
                 backend="hf-peft"):
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
        "lora_alpha": 8, "lora_dropout": 0.0,
        "target_modules": ["c_attn"]}))
    (adir / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adir / "TRAINING_RECEIPT.json").write_text("{}")

    model_dir = storage / "models" / "base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
    (model_dir / "model.safetensors").write_bytes(b"base-model-weights")
    tok_dir = storage / "models" / "tok"
    tok_dir.mkdir(parents=True)
    (tok_dir / "tokenizer.json").write_text('{"vocab": []}')
    model_d = close_tree(model_dir, resolve_symlinks=True).digest
    tokenizer_d = close_tree(tok_dir, resolve_symlinks=True).digest

    proto = _protocol()
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
        model_id="m", model_revision="r1", model_digest=model_d,
        tokenizer_digest=tokenizer_d, adapter_dir=adir, protocol=proto,
        campaign_digest=plan_doc["digest"],
        qualification_record_digest=qual_doc["digest"])

    decision_value = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "campaign_id": "camp",
        "campaign_plan_digest": plan_doc["digest"],
        "qualification_record_digest": qual_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "adapter": "L6",
        "adapter_artifact_digests": {"seed-0": manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": manifest["digest"]},
        "authorized_at": TS - 60 if authorized_at is None else authorized_at,
        "expires_at": TS + 3600 if expires_at is None else expires_at}
    decision_doc = _signed(signers["promotion"], decision_value)
    # promote.py's document shape: the per-seed runtime manifests travel
    # inside the promotion document (outside the signed value)
    decision_doc["runtime_manifests"] = {"seed-0": manifest}

    return SimpleNamespace(
        storage=storage, registry=registry, signers=signers, adir=adir,
        model_dir=model_dir, tok_dir=tok_dir, model_digest=model_d,
        tokenizer_digest=tokenizer_d, proto=proto, plan_doc=plan_doc,
        qual_doc=qual_doc, manifest=manifest, decision_doc=decision_doc,
        manifest_backend=backend)


def _resign(chain, value):
    """Re-sign a decision value with the promotion key (test helper for
    building adversarial-but-validly-signed variants)."""
    env = chain.signers["promotion"].sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


def _rebind_manifest(chain, **overrides):
    manifest = dict(chain.manifest)
    manifest.update(overrides)
    manifest["digest"] = digest(
        {k: v for k, v in manifest.items() if k != "digest"})
    return manifest


def _controller(chain, **kw):
    return RuntimeAdmissionController(chain.registry, now=NOW, **kw)


def _admit_kwargs(chain):
    return dict(decision_doc=chain.decision_doc,
                qualification_doc=chain.qual_doc,
                plan_doc=chain.plan_doc,
                runtime_manifest=chain.manifest,
                adapter_dir=chain.adir, seed="seed-0",
                runtime_model_digest=chain.model_digest,
                runtime_tokenizer_digest=chain.tokenizer_digest,
                expected_backend="hf-peft")


# ---------- happy path --------------------------------------------------

def test_clean_admission_produces_activation_receipt(tmp_path):
    chain = _build_chain(tmp_path)
    receipt = _controller(chain).admit(**_admit_kwargs(chain))
    assert isinstance(receipt, ActivationReceipt)
    assert receipt.decision_digest == chain.decision_doc["digest"]
    assert receipt.qualification_record_digest == chain.qual_doc["digest"]
    assert receipt.runtime_manifest_digest == chain.manifest["digest"]
    assert receipt.adapter_digest == chain.manifest["adapter_digest"]
    assert receipt.backend == "hf-peft"
    assert receipt.rollback_of == ""
    doc = write_activation_receipt(
        receipt, chain.storage / "ACTIVATION_RECEIPT.json",
        signer=chain.signers["runtime"])
    assert check_activation_receipt(doc, chain.registry, now=NOW) == []


# ---------- the plan's four named attacks -------------------------------

def test_altered_adapter_byte_refused(tmp_path):
    chain = _build_chain(tmp_path)
    with (chain.adir / "adapter_model.safetensors").open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(AdmissionRefused, match="adapter bytes differ"):
        _controller(chain).admit(**_admit_kwargs(chain))


def test_substituted_qualification_refused(tmp_path):
    chain = _build_chain(tmp_path)
    old = _signed(chain.signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": "camp", "decision": "QUALIFIED",
        "campaign_plan_digest": chain.plan_doc["digest"],
        "evaluation_bundle_digest": digest({"bundle": 0})})
    kwargs = _admit_kwargs(chain) | {"qualification_doc": old}
    with pytest.raises(AdmissionRefused, match="substituted or stale"):
        _controller(chain).admit(**kwargs)


def test_revoked_decision_replay_refused(tmp_path):
    chain = _build_chain(tmp_path)
    ctrl = _controller(
        chain, revoked_decision_digests=[chain.decision_doc["digest"]])
    with pytest.raises(AdmissionRefused, match="revoked"):
        ctrl.admit(**_admit_kwargs(chain))


def test_research_plane_identity_refused(tmp_path):
    """A plan-role (research-plane) key cannot promote — even a
    registered one."""
    chain = _build_chain(tmp_path)
    forged = _signed(chain.signers["plan"], chain.decision_doc["value"])
    kwargs = _admit_kwargs(chain) | {"decision_doc": forged}
    with pytest.raises(AdmissionRefused, match="not an authorized"):
        _controller(chain).admit(**kwargs)


# ---------- further refusals -------------------------------------------

def test_expired_decision_refused(tmp_path):
    chain = _build_chain(tmp_path, authorized_at=TS - 7200,
                         expires_at=TS - 3600)
    with pytest.raises(AdmissionRefused, match="expired"):
        _controller(chain).admit(**_admit_kwargs(chain))


def test_future_decision_refused(tmp_path):
    chain = _build_chain(tmp_path, authorized_at=TS + 3600,
                         expires_at=TS + 7200)
    with pytest.raises(AdmissionRefused, match="not yet valid"):
        _controller(chain).admit(**_admit_kwargs(chain))


def test_unsigned_decision_refused(tmp_path):
    chain = _build_chain(tmp_path)
    kwargs = _admit_kwargs(chain) | {
        "decision_doc": chain.decision_doc["digest"]}
    with pytest.raises(AdmissionRefused, match="signed envelope"):
        _controller(chain).admit(**kwargs)


def test_forged_decision_value_refused(tmp_path):
    chain = _build_chain(tmp_path)
    tampered = json.loads(json.dumps(chain.decision_doc))
    tampered["value"]["adapter_artifact_digests"]["seed-0"] = \
        "sha256:" + "9" * 64
    with pytest.raises(AdmissionRefused, match="digest mismatch"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"decision_doc": tampered}))


def test_backend_not_covered_refused(tmp_path):
    chain = _build_chain(tmp_path)
    # a validly signed decision for a backend the qualification never
    # covered — admission must refuse on coverage, not on signature
    native_manifest = _rebind_manifest(chain, serving_stack="qw3-native")
    decision_value = {
        **chain.decision_doc["value"],
        "runtime_manifest_digests": {"seed-0": native_manifest["digest"]}}
    decision = _resign(chain, decision_value)
    with pytest.raises(AdmissionRefused, match="not covered"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {
                "runtime_manifest": native_manifest,
                "decision_doc": decision,
                "expected_backend": "qw3-native"}))
    # requested backend disagreeing with the manifest is refused too
    with pytest.raises(AdmissionRefused, match="!= manifest"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"expected_backend": "qw3-native"}))


def test_model_and_tokenizer_substitution_refused(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(AdmissionRefused, match="base model"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {
                "runtime_model_digest": "sha256:" + "7" * 64}))
    with pytest.raises(AdmissionRefused, match="tokenizer"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {
                "runtime_tokenizer_digest": "sha256:" + "7" * 64}))


def test_tampered_manifest_refused(tmp_path):
    chain = _build_chain(tmp_path)
    manifest = dict(chain.manifest)
    manifest["model_id"] = "evil/base"  # digest no longer matches body
    with pytest.raises(AdmissionRefused, match="manifest digest mismatch"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"runtime_manifest": manifest}))


def test_incomplete_chain_refused(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(AdmissionRefused, match="campaign plan"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"plan_doc": None}))
    # a REFUSED qualification, properly bound by a re-signed decision:
    # the decision gate itself must refuse
    refused = _signed(chain.signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": "camp", "decision": "REFUSED",
        "campaign_plan_digest": chain.plan_doc["digest"]})
    decision = _resign(chain, {
        **chain.decision_doc["value"],
        "qualification_record_digest": refused["digest"]})
    with pytest.raises(AdmissionRefused, match="QUALIFIED"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"qualification_doc": refused,
                                       "decision_doc": decision}))
    other_plan = _signed(chain.signers["plan"], {
        "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
        "campaign_id": "other",
        "experiment_protocol_digest": chain.proto.digest})
    with pytest.raises(AdmissionRefused, match="campaign plan digest"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"plan_doc": other_plan}))


def test_unauthorized_seed_refused(tmp_path):
    chain = _build_chain(tmp_path)
    with pytest.raises(AdmissionRefused, match="does not authorize seed"):
        _controller(chain).admit(
            **(_admit_kwargs(chain) | {"seed": "seed-9"}))


# ---------- rollback ----------------------------------------------------

def test_rollback_readmits_previous_release(tmp_path):
    chain = _build_chain(tmp_path)
    ctrl = _controller(chain)
    receipt = ctrl.admit(**_admit_kwargs(chain))
    current = _signed(chain.signers["promotion"], {
        **chain.decision_doc["value"],
        "authorized_at": TS - 30, "expires_at": TS + 1800})
    rolled = ctrl.rollback(current_decision_digest=current["digest"],
                           **_admit_kwargs(chain))
    assert rolled.rollback_of == current["digest"]
    assert rolled.adapter_digest == receipt.adapter_digest
    # a tampered rollback target is refused, not silently activated
    with (chain.adir / "adapter_model.safetensors").open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(AdmissionRefused, match="adapter bytes differ"):
        ctrl.rollback(current_decision_digest=current["digest"],
                      **_admit_kwargs(chain))


# ---------- activation receipt integrity --------------------------------

def test_receipt_tamper_and_wrong_signer_flagged(tmp_path):
    chain = _build_chain(tmp_path)
    receipt = _controller(chain).admit(**_admit_kwargs(chain))
    doc = write_activation_receipt(receipt, chain.storage / "A.json",
                                   signer=chain.signers["runtime"])
    doc["value"]["adapter_digest"] = "sha256:" + "5" * 64
    problems = check_activation_receipt(doc, chain.registry, now=NOW)
    assert any("digest" in p for p in problems)
    wrong = write_activation_receipt(receipt, chain.storage / "B.json",
                                     signer=chain.signers["promotion"])
    problems = check_activation_receipt(wrong, chain.registry, now=NOW)
    assert any("not an authorized" in p for p in problems)


def test_activation_receipt_in_authority_ledger(tmp_path):
    chain = _build_chain(tmp_path)
    receipt = _controller(chain).admit(**_admit_kwargs(chain))
    ledger = AuthorityLedger(chain.storage / "AUTHORITY_LEDGER.jsonl")
    ledger.append(chain.signers["plan"], "experiment_preregistration",
                  {"plan": 1})
    ledger.append(chain.signers["runtime"], "activation_receipt",
                  receipt.to_doc()["value"])
    assert ledger.verify(chain.registry, now=NOW) == []
    # the promotion role cannot sign an activation receipt
    ledger2 = AuthorityLedger(chain.storage / "LEDGER2.jsonl")
    ledger2.append(chain.signers["plan"], "experiment_preregistration",
                   {"plan": 1})
    ledger2.append(chain.signers["promotion"], "activation_receipt",
                   {"receipt": 1})
    failures = ledger2.verify(chain.registry, now=NOW)
    assert any("not authorized" in f for f in failures)


# ---------- operations CLI ----------------------------------------------

CLI = ROOT / "scripts" / "admit_runtime.py"


def _cli_inputs(chain):
    cdir = chain.storage / "evidence" / "camp"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "RUNTIME_MANIFEST.json").write_text(
        json.dumps(chain.decision_doc))
    (cdir / "QUALIFICATION_RECORD.json").write_text(
        json.dumps(chain.qual_doc))
    (cdir / "CAMPAIGN_PLAN.json").write_text(json.dumps(chain.plan_doc))
    return cdir


def _run_cli(chain, cdir, out):
    import subprocess
    return subprocess.run(
        [sys.executable, str(CLI), "--storage-root", str(chain.storage),
         "--campaign-id", "camp", "--seed", "seed-0",
         "--adapter-dir", str(chain.adir),
         "--decision", str(cdir / "RUNTIME_MANIFEST.json"),
         "--runtime-model-path", str(chain.model_dir),
         "--runtime-tokenizer-path", str(chain.tok_dir),
         "--out", str(out)], capture_output=True, text=True)


def test_cli_admits_and_writes_signed_receipt(tmp_path):
    import time as _time
    now_ts = int(_time.time())
    chain = _build_chain(tmp_path, authorized_at=now_ts - 60,
                         expires_at=now_ts + 3600)
    cdir = _cli_inputs(chain)
    out = chain.storage / "ACTIVATION_RECEIPT.json"
    proc = _run_cli(chain, cdir, out)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["status"] == "ADMITTED"
    doc = json.loads(out.read_text())
    assert check_activation_receipt(doc, chain.registry) == []


def test_cli_refuses_tampered_adapter(tmp_path):
    import time as _time
    now_ts = int(_time.time())
    chain = _build_chain(tmp_path, authorized_at=now_ts - 60,
                         expires_at=now_ts + 3600)
    cdir = _cli_inputs(chain)
    with (chain.adir / "adapter_model.safetensors").open("ab") as f:
        f.write(b"\x00")
    out = chain.storage / "ACTIVATION_RECEIPT.json"
    proc = _run_cli(chain, cdir, out)
    assert proc.returncode == 2
    assert "ADMISSION REFUSED" in proc.stderr
    assert not out.exists()
