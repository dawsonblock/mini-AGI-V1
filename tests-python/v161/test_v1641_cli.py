"""v16.4.1 — deployment CLIs exercise the closed paths end to end.

`scripts/admit_runtime.py` (admission evidence) and
`scripts/trusted_launch.py` (the supported model-loading path) are run
as subprocesses against a real storage layout: signed chain on disk,
physical model/tokenizer/adapter artifacts, revocation list, and a
test serving backend. The refusals mirror the audit's acceptance
criteria — a symlink or a single altered byte prevents activation, a
stale revocation list prevents activation, and no receipt is written
when anything fails.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2, RevocationStore)
from minagi.v161.artifact_closure import (close_tree,  # noqa: E402
                                          tokenizer_artifact_digest)
from minagi.v161.authority import AUTHORITY_ROLES, write_trust_root  # noqa: E402
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import runtime_manifest  # noqa: E402

TS = int(time.time())  # the CLIs verify against the real clock
ADMIT_CLI = ROOT / "scripts" / "admit_runtime.py"
LAUNCH_CLI = ROOT / "scripts" / "trusted_launch.py"


def _signed(signer, value):
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


def _storage(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
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
    # the signed plan binds the tokenizer inside the model snapshot
    (model / "tokenizer.json").write_text('{"vocab": []}')
    tok = storage / "models" / "tok"
    tok.mkdir(parents=True)
    (tok / "tokenizer.json").write_text('{"vocab": []}')

    proto = ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48}})
    plan_doc = _signed(signers["plan"], {
        "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
        "campaign_id": "camp",
        "experiment_protocol_digest": proto.digest})
    bundle = digest({"bundle": 1})
    qual_doc = _signed(signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": "camp", "decision": "QUALIFIED",
        "campaign_plan_digest": plan_doc["digest"],
        "evaluation_bundle_digest": bundle,
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
        "evaluation_bundle_digest": bundle,
        "adapter": "L6",
        "adapter_artifact_digests": {"seed-0": manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": manifest["digest"]},
        "authorized_at": TS - 60, "expires_at": TS + 3600})
    decision["runtime_manifests"] = {"seed-0": manifest}

    cdir = storage / "evidence" / "camp"
    cdir.mkdir(parents=True)
    (cdir / "CAMPAIGN_PLAN.json").write_text(json.dumps(plan_doc))
    (cdir / "QUALIFICATION_RECORD.json").write_text(json.dumps(qual_doc))
    decision_path = cdir / "RUNTIME_MANIFEST.json"
    decision_path.write_text(json.dumps(decision))
    revocation = tmp_path / "REVOCATIONS.json"
    revocation.write_text(json.dumps({
        "schema": "mini-agi-v16.4.1-revocation-list-v1",
        "digests": [], "generated_at": TS}))
    # v16.4.2: the launch path consumes signed revocation snapshots
    # from a durable store, not a bare list
    rev_dir = storage / "revocations"
    snap = RevocationSnapshotV2(
        epoch=0, issued_at=TS - 60, valid_until=TS + 86400)
    RevocationStore(rev_dir).publish(
        snap.to_doc(signer=signers["revocation"]))
    return SimpleNamespace(storage=storage, adir=adir, model=model, tok=tok,
                           cdir=cdir, decision_path=decision_path,
                           revocation=revocation, rev_dir=rev_dir,
                           manifest=manifest)


def _run(cli, *args, env=None):
    return subprocess.run([sys.executable, str(cli), *args],
                          capture_output=True, text=True, env=env)


def _admit_args(chain, out):
    return ["--storage-root", str(chain.storage), "--campaign-id", "camp",
            "--seed", "seed-0", "--adapter-dir", str(chain.adir),
            "--decision", str(chain.decision_path),
            "--runtime-model-path", str(chain.model),
            "--runtime-tokenizer-path", str(chain.tok),
            "--out", str(out)]


# ---------- admit_runtime.py ---------------------------------------------

def test_admit_cli_measures_artifacts_and_requires_signature(tmp_path):
    chain = _storage(tmp_path)
    out = chain.storage / "A.json"
    proc = _run(ADMIT_CLI, *_admit_args(chain, out))
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert "signer_key_id" in doc, "receipt must be signed"
    assert json.loads(proc.stdout)["production"] is False


def test_admit_cli_refuses_symlinked_adapter(tmp_path):
    chain = _storage(tmp_path)
    (chain.adir / "extra.safetensors").symlink_to(
        chain.adir / "adapter_model.safetensors")
    out = chain.storage / "A.json"
    proc = _run(ADMIT_CLI, *_admit_args(chain, out))
    assert proc.returncode == 2
    assert "symbolic link" in proc.stderr
    assert not out.exists()


def test_admit_cli_refuses_caller_digest_disagreement(tmp_path):
    chain = _storage(tmp_path)
    out = chain.storage / "A.json"
    proc = _run(ADMIT_CLI, *_admit_args(chain, out),
                "--runtime-model-digest", "sha256:" + "9" * 64)
    assert proc.returncode == 2
    assert "disagrees with the physically measured" in proc.stderr
    assert not out.exists()


def test_admit_cli_refuses_stale_revocation_list(tmp_path):
    chain = _storage(tmp_path)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({
        "schema": "mini-agi-v16.4.1-revocation-list-v1",
        "digests": [], "generated_at": TS - 10**7}))
    out = chain.storage / "A.json"
    proc = _run(ADMIT_CLI, *_admit_args(chain, out),
                "--revoked-decisions", str(stale),
                "--max-revocation-age-days", "1")
    assert proc.returncode == 2
    assert "stale" in proc.stderr
    assert not out.exists()


def test_admit_cli_refuses_without_runtime_key(tmp_path):
    chain = _storage(tmp_path)
    (chain.storage / ".keys" / "runtime.pem").unlink()
    out = chain.storage / "A.json"
    proc = _run(ADMIT_CLI, *_admit_args(chain, out))
    assert proc.returncode == 3
    assert "unsigned receipt is not authorization evidence" in proc.stderr
    assert not out.exists()


# ---------- trusted_launch.py --------------------------------------------

BACKEND_SRC = '''
import json


class Backend:
    backend_id = "hf-peft"

    def load(self, snapshot):
        assert type(snapshot).__name__ == "MeasuredSnapshot"
        cfg = (snapshot.path("adapter") / "adapter_config.json").read_text()
        return {"config": json.loads(cfg)}

    def health_probe(self, handle):
        return None

    def unload(self, handle):
        return None
'''


def _launch_args(chain, factory, receipt):
    return ["--storage-root", str(chain.storage), "--campaign-id", "camp",
            "--seed", "seed-0", "--decision", str(chain.decision_path),
            "--adapter-dir", str(chain.adir),
            "--model-path", str(chain.model),
            "--tokenizer-path", str(chain.tok),
            "--revocation-store", str(chain.rev_dir),
            "--backend-factory", f"c3a_fake_backend:{factory}",
            "--receipt", str(receipt)]


def _factory_env(tmp_path, source=BACKEND_SRC):
    (tmp_path / "c3a_fake_backend.py").write_text(source)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get(
        "PYTHONPATH", "")
    return env


def test_launch_cli_stages_loads_and_writes_production_receipt(tmp_path):
    chain = _storage(tmp_path)
    receipt = chain.cdir / "ACTIVATION_RECEIPT.json"
    proc = _run(LAUNCH_CLI, *_launch_args(chain, "Backend", receipt),
                env=_factory_env(tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["status"] == "LAUNCHED" and out["production"] is True
    doc = json.loads(receipt.read_text())
    assert "signer_key_id" in doc

    from minagi.v161.authority import AuthorityRegistry
    from minagi.v161.runtime_admission import check_activation_receipt
    registry = AuthorityRegistry.load(chain.storage / "trust_root.json")
    problems = check_activation_receipt(doc, registry,
                                        require_production=True)
    assert problems == []
    # the snapshot the backend loaded from is frozen and staged
    snapshot_root = Path(out["snapshot_root"])
    assert snapshot_root.is_dir()
    assert not ((snapshot_root / "adapter"
                 / "adapter_model.safetensors").stat().st_mode & 0o200)


def test_launch_cli_refuses_altered_byte_and_writes_no_receipt(tmp_path):
    chain = _storage(tmp_path)
    (chain.adir / "adapter_model.safetensors").write_bytes(b"altered")
    receipt = chain.cdir / "ACTIVATION_RECEIPT.json"
    proc = _run(LAUNCH_CLI, *_launch_args(chain, "Backend", receipt),
                env=_factory_env(tmp_path))
    assert proc.returncode == 2
    assert "adapter bytes differ" in proc.stderr
    assert not receipt.exists()


def test_launch_cli_refuses_backend_that_is_not_the_requested_one(tmp_path):
    chain = _storage(tmp_path)
    receipt = chain.cdir / "ACTIVATION_RECEIPT.json"
    bad = "class Backend:\n    backend_id = 'qw3-native'\n" \
          "    def load(self, snapshot):\n        return None\n"
    proc = _run(LAUNCH_CLI, *_launch_args(chain, "Backend", receipt),
                env=_factory_env(tmp_path, bad))
    assert proc.returncode == 2
    assert "does not match" in proc.stderr
    assert not receipt.exists()
