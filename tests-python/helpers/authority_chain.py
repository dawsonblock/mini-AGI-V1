"""Shared test scaffolding: a complete signed authority chain for
activation + restoration tests (v16.4.5).

Lives in an importable helpers module so spawned workers and multiple
test files can reuse it — never defined inside __main__.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer
from minagi.security.admission_grants import issue_grant
from minagi.security.signed_revocations import (RevocationSnapshotV2,
                                              RevocationStore)
from minagi.v161.artifact_closure import (close_tree,
                                        tokenizer_artifact_digest)
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,
                                   write_trust_root)
from minagi.v161.experiment_protocol import ExperimentProtocolV1
from minagi.v161.peft_serving import runtime_manifest

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
TS = int(NOW.timestamp())


def signed(signer, value):
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


class FakeBackend:
    backend_id = "hf-peft"

    def __init__(self):
        self.loaded = []
        self.unloaded = []

    def load(self, snapshot):
        handle = {"from": str(snapshot.root)}
        self.loaded.append(handle)
        return handle

    def health_probe(self, handle):
        return None

    def infer(self, handle, request):
        return {"echo": request.get("prompt", "")}

    def unload(self, handle):
        self.unloaded.append(handle)


def trust_chain(tmp_path):
    storage = Path(tmp_path) / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def artifacts(tmp_path, tag="a"):
    model = Path(tmp_path) / f"model-{tag}"
    if not model.exists():
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "gpt2"}')
        (model / "tokenizer.json").write_text('{"vocab": []}')
        (model / "tokenizer_config.json").write_text('{"chat": "x"}')
    adir = Path(tmp_path) / f"adapter-{tag}"
    if not adir.exists():
        adir.mkdir()
        (adir / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4,
            "lora_alpha": 8, "lora_dropout": 0.0,
            "target_modules": ["c_attn"]}))
        (adir / "adapter_model.safetensors").write_bytes(
            b"w-" + tag.encode())
    return ({"model": close_tree(model).digest,
             "adapter": close_tree(adir).digest},
            {"model": str(model), "adapter": str(adir)})


def docs(signers, tmp_path, tag, *, backend="hf-peft",
         seed_name="seed-0", expires_at=None, authorized_at=None):
    """The complete signed authority chain for one activation."""
    digests, paths = artifacts(tmp_path, tag)
    proto = ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48}})
    plan_doc = signed(signers["plan"], {
        "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
        "campaign_id": f"camp-{tag}",
        "experiment_protocol_digest": proto.digest})
    bundle_d = digest({"bundle": tag})
    qual_doc = signed(signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": f"camp-{tag}", "decision": "QUALIFIED",
        "campaign_plan_digest": plan_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "runtime_backends": [backend]})
    manifest = runtime_manifest(
        model_id="tiny", model_revision="r1",
        model_digest=digests["model"],
        tokenizer_digest=tokenizer_artifact_digest(paths["model"]),
        adapter_dir=paths["adapter"], protocol=proto,
        campaign_digest=plan_doc["digest"],
        qualification_record_digest=qual_doc["digest"])
    decision_doc = signed(signers["promotion"], {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": f"camp-{tag}",
        "campaign_plan_digest": plan_doc["digest"],
        "qualification_record_digest": qual_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "adapter": "L6",
        "adapter_artifact_digests": {seed_name:
                                   manifest["adapter_digest"]},
        "runtime_manifest_digests": {seed_name: manifest["digest"]},
        "authorized_at": TS - 60 if authorized_at is None
        else authorized_at,
        "expires_at": TS + 3600 if expires_at is None else expires_at})
    return {"decision": decision_doc, "qualification": qual_doc,
            "plan": plan_doc, "runtime_manifest": manifest,
            "artifact_root": digest(digests), "digests": digests,
            "paths": paths, "seed": seed_name}


def grant_for(signers, chain, *, backend="hf-peft",
              audience="local-supervisor", epoch=0):
    return issue_grant(
        signers["admission"],
        decision_digest=chain["decision"]["digest"],
        qualification_digest=chain["qualification"]["digest"],
        runtime_manifest_digest=chain["runtime_manifest"]["digest"],
        artifact_root_digest=chain["artifact_root"],
        backend_id=backend, audience_runtime_identity=audience,
        now=NOW, revocation_epoch=epoch)


def authority_docs(chain):
    return {k: chain[k] for k in
            ("decision", "qualification", "plan", "runtime_manifest")}


def revocation_store(signers, tmp_path, *, epoch=0, revoked=(),
                     revoked_keys=(), issued_at=None, valid_until=None):
    rstore = RevocationStore(Path(tmp_path) / "revocations")
    rstore.publish(RevocationSnapshotV2(
        epoch=epoch,
        issued_at=TS - 60 if issued_at is None else issued_at,
        valid_until=TS + 86400 if valid_until is None else valid_until,
        revoked_decision_digests=tuple(revoked),
        revoked_key_ids=tuple(revoked_keys)).to_doc(
            signer=signers["revocation"]))
    return rstore


def snapshot_provider(rstore, registry):
    return lambda: rstore.latest_valid(registry, now=NOW,
                                       require=False)


def drive(sup, signers, tmp_path, backend, *, tag="a",
          snapshot_root=None):
    """Full activation with a real chain; retains the signed docs."""
    from minagi.v161.immutable_snapshot import stage_snapshot
    chain = docs(signers, tmp_path, tag,
                 backend=getattr(backend, "backend_id", "hf-peft"))
    grant = grant_for(signers, chain,
                      backend=getattr(backend, "backend_id", "hf-peft"))
    aid = sup.request().activation_id
    sup.authorize(aid, grant, authority_docs=authority_docs(chain))
    dest = (Path(snapshot_root) / aid if snapshot_root
            else Path(tmp_path) / "snap" / tag)
    sup.stage(aid, stage_snapshot(
        dest, chain["paths"], expected_digests=chain["digests"],
        manifest_digest=chain["runtime_manifest"]["digest"]))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid, chain
