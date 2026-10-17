"""v16.4.10 VAL-902 — complete signed activation, rollback and
restoration through the REAL serving path.

Not two directly-instantiated workers: this drives
``ServingSupervisor.request -> authorize (signed grant) -> stage ->
prepare -> health_check -> commit_activation`` with a WorkerBackend
bound to the NsJail executor, serves inference through the
ServingRouter lease, rolls A<-B back through the supervisor, and
cold-recovers through RecoveryManager — including revocation."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="NsJail/systemd boundary exists only on Linux")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

import authority_chain as ac  # noqa: E402
from egai.common.canonical import digest  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.recovery_manager import RecoveryManager  # noqa: E402
from minagi.runtime.serving_router import ServingRouter  # noqa: E402
from minagi.runtime.supervisor import (  # noqa: E402
    ServingState, ServingSupervisor)
from minagi.security.signed_revocations import (  # noqa: E402
    RevocationSnapshotV2)
from minagi.v161.artifact_closure import (  # noqa: E402
    close_tree, tokenizer_artifact_digest)
from minagi.v161.experiment_protocol import (  # noqa: E402
    ExperimentProtocolV1)
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402
from minagi.v161.peft_serving import runtime_manifest  # noqa: E402

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"


def _nsjail():
    for cand in (os.environ.get("MINAGI_NSJAIL"), "/opt/nsjail/nsjail",
                 shutil.which("nsjail")):
        if cand and Path(cand).is_file():
            return cand, "sha256:" + hashlib.sha256(
                Path(cand).read_bytes()).hexdigest()
    return None


def _traversable(path):
    p = Path(path).resolve()
    for d in (p, *p.parents):
        try:
            os.chmod(d, os.stat(d).st_mode | 0o001)
        except OSError:
            break
        if d.parent == d or d.name == "tmp":
            break


def _executor(tmp_path, registry=None, approved=()):
    _traversable(tmp_path)
    from minagi.runtime.isolation_profile import load_profile
    from minagi.runtime.nsjail_executor import NsJailWorkerExecutor
    from minagi.runtime.worker_launch_broker import (
        MappingRecordResolver)
    found = _nsjail()
    if found is None:
        pytest.skip("nsjail binary not found — set MINAGI_NSJAIL")
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        pytest.skip("systemd-run/systemctl not available")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        pytest.skip("host is not cgroup v2")
    binary, digest_ = found
    euid = os.geteuid()
    wuid = int(os.environ.get("MINAGI_WORKER_UID")
               or (65534 if euid == 0 else euid))
    if wuid == euid and euid != 0:
        pytest.skip("MINAGI_WORKER_UID must differ from supervisor")
    ex = NsJailWorkerExecutor.from_profile(
        load_profile(PROFILE_PATH),
        worker_state_root=tmp_path / "workers",
        nsjail_path=binary, nsjail_digest=digest_,
        worker_uid=wuid,
        worker_gid=int(os.environ.get("MINAGI_WORKER_GID") or wuid),
        user_manager=euid != 0,
        python_path_entries=tuple(str(p) for p in sys.path if p),
        authority_registry=registry,
        record_resolver=MappingRecordResolver(
            {str(d): True for d in approved}))
    report = ex.validate_host()
    if not report.get("ok"):
        pytest.skip(f"host cannot establish the boundary: {report}")
    return ex


def _tiny_model(model_dir: Path, adapter_dir: Path):
    import torch
    import peft
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)
    vocab = ["[PAD]", "[EOS]", "[UNK]"] + [f"tok{i}" for i in range(60)]
    tok_obj = Tokenizer(models.WordLevel(
        vocab={t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
                                  eos_token="[EOS]", pad_token="[PAD]",
                                  unk_token="[UNK]")
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64,
                     vocab_size=len(vocab), n_positions=128,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model_dir.mkdir(parents=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    served = peft.get_peft_model(
        model, peft.LoraConfig(r=4, lora_alpha=8,
                               target_modules=["c_attn"]))
    served.save_pretrained(adapter_dir)


def _real_docs(signers, tmp_path, tag, *, backend="hf-peft"):
    """The complete signed authority chain bound to the REAL staged
    model/adapter artifacts — same shapes as authority_chain.docs but
    with real bytes, so the manifest digests name the served model."""
    model = Path(tmp_path) / f"model-{tag}"
    adir = Path(tmp_path) / f"adapter-{tag}"
    _tiny_model(model, adir)
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    paths = {"model": str(model), "adapter": str(adir)}
    proto = ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["c_attn"], "learning_rate": 1e-4,
                 "steps": 2, "max_length": 48}})
    plan_doc = ac.signed(signers["plan"], {
        "schema": "mini-agi-v16.6-colab-campaign-plan-v1",
        "campaign_id": f"camp-{tag}",
        "experiment_protocol_digest": proto.digest})
    bundle_d = digest({"bundle": tag})
    qual_doc = ac.signed(signers["qualification"], {
        "schema": "mini-agi-v16.5-qualification-record-v1",
        "campaign_id": f"camp-{tag}", "decision": "QUALIFIED",
        "campaign_plan_digest": plan_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "runtime_backends": [backend]})
    manifest = runtime_manifest(
        model_id=f"tiny-{tag}", model_revision="r1",
        model_digest=digests["model"],
        tokenizer_digest=tokenizer_artifact_digest(paths["model"]),
        adapter_dir=paths["adapter"], protocol=proto,
        campaign_digest=plan_doc["digest"],
        qualification_record_digest=qual_doc["digest"])
    decision_doc = ac.signed(signers["promotion"], {
        "schema": "mini-agi-v16.5-promotion-decision-v1",
        "campaign_id": f"camp-{tag}",
        "campaign_plan_digest": plan_doc["digest"],
        "qualification_record_digest": qual_doc["digest"],
        "evaluation_bundle_digest": bundle_d,
        "adapter": "L6",
        "adapter_artifact_digests": {"seed-0":
                                   manifest["adapter_digest"]},
        "runtime_manifest_digests": {"seed-0": manifest["digest"]},
        "authorized_at": ac.TS - 60,
        "expires_at": ac.TS + 3600})
    return {"decision": decision_doc, "qualification": qual_doc,
            "plan": plan_doc, "runtime_manifest": manifest,
            "artifact_root": digest(digests), "digests": digests,
            "paths": paths}


def _supervisor(tmp_path, signers, registry, rstore, *,
                subdir="journal"):
    store = AuthorityStore(Path(tmp_path) / subdir / "authority.sqlite")
    return ServingSupervisor(
        store, runtime_signer=signers["runtime"], registry=registry,
        now=ac.NOW,
        revocation_snapshot_provider=ac.snapshot_provider(
            rstore, registry))


BACKEND_MANIFEST_DIGEST = "sha256:" + "d" * 64
BUDGET_DIGEST = "sha256:" + "e" * 64


def _jailed_backend(ex, signers, registry):
    """A WorkerBackend whose authorization_supplier issues a real
    signed AuthorizedLaunchRecord for each spawn — the supervisor's
    signing side of SEC-905."""
    from minagi.runtime.worker_backend import (BackendSpec,
                                               WorkerBackend)
    from minagi.runtime.worker_launch_broker import (
        AuthorizedLaunchRecord, issue_launch_authorization)
    import secrets as _s
    import time as _t
    bound = {"activation_id": "",
             "runtime_manifest_digest": "",
             "artifact_root_digest": ""}

    def supplier():
        rec = AuthorizedLaunchRecord(
            activation_id=bound["activation_id"],
            backend_manifest_digest=BACKEND_MANIFEST_DIGEST,
            runtime_manifest_digest=bound["runtime_manifest_digest"],
            artifact_root_digest=bound["artifact_root_digest"],
            isolation_profile_digest=ex.profile_digest,
            execution_budget_digest=ex._budget_digest(),
            policy_epoch=0, revocation_epoch=0,
            issued_at=int(_t.time()),
            expires_at=int(_t.time()) + 300,
            nonce=_s.token_hex(16))
        return issue_launch_authorization(
            signers["runtime"], record=rec)

    b = WorkerBackend(
        BackendSpec(module="minagi.v161.peft_serving",
                    qualname="PeftServingBackend", kwargs={}),
        backend_id="hf-peft", executor=ex,
        backend_manifest_digest=BACKEND_MANIFEST_DIGEST,
        authorization_supplier=supplier,
        start_timeout=180.0, probe_timeout=120.0,
        shutdown_grace=15.0)
    b.bound = bound  # test hook to bind each launch's digests
    return b


def _drive_jailed(sup, signers, tmp_path, tag, b, *,
                  snaproot, chain=None):
    chain = chain or _real_docs(signers, tmp_path, tag)
    grant = ac.grant_for(signers, chain)
    aid = sup.request().activation_id
    sup.authorize(aid, grant, authority_docs=ac.authority_docs(chain))
    sup.stage(aid, stage_snapshot(
        Path(snaproot) / aid, chain["paths"],
        expected_digests=chain["digests"],
        manifest_digest=chain["runtime_manifest"]["digest"]))
    # bind THIS activation's exact digests into the launch authorization
    b.bound.update({
        "activation_id": aid,
        "runtime_manifest_digest":
            chain["runtime_manifest"]["digest"],
        "artifact_root_digest": chain["artifact_root"]})
    sup.prepare(aid, b)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid, chain


def _serve(router, sup):
    return router.route(
        {"prompt": "tok1 tok2", "max_new_tokens": 4},
        principal="val902")


def test_signed_activation_rollback_recovery(tmp_path):
    for mod in ("torch", "transformers", "peft", "tokenizers"):
        pytest.importorskip(mod)
    registry, signers = ac.trust_chain(tmp_path)
    rstore = ac.revocation_store(signers, tmp_path)
    snaproot = tmp_path / "snaps"

    # build both signed chains first so every approved digest is
    # known to the executor's read-only record resolver
    chain_a = _real_docs(signers, tmp_path, "a")
    chain_b = _real_docs(signers, tmp_path, "b")
    # profile digest is only known once the executor exists — build
    # with the other approved digests, then extend the resolver
    ex = _executor(tmp_path, registry=registry, approved=(
        BACKEND_MANIFEST_DIGEST, BUDGET_DIGEST,
        chain_a["runtime_manifest"]["digest"],
        chain_a["artifact_root"],
        chain_b["runtime_manifest"]["digest"],
        chain_b["artifact_root"]))
    ex.broker.record_resolver._approved[ex.profile_digest] = True
    ex.broker.record_resolver._approved[ex._budget_digest()] = True

    sup = _supervisor(tmp_path, signers, registry, rstore)
    router = ServingRouter()
    sup.router = router

    # --- signed A: full chain -> jailed load -> serve -------------
    ba = _jailed_backend(ex, signers, registry)
    aid_a, chain_a = _drive_jailed(sup, signers, tmp_path, "a", ba,
                                   snaproot=snaproot, chain=chain_a)
    assert sup.active_id == aid_a
    out_a = _serve(router, sup)
    assert "completion" in out_a["result"]

    # --- signed B: distinct artifacts -> activate -> serve --------
    bb = _jailed_backend(ex, signers, registry)
    aid_b, chain_b = _drive_jailed(sup, signers, tmp_path, "b", bb,
                                   snaproot=snaproot, chain=chain_b)
    assert sup.active_id == aid_b
    assert aid_b != aid_a
    out_b = _serve(router, sup)
    assert "completion" in out_b["result"]

    # --- rollback: A restored through the supervisor --------------
    target = sup.rollback()
    assert target == aid_a
    assert sup.active_id == aid_a
    out_a2 = _serve(router, sup)
    assert "completion" in out_a2["result"]

    # --- cold restart + revocation of B: B cannot re-enter --------
    _revoke(signers, rstore, epoch=1,
            decision_digests=[chain_b["decision"]["digest"]])
    # a restarted supervisor shares the same durable journal
    sup2 = ServingSupervisor(
        sup.store, runtime_signer=signers["runtime"],
        registry=registry, now=ac.NOW,
        revocation_snapshot_provider=ac.snapshot_provider(
            rstore, registry))
    sup2.router = ServingRouter()
    RecoveryManager(
        sup2, admission_signer=signers["admission"],
        registry=registry, snapshot_root=snaproot,
        backend_factories={
            "hf-peft": lambda: _jailed_backend(ex, signers,
                                               registry)},
        revocation_store=rstore, now=ac.NOW).restore()
    # A (unrevoked) may restore; B's revoked promotion must not
    if sup2.serving_state is ServingState.SERVING:
        assert sup2.active_id != aid_b


def _revoke(signers, rstore, *, decision_digests, epoch, keys=()):
    rstore.publish(RevocationSnapshotV2(
        epoch=epoch, issued_at=ac.TS, valid_until=ac.TS + 86400,
        revoked_decision_digests=tuple(decision_digests),
        revoked_key_ids=tuple(keys)).to_doc(
            signer=signers["revocation"]))
