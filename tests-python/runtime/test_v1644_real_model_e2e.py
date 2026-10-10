"""v16.4.4 real-model end-to-end evidence (Experiments 2 & 4).

A real Hugging Face model — a locally-constructed single-layer GPT-2
with a word-level tokenizer, saved in HF format and served through the
REAL PeftServingBackend (torch inference, real generate()) — is
authorized, activated, queried, switched, rolled back and cold-start
restored. Offline by construction: every artifact is built in tmp_path.

  activate A -> real query -> activate B -> real query
  -> rollback to A -> real query
  -> simulated crash (fresh supervisor on same store)
  -> RecoveryManager reauthorizes/restores -> real query
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel,  # noqa: E402
                          PreTrainedTokenizerFast)
import peft  # noqa: E402

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.inference_policy import (  # noqa: E402
    InferenceBudgetPolicyV1)
from minagi.runtime.recovery_manager import RecoveryManager  # noqa: E402
from minagi.runtime.serving_router import ServingRouter  # noqa: E402
from minagi.runtime.supervisor import (  # noqa: E402
    ServingState, ServingSupervisor)
from minagi.security.admission_grants import issue_grant  # noqa: E402
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)
from minagi.v161.immutable_snapshot import stage_snapshot  # noqa: E402
from minagi.v161.peft_serving import PeftServingBackend  # noqa: E402

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)

VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask", "me",
         "ans", "ok", "a1", "b2", "c3", "d4", "x", "y", "hello", "world"]

BUDGET = InferenceBudgetPolicyV1(
    max_prompt_tokens=64, max_new_tokens=16,
    execution_deadline_seconds=30.0,
    max_concurrent_requests=2, max_queued_per_principal=4)


def _save_real_model(model_dir: Path, adapter_dir: Path, *, seed: int):
    vocab = {t: i for i, t in enumerate(VOCAB)}
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
                                  eos_token="[EOS]", pad_token="[PAD]",
                                  unk_token="[UNK]")
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32,
                     vocab_size=len(VOCAB), n_positions=64,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(cfg)
    model_dir.mkdir(parents=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    served = peft.get_peft_model(
        model, peft.LoraConfig(r=2, lora_alpha=4,
                               target_modules=["c_attn"]))
    served.save_pretrained(adapter_dir)


def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _real_artifacts(tmp_path, tag, *, seed):
    model_dir = tmp_path / f"model-{tag}"
    adapter_dir = tmp_path / f"adapter-{tag}"
    if not model_dir.exists():
        _save_real_model(model_dir, adapter_dir, seed=seed)
    return ({"model": close_tree(model_dir).digest,
             "adapter": close_tree(adapter_dir).digest},
            {"model": str(model_dir), "adapter": str(adapter_dir)})


def _drive(sup, signers, tmp_path, backend, *, tag, seed,
           snapshot_root):
    digests, paths = _real_artifacts(tmp_path, tag, seed=seed)
    grant = issue_grant(
        signers["admission"], decision_digest=digest({"d": tag}),
        qualification_digest=digest({"q": tag}),
        runtime_manifest_digest=digest({"m": tag}),
        artifact_root_digest=digest(digests), backend_id="hf-peft",
        audience_runtime_identity="local-supervisor", now=NOW,
        revocation_epoch=0)
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, stage_snapshot(
        Path(snapshot_root) / aid, paths,
        expected_digests=digests,
        manifest_digest=digest({"m": tag})))
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid


def _query(router, prompt, *, tokens=4):
    out = router.route({"prompt": prompt,
                        "max_new_tokens": tokens,
                        "request_id": f"q-{prompt}"})
    return out["result"]


def test_real_model_activate_query_rollback_restore(tmp_path):
    registry, signers = _chain(tmp_path)
    snaps = tmp_path / "snaps"
    router = ServingRouter(budget=BUDGET)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    sup.router = router

    # --- activate real model A and query it --------------------------
    aid_a = _drive(sup, signers, tmp_path,
                   PeftServingBackend(budget=BUDGET),
                   tag="a", seed=0, snapshot_root=snaps)
    assert sup.serving_state is ServingState.SERVING
    out_a = _query(router, "ask hello")
    assert isinstance(out_a["completion"], str)
    assert out_a["metrics"]["prompt_tokens"] > 0
    assert out_a["metrics"]["max_new_tokens_applied"] <= \
        BUDGET.max_new_tokens

    # --- activate real model B: live switching -----------------------
    aid_b = _drive(sup, signers, tmp_path,
                   PeftServingBackend(budget=BUDGET),
                   tag="b", seed=1, snapshot_root=snaps)
    dep = store.deployment()
    assert dep["desired_activation_id"] == aid_b
    assert dep["previous_activation_id"] == aid_a
    out_b = _query(router, "ask world")
    assert isinstance(out_b["completion"], str)

    # --- rollback: durable intent AND live routing return to A -------
    restored = sup.rollback()
    assert restored == aid_a
    dep = store.deployment()
    assert dep["desired_activation_id"] == aid_a
    assert dep["transition_phase"] == "ROUTED"
    out_a2 = _query(router, "ask ok")
    assert isinstance(out_a2["completion"], str)

    # --- simulated crash: fresh supervisor on the same store ---------
    router2 = ServingRouter(budget=BUDGET)
    sup2 = ServingSupervisor(
        AuthorityStore(tmp_path / "journal" / "authority.sqlite"),
        runtime_signer=signers["runtime"], registry=registry, now=NOW)
    sup2.router = router2
    report = sup2.recover()
    assert report["requires_restoration"] == aid_a
    assert sup2.serving_state is ServingState.RECOVERY_REQUIRED

    mgr = RecoveryManager(
        sup2, admission_signer=signers["admission"], registry=registry,
        snapshot_root=snaps,
        backend_factories={"hf-peft":
                           (lambda: PeftServingBackend(budget=BUDGET))},
        now=NOW)
    result = mgr.restore()
    assert result["restoration"]["status"] == "restored"
    new_id = result["restoration"]["restored_activation"]
    assert new_id != aid_a          # a NEW activation, fresh grant
    assert sup2.serving_state is ServingState.SERVING

    # --- the restored REAL model actually serves ---------------------
    out_r = _query(router2, "ask me")
    assert isinstance(out_r["completion"], str)
    assert out_r["metrics"]["prompt_tokens"] > 0
