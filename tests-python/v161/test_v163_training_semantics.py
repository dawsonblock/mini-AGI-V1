"""v16.3.0 Phase 2.1 — explicit LoRA training schedule.

Reproduces and closes the v16.2.1 defect: the trainer placed
`gradient_accumulation_steps` examples in each microbatch *and*
accumulated over `gradient_accumulation_steps` microbatches, silently
squaring the effective batch size and leaving no explicit microbatch
size anywhere in the signed surface.

The corrected contract:

    effective_batch_size = microbatch_size * gradient_accumulation_steps
    optimizer_updates    = steps
    sample_presentations = steps * effective_batch_size

Counts are asserted at the *engine* level (forward rows, optimizer step
calls), not just against receipt arithmetic. Runs on CPU with a tiny
locally-constructed GPT-2 — no network, no GPU.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel,  # noqa: E402
                          PreTrainedTokenizerFast)

import minagi.platforms.cuda.peft_trainer as trainer  # noqa: E402
from minagi.platforms.cuda.peft_trainer import (  # noqa: E402
    LoraTrainSpec, adapter_artifact_digest, train_lora)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import adapter_closure  # noqa: E402
from minagi.v161.runtime_closure3 import sha256_path  # noqa: E402

VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask", "me",
         "ans", "ok", "a1", "b2", "c3", "d4", "x", "y", "hello", "world"]

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")


def _tokenizer():
    vocab = {t: i for i, t in enumerate(VOCAB)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t,
                                   eos_token="[EOS]", pad_token="[PAD]",
                                   unk_token="[UNK]")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def _model():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32,
                     vocab_size=len(VOCAB), n_positions=64,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg)


EXAMPLES = [{"prompt": "ask hello", "response": "ans a1"},
            {"prompt": "ask world", "response": "ans b2"},
            {"prompt": "ask me", "response": "ans c3"},
            {"prompt": "ask ok", "response": "ans d4"}]


# ---------- engine-level schedule enforcement --------------------------

def test_engine_counts_match_declared_schedule(tmp_path, monkeypatch):
    """microbatch=2, accumulation=4, steps=3 => 12 forwards of 2 rows,
    3 optimizer updates, 24 sample presentations."""
    model, tok = _model(), _tokenizer()

    forward_rows = []
    orig_forward = type(model).forward

    def counting_forward(self, *a, **kw):
        ids = kw.get("input_ids", a[0] if a else None)
        forward_rows.append(int(ids.shape[0]))
        return orig_forward(self, *a, **kw)

    monkeypatch.setattr(type(model), "forward", counting_forward)

    real_make = trainer._make_optimizer
    step_calls = []

    def counting_make(model_, spec_):
        opt = real_make(model_, spec_)
        orig_step = opt.step

        def step(*a, **k):
            step_calls.append(1)
            return orig_step(*a, **k)

        opt.step = step
        return opt

    monkeypatch.setattr(trainer, "_make_optimizer", counting_make)

    spec = LoraTrainSpec(rank=4, alpha=8, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=3, gradient_accumulation_steps=4,
                         microbatch_size=2, max_length=48, seed=5,
                         response_only_loss=True)
    _, receipt = train_lora(model=model, tokenizer=tok, examples=EXAMPLES,
                            output_dir=tmp_path / "ad", spec=spec)

    assert len(step_calls) == 3                      # optimizer updates
    assert len(forward_rows) == 12                   # microbatches
    assert set(forward_rows) == {2}                  # microbatch size
    assert receipt["optimizer_updates"] == 3
    assert receipt["sample_presentations"] == 24
    assert receipt["effective_batch_size"] == 8
    assert receipt["microbatch_size"] == 2
    assert receipt["gradient_accumulation_steps"] == 4
    assert len(receipt["loss_curve"]) == 12
    assert len(receipt["grad_norms"]) == 3


def test_spec_schedule_validation():
    with pytest.raises(ValueError, match="microbatch_size"):
        LoraTrainSpec(microbatch_size=0)
    with pytest.raises(ValueError, match="malformed_policy"):
        LoraTrainSpec(malformed_policy="ignore")
    spec = LoraTrainSpec(microbatch_size=2, gradient_accumulation_steps=4)
    assert spec.effective_batch_size == 8


def test_legacy_texts_path_keeps_step_semantics(tmp_path):
    """Legacy concatenated replay: one text per step, unchanged."""
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=3, max_length=48)
    _, receipt = train_lora(model=model, tokenizer=tok,
                            texts=["ask hello", "ask world"],
                            output_dir=tmp_path / "ad", spec=spec)
    assert receipt["supervision_mode"] == "concatenated-legacy"
    assert receipt["optimizer_updates"] == 3
    assert receipt["sample_presentations"] == 3
    assert receipt["sample_order_digest"].startswith("sha256:")


# ---------- malformed-sample policy ------------------------------------

def test_malformed_policy_fail_invalidates_run(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=2, max_length=48,
                         malformed_policy="fail")
    bad = EXAMPLES + [{"prompt": "ask", "response": "  "}]
    with pytest.raises(ValueError, match="invalidated"):
        train_lora(model=model, tokenizer=tok, examples=bad,
                   output_dir=tmp_path / "ad", spec=spec)
    assert not (tmp_path / "ad").exists()  # nothing persisted


def test_malformed_policy_reject_records_and_trains(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=2, max_length=48,
                         malformed_policy="reject")
    bad = EXAMPLES + [{"prompt": "ask", "response": "  "}]
    _, receipt = train_lora(model=model, tokenizer=tok, examples=bad,
                            output_dir=tmp_path / "ad", spec=spec)
    assert receipt["malformed_policy"] == "reject"
    assert receipt["rejected_examples"] == 1
    assert receipt["example_count"] == len(EXAMPLES)


# ---------- receipt bindings -------------------------------------------

def test_receipt_binds_adapter_artifact_digest(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(rank=4, alpha=8, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=2, max_length=48, seed=5,
                         response_only_loss=True)
    _, receipt = train_lora(model=model, tokenizer=tok, examples=EXAMPLES,
                            output_dir=tmp_path / "ad", spec=spec)
    adir = tmp_path / "ad"
    assert receipt["adapter_artifact_digest"] == \
        adapter_artifact_digest(adir)
    # the directory closure digest covers the receipt/spec evidence too —
    # a different (broader) scope, bound by runtime manifests
    closure = adapter_closure(adir)
    assert closure.adapter_digest == sha256_path(adir)
    assert closure.adapter_digest != receipt["adapter_artifact_digest"]
    # tampering with the receipt evidence changes the closure digest
    (adir / "TRAINING_RECEIPT.json").write_text("{}")
    assert adapter_closure(adir).adapter_digest != closure.adapter_digest


def test_sample_order_digest_binds_actual_sampling(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=2, max_length=48,
                         seed=5)
    _, r1 = train_lora(model=model, tokenizer=tok, examples=EXAMPLES,
                       output_dir=tmp_path / "a", spec=spec)
    model2 = _model()
    _, r2 = train_lora(model=model2, tokenizer=tok, examples=EXAMPLES,
                       output_dir=tmp_path / "b", spec=spec)
    model3 = _model()
    spec3 = LoraTrainSpec(target_modules=("c_attn",), steps=2,
                          max_length=48, seed=6)
    _, r3 = train_lora(model=model3, tokenizer=tok, examples=EXAMPLES,
                       output_dir=tmp_path / "c", spec=spec3)
    assert r1["sample_order_digest"] == r2["sample_order_digest"]
    assert r1["sample_order_digest"] != r3["sample_order_digest"]
    assert r1["sample_order_digest"] != r1["data_order_digest"]


def test_gradient_statistics_recorded(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=3,
                         gradient_accumulation_steps=2, microbatch_size=2,
                         max_length=48, seed=5)
    _, receipt = train_lora(model=model, tokenizer=tok, examples=EXAMPLES,
                            output_dir=tmp_path / "ad", spec=spec)
    norms = receipt["grad_norms"]
    assert len(norms) == 3
    assert all(n == n and n > 0 for n in norms)  # finite, non-zero
    assert receipt["grad_norm_mean"] == pytest.approx(sum(norms) / 3)


# ---------- protocol binding -------------------------------------------

def _cfg(lora_overrides=None):
    lora = {"rank": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": ["q_proj"], "learning_rate": 1e-4,
            "steps": 100, "max_length": 256, "optimizer": "adamw",
            "gradient_accumulation_steps": 4, "microbatch_size": 2,
            "malformed_policy": "fail"}
    lora.update(lora_overrides or {})
    return {"model": {"dtype": "bfloat16", "quantization": "none",
                      "trust_remote_code": False},
            "retention_scorer": "retention_score",
            "require_native_servable_adapter": False,
            "lora": lora}


def test_protocol_binds_schedule_and_policy():
    proto = ExperimentProtocolV1.from_config(_cfg())
    assert proto.train_microbatch_size == 2
    assert proto.effective_batch_size == 8
    assert proto.malformed_policy == "fail"
    spec = LoraTrainSpec(**proto.lora_train_spec_kwargs(), seed=0)
    assert spec.microbatch_size == 2
    assert spec.malformed_policy == "fail"
    assert spec.effective_batch_size == 8


def test_protocol_defaults_and_validation():
    cfg = _cfg()
    del cfg["lora"]["microbatch_size"]
    del cfg["lora"]["malformed_policy"]
    proto = ExperimentProtocolV1.from_config(cfg)
    assert proto.train_microbatch_size == 1
    assert proto.malformed_policy == "reject"
    with pytest.raises(ValueError, match="microbatch"):
        ExperimentProtocolV1.from_config(_cfg({"microbatch_size": 0}))
    with pytest.raises(ValueError, match="malformed_policy"):
        ExperimentProtocolV1.from_config(
            _cfg({"malformed_policy": "ignore"}))
