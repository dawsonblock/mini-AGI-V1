"""v16.2.1 — corrected LoRA training + PEFT serving-path closure tests.

Runs on CPU with a tiny locally-constructed GPT-2 and word-level
tokenizer — no network, no GPU. Covers the Phase-4/5 requirements:
response-only masking, padding exclusion, malformed-example rejection,
gradient-accumulation + optimizer validation, receipt contents,
adapter save/reload parity, and runtime-manifest closure checks.
"""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel,  # noqa: E402
                          PreTrainedTokenizerFast)

from minagi.v161.arms import shuffled_label_examples  # noqa: E402
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.peft_serving import (AdapterClosureError,  # noqa: E402
                                      adapter_closure,
                                      assert_adapter_matches_protocol,
                                      logits_digest, runtime_manifest)
from minagi.platforms.cuda.peft_trainer import (  # noqa: E402
    LoraTrainSpec, _pad_batch, supervised_tokens, train_lora)

VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask", "me",
         "ans", "ok", "a1", "b2", "c3", "d4", "x", "y", "hello", "world"]

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")


def _tokenizer(with_template=True):
    vocab = {t: i for i, t in enumerate(VOCAB)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=t,
                                   eos_token="[EOS]", pad_token="[PAD]",
                                   unk_token="[UNK]")
    if with_template:
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


# ------------------------------------------------------------- tokenization

def test_response_masking_marks_prompt_minus_100():
    tok = _tokenizer()
    enc = supervised_tokens(tok, "ask hello", "ans a1", 64)
    labels, ids = enc["labels"], enc["input_ids"]
    assert len(labels) == len(ids)
    sup = [t for t in labels if t != -100]
    # the supervised suffix decodes to the response, prompt is masked
    n_sup = enc["n_supervised"]
    assert n_sup == len(sup) > 0
    assert all(t == -100 for t in labels[:-n_sup])
    assert enc["n_prompt"] == len(labels) - n_sup


def test_response_masking_uses_chat_template_boundary():
    tok = _tokenizer()
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": "ask hello"}],
        add_generation_prompt=True, tokenize=False)
    assert prompt_text == "user: ask hello assistant:"
    enc = supervised_tokens(tok, "ask hello", "ans a1", 64)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    # prompt ids occupy the masked prefix exactly
    assert enc["input_ids"][:len(p_ids)] == p_ids
    assert enc["labels"][:len(p_ids)] == [-100] * len(p_ids)


def test_no_template_fallback_masks_prompt():
    tok = _tokenizer(with_template=False)
    enc = supervised_tokens(tok, "ask hello", "ans a1", 64)
    assert enc["n_supervised"] > 0
    assert all(t == -100 for t in enc["labels"][:enc["n_prompt"]])


def test_malformed_and_fully_truncated_examples_rejected():
    tok = _tokenizer()
    with pytest.raises(ValueError):
        supervised_tokens(tok, "ask hello", "  ", 64)      # empty response
    with pytest.raises(ValueError):
        supervised_tokens(tok, "", "ans a1", 64)           # empty prompt
    # truncation keeps the supervised tail — response survives
    enc = supervised_tokens(tok, "ask hello world x y", "ans a1", 4)
    assert enc["n_supervised"] > 0
    assert len(enc["input_ids"]) == 4

    class _NoResponseTokens:
        chat_template = None
        eos_token = ""
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2] if "ask" in text else []}
    with pytest.raises(ValueError):
        supervised_tokens(_NoResponseTokens(), "ask hello",
                          "ans a1", 64)


def test_padding_excluded_from_loss_positions():
    tok = _tokenizer()
    a = supervised_tokens(tok, "ask hello", "ans a1", 64)
    b = supervised_tokens(tok, "ask hello world x y me ok", "ans b2 c3", 64)
    ids, labels, attn = _pad_batch([a, b], 0)
    assert len(ids[0]) == len(ids[1]) == len(b["input_ids"])
    for i in range(len(a["input_ids"]), len(ids[0])):
        assert labels[0][i] == -100 and attn[0][i] == 0 and ids[0][i] == 0


def test_spec_validation_rejects_bad_optimizer_and_accum():
    with pytest.raises(ValueError):
        LoraTrainSpec(optimizer="not-an-optimizer")
    with pytest.raises(ValueError):
        LoraTrainSpec(gradient_accumulation_steps=0)


# ------------------------------------------------------------------ trainer

def test_train_lora_examples_masked_receipt(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(rank=4, alpha=8, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=3, max_length=48, seed=5,
                         response_only_loss=True)
    model, receipt = train_lora(model=model, tokenizer=tok,
                                examples=EXAMPLES,
                                output_dir=tmp_path / "ad", spec=spec)
    assert receipt["supervision_mode"] == "response-masked"
    assert len(receipt["loss_curve"]) == spec.steps
    assert receipt["trainable_params"] > 0
    assert receipt["trainable_params"] < receipt["total_params"]
    assert receipt["example_count"] == len(EXAMPLES)
    assert receipt["data_order_digest"].startswith("sha256:")
    assert receipt["library_versions"]["torch"] != "unknown"
    assert (tmp_path / "ad" / "adapter_model.safetensors").is_file()


def test_train_lora_grad_accumulation(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(rank=4, alpha=8, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=2, gradient_accumulation_steps=2,
                         max_length=48, seed=5, response_only_loss=True)
    _, receipt = train_lora(model=model, tokenizer=tok,
                            examples=EXAMPLES,
                            output_dir=tmp_path / "ad", spec=spec)
    # steps*gacc micro-losses recorded, opt stepped `steps` times
    assert len(receipt["loss_curve"]) == 4
    assert all(loss == loss and loss > 0 for loss in receipt["loss_curve"])


def test_every_example_rejected_fails(tmp_path):
    model, tok = _model(), _tokenizer()
    with pytest.raises(ValueError):
        train_lora(model=model, tokenizer=tok,
                   examples=[{"prompt": "ask", "response": ""}],
                   output_dir=tmp_path / "ad", spec=LoraTrainSpec())


def test_data_order_digest_binds_order(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(target_modules=("c_attn",), steps=1,
                         max_length=48)
    _, r1 = train_lora(model=model, tokenizer=tok, examples=EXAMPLES,
                       output_dir=tmp_path / "a", spec=spec)
    model2 = _model()
    _, r2 = train_lora(model=model2, tokenizer=tok,
                       examples=list(reversed(EXAMPLES)),
                       output_dir=tmp_path / "b", spec=spec)
    assert r1["data_order_digest"] != r2["data_order_digest"]


def test_shuffled_label_examples_deranged_and_deterministic():
    rows = [{"prompt": f"p{i}", "expected": f"e{i}"} for i in range(8)]
    ex = shuffled_label_examples(rows, 3)
    assert all(ex[i]["response"] != rows[i]["expected"] for i in range(8))
    assert {e["response"] for e in ex} == {r["expected"] for r in rows}
    assert ex == shuffled_label_examples(rows, 3)


# ------------------------------------------------------ serving path (A)

def _trained_adapter(tmp_path):
    model, tok = _model(), _tokenizer()
    spec = LoraTrainSpec(rank=4, alpha=8, dropout=0.0,
                         target_modules=("c_attn",), learning_rate=1e-3,
                         steps=2, max_length=48, seed=5,
                         response_only_loss=True)
    model, receipt = train_lora(model=model, tokenizer=tok,
                                examples=EXAMPLES,
                                output_dir=tmp_path / "ad", spec=spec)
    model.eval()
    return model, tok, tmp_path / "ad", spec


def test_adapter_reload_parity_bit_exact(tmp_path):
    model, tok, adir, spec = _trained_adapter(tmp_path)
    prompts = ["ask hello", "ask world"]
    with torch.inference_mode():
        trained_d = logits_digest(model, tok, prompts)
    # fresh base + reload — same artifact bytes must reproduce outputs
    base = _model()
    model.config.use_cache = True
    reloaded = peft.PeftModel.from_pretrained(base, str(adir))
    reloaded.eval()
    assert logits_digest(reloaded, tok, prompts) == trained_d


def test_adapter_closure_and_protocol_match(tmp_path):
    _, _, adir, _ = _trained_adapter(tmp_path)
    c = adapter_closure(adir)
    assert c.target_modules == ("c_attn",)
    proto = ExperimentProtocolV1(
        lora_rank=4, lora_alpha=8, lora_dropout=0.0,
        lora_target_modules=("c_attn",), learning_rate=1e-3,
        train_steps=2, train_max_length=48,
        require_native_servable_adapter=False)
    assert assert_adapter_matches_protocol(adir, proto) == c


def test_adapter_closure_rejects_mismatch(tmp_path):
    _, _, adir, _ = _trained_adapter(tmp_path)
    proto = ExperimentProtocolV1(
        lora_rank=8, lora_alpha=8, lora_dropout=0.0,
        lora_target_modules=("c_attn",), learning_rate=1e-3,
        train_steps=2, train_max_length=48,
        require_native_servable_adapter=False)
    with pytest.raises(AdapterClosureError):
        assert_adapter_matches_protocol(adir, proto)
    proto2 = ExperimentProtocolV1(
        lora_rank=4, lora_alpha=8, lora_dropout=0.0,
        lora_target_modules=("c_attn", "c_proj"), learning_rate=1e-3,
        train_steps=2, train_max_length=48,
        require_native_servable_adapter=False)
    with pytest.raises(AdapterClosureError):
        assert_adapter_matches_protocol(adir, proto2)


def test_runtime_manifest_binds_exact_artifacts(tmp_path):
    _, _, adir, _ = _trained_adapter(tmp_path)
    proto = ExperimentProtocolV1(
        lora_rank=4, lora_alpha=8, lora_dropout=0.0,
        lora_target_modules=("c_attn",), learning_rate=1e-3,
        train_steps=2, train_max_length=48,
        require_native_servable_adapter=False)
    man = runtime_manifest(model_id="tiny", model_revision="r1",
                           model_digest="sha256:" + "1" * 64,
                           tokenizer_digest="sha256:" + "2" * 64,
                           adapter_dir=adir, protocol=proto)
    assert man["adapter_digest"] == adapter_closure(adir).adapter_digest
    assert man["lora"]["target_modules"] == ["c_attn"]
    with pytest.raises(AdapterClosureError):
        runtime_manifest(model_id="tiny", model_revision="r1",
                         model_digest="sha256:" + "1" * 64,
                         tokenizer_digest="sha256:" + "2" * 64,
                         adapter_dir=adir, protocol=proto,
                         expected_adapter_digest="sha256:" + "9" * 64)


def test_non_adapter_dir_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AdapterClosureError):
        adapter_closure(empty)
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / "adapter_config.json").write_text(
        json.dumps({"peft_type": "PREFIX_TUNING",
                    "target_modules": ["x"]}))
    with pytest.raises(AdapterClosureError):
        adapter_closure(bogus)
