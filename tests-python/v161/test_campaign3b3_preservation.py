"""Campaign 3B.3 P2/P3 — rehearsal interleave + frozen-reference KL.

P2 rehearsal: a deterministic share of microbatches draw VERIFIED
pairs into the gradient stream — the mechanism is optimization-time,
distinct from L7's inference-time replay. The receipt must bind the
schedule, the realized fraction, the corpus digest, and the per-source
sample order.

P3 KL: kl_weight > 0 adds mean KL(frozen || adapted) on a prompt set,
computed against the SAME weights with the adapter disabled, and the
receipt binds the reference parameter digest — "the frozen reference"
is a measured artifact, not a name.

Runs on CPU with the tiny locally-constructed GPT-2 — no network.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (GPT2Config, GPT2LMHeadModel,  # noqa: E402
                          PreTrainedTokenizerFast)

from minagi.platforms.cuda.peft_trainer import (  # noqa: E402
    LoraTrainSpec, train_lora)

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


TASK = [{"prompt": "ask hello", "response": "ans a1"},
        {"prompt": "ask world", "response": "ans b2"},
        {"prompt": "ask me", "response": "ans c3"},
        {"prompt": "ask ok", "response": "ans d4"}]

REHEARSAL = [{"prompt": "ask x", "response": "ans x"},
             {"prompt": "ask y", "response": "ans y"}]

KL_PROMPTS = ["ask hello world", "ask me ok"]


def _spec(**kw):
    base = dict(rank=4, alpha=8, dropout=0.0, target_modules=("c_attn",),
                learning_rate=1e-3, steps=8, max_length=48, seed=5,
                response_only_loss=True)
    base.update(kw)
    return LoraTrainSpec(**base)


# ---------- P2: rehearsal interleave -------------------------------------

def test_rehearsal_fraction_interleaves_deterministically(tmp_path):
    """fraction=0.25 → every 4th microbatch draws from the rehearsal
    pool; the receipt binds realized fraction, corpus identity, and
    per-source order."""
    spec = _spec(rehearsal_fraction=0.25, gradient_accumulation_steps=2,
                 microbatch_size=1)
    _, receipt = train_lora(model=_model(), tokenizer=_tokenizer(),
                            examples=TASK, output_dir=tmp_path / "ad",
                            spec=spec, rehearsal_examples=REHEARSAL)
    # 8 steps x gacc 2 = 16 microbatches; every 4th is rehearsal → 4
    assert receipt["rehearsal_presentations"] == 4
    assert receipt["sample_presentations"] == 16
    assert receipt["rehearsal_fraction_realized"] == 0.25
    assert receipt["rehearsal_period_microbatches"] == 4
    assert receipt["rehearsal_corpus_digest"].startswith("sha256:")
    assert receipt["rehearsal_corpus_size"] == 2


def test_rehearsal_fraction_zero_never_samples_rehearsal(tmp_path):
    spec = _spec(rehearsal_fraction=0.0)
    _, receipt = train_lora(model=_model(), tokenizer=_tokenizer(),
                            examples=TASK, output_dir=tmp_path / "ad",
                            spec=spec, rehearsal_examples=REHEARSAL)
    assert receipt["rehearsal_presentations"] == 0
    assert receipt["rehearsal_fraction_realized"] == 0.0
    # corpus was bound but never presented — honest accounting
    assert receipt["rehearsal_corpus_digest"].startswith("sha256:")


def test_rehearsal_requires_corpus(tmp_path):
    """rehearsal_fraction > 0 with no rehearsal_examples refuses — a
    fraction over an empty pool would silently degrade to plain LoRA."""
    spec = _spec(rehearsal_fraction=0.5)
    with pytest.raises(ValueError, match="rehearsal_examples"):
        train_lora(model=_model(), tokenizer=_tokenizer(), examples=TASK,
                   output_dir=tmp_path / "ad", spec=spec)


def test_rehearsal_same_seed_same_order(tmp_path):
    spec = _spec(rehearsal_fraction=0.5)
    _, r1 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "a",
                       spec=spec, rehearsal_examples=REHEARSAL)
    _, r2 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "b",
                       spec=spec, rehearsal_examples=REHEARSAL)
    assert r1["sample_order_digest"] == r2["sample_order_digest"]
    # and it differs from a no-rehearsal run's order
    _, r3 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "c",
                       spec=_spec())
    assert r1["sample_order_digest"] != r3["sample_order_digest"]


def test_rehearsal_spec_validation():
    with pytest.raises(ValueError, match="rehearsal_fraction"):
        LoraTrainSpec(rehearsal_fraction=-0.1)
    with pytest.raises(ValueError, match="rehearsal_fraction"):
        LoraTrainSpec(rehearsal_fraction=1.5)
    assert LoraTrainSpec(rehearsal_fraction=0.25).rehearsal_period == 4
    assert LoraTrainSpec(rehearsal_fraction=1.0).rehearsal_period == 1
    assert LoraTrainSpec().rehearsal_period is None


# ---------- P3: frozen-reference KL ---------------------------------------

def test_kl_preservation_records_reference_and_term(tmp_path):
    spec = _spec(kl_weight=0.1, kl_every=2)
    _, receipt = train_lora(model=_model(), tokenizer=_tokenizer(),
                            examples=TASK, output_dir=tmp_path / "ad",
                            spec=spec, kl_prompts=KL_PROMPTS)
    kl = receipt["kl_preservation"]
    assert kl["kl_weight"] == 0.1
    assert kl["kl_every"] == 2
    # 8 microbatches, kl every 2nd → 4 evaluations
    assert kl["n_kl_evaluations"] == 4
    assert kl["kl_term_mean"] is not None
    assert kl["kl_prompt_count"] == 2
    assert kl["kl_prompts_digest"].startswith("sha256:")
    # the reference is bound by its actual weights, not a name
    assert kl["reference_model"]["weights_digest"]
    assert kl["reference_model"]["binding"] == \
        "same-weights, adapter-disabled forward"


def test_kl_requires_prompts(tmp_path):
    spec = _spec(kl_weight=0.1)
    with pytest.raises(ValueError, match="kl_prompts"):
        train_lora(model=_model(), tokenizer=_tokenizer(), examples=TASK,
                   output_dir=tmp_path / "ad", spec=spec)


def test_kl_changes_the_optimized_loss(tmp_path):
    """A nonzero KL weight must alter the objective — otherwise the
    preservation term is decorative. Compare loss curves at a strong
    weight; they must differ somewhere."""
    _, r0 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "a",
                       spec=_spec(kl_weight=0.0))
    _, r1 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "b",
                       spec=_spec(kl_weight=5.0), kl_prompts=KL_PROMPTS)
    assert r0["loss_curve"] != r1["loss_curve"]


def test_kl_reference_digest_identifies_the_base(tmp_path):
    """Two runs on the same base must produce the same reference
    digest; that digest is what anchors 'frozen reference' claims."""
    spec = _spec(kl_weight=0.1)
    _, r1 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "a",
                       spec=spec, kl_prompts=KL_PROMPTS)
    _, r2 = train_lora(model=_model(), tokenizer=_tokenizer(),
                       examples=TASK, output_dir=tmp_path / "b",
                       spec=spec, kl_prompts=KL_PROMPTS)
    assert (r1["kl_preservation"]["reference_model"]["weights_digest"]
            == r2["kl_preservation"]["reference_model"]
            ["weights_digest"])


def test_kl_spec_validation():
    with pytest.raises(ValueError, match="kl_weight"):
        LoraTrainSpec(kl_weight=-1.0)
    with pytest.raises(ValueError, match="kl_every"):
        LoraTrainSpec(kl_every=0)


# ---------- both mechanisms together --------------------------------------

def test_rehearsal_and_kl_compose(tmp_path):
    spec = _spec(rehearsal_fraction=0.25, kl_weight=0.1, kl_every=2)
    _, receipt = train_lora(model=_model(), tokenizer=_tokenizer(),
                            examples=TASK, output_dir=tmp_path / "ad",
                            spec=spec, rehearsal_examples=REHEARSAL,
                            kl_prompts=KL_PROMPTS)
    assert receipt["rehearsal_presentations"] > 0
    assert receipt["kl_preservation"]["n_kl_evaluations"] > 0
    assert receipt["optimizer_updates"] == 8
