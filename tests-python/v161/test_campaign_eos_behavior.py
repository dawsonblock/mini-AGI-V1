"""SCI-301/303 — end-of-turn supervision and stop behavior.

The adapter must be trained to emit the template's end-of-turn token
and generation must stop on it:

  * template-less tokenizers supervise the appended EOS,
  * chat templates supervise their rendered end-of-message token
    (pinned Qwen tokenizer),
  * ``generate`` observes a precise eos_hit from the final token,
  * an output contract that disables EOS stopping is refused.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.prompt_contract import (  # noqa: E402
    LABEL_IGNORE, OutputContract, build_prompt_contract,
    encode_supervised)

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (AutoTokenizer,  # noqa: E402
                          PreTrainedTokenizerFast)

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


def _qwen_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=True)
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
        except Exception:
            pytest.skip("pinned Qwen tokenizer not cached locally")


def test_no_template_appends_and_supervises_eos():
    tok = _tokenizer(with_template=False)
    enc = encode_supervised(tok, "ask hello", "ans a1",
                                  build_prompt_contract(tok), max_length=64)
    supervised = [t for t in enc["labels"] if t != LABEL_IGNORE]
    assert enc["eos_supervised"] is True
    assert supervised[-1] == tok.eos_token_id
    assert enc["input_ids"][-1] == tok.eos_token_id


def test_chat_template_end_of_turn_is_supervised():
    tok = _qwen_tokenizer()
    enc = encode_supervised(tok, "ask hello", "ans a1",
                                  build_prompt_contract(tok), max_length=256)
    supervised = [t for t in enc["labels"] if t != LABEL_IGNORE]
    assert enc["eos_supervised"] is True
    assert tok.eos_token_id in supervised, (
        "the chat template's end-of-turn token must carry supervised "
        "loss so the adapter learns to stop")


class _ScriptedModel:
    """Emits a fixed continuation after the prompt."""

    def __init__(self, continuation):
        self._param = torch.zeros(1)
        self.continuation = list(continuation)

    def parameters(self):
        yield self._param

    def generate(self, **batch):
        ids = [int(t) for t in batch["input_ids"][0]]
        return torch.tensor([ids + self.continuation], dtype=torch.long)


def _load_runner():
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location(
        "run_campaign1_eos_under_test", ROOT / "scripts" / "run_campaign1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generation_observes_eos_hit_precisely():
    runner = _load_runner()
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    # greedy stop: the model emits the end-of-turn token as its last
    # generated token
    model = _ScriptedModel([3, tok.eos_token_id])
    _, nt, eos_hit = runner.generate(model, tok, "ask hello", 8, contract)
    assert nt == 2 and eos_hit is True
    # limit hit: last token is not the end-of-turn token
    model = _ScriptedModel([3, 4])
    _, nt, eos_hit = runner.generate(model, tok, "ask hello", 8, contract)
    assert nt == 2 and eos_hit is False


def test_output_contract_refuses_disabled_eos_stop():
    with pytest.raises(ValueError, match="stop_on_eos"):
        OutputContract(instruction="", max_new_tokens=8, stop_on_eos=False)


def test_output_contract_validates_limit():
    with pytest.raises(ValueError):
        OutputContract(instruction="", max_new_tokens=0)
