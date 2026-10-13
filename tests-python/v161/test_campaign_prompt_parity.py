"""SCI-301 — training/inference prompt parity.

Campaign 3A trained LoRA adapters on chat-template-rendered pairs but
evaluated them on raw prompts. These tests pin the repaired contract:

  * training's prompt token sequence equals inference's input token
    sequence for the same task (stub tokenizer + the pinned Qwen
    tokenizer when it is cached locally),
  * the runner's ``generate`` renders through the shared contract — the
    test fails if a future runner evaluates raw prompts again,
  * the historical mismatch is demonstrated (raw tokenization differs
    from contract rendering), so the regression class stays visible.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.prompt_contract import (  # noqa: E402
    answer_only_output_contract, build_prompt_contract,
    render_generation_input, render_prompt_text, render_training_example)

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (AutoTokenizer,  # noqa: E402
                          PreTrainedTokenizerFast)

VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask", "me",
         "ans", "ok", "a1", "b2", "c3", "d4", "x", "y", "hello", "world",
         "reverse", "the", "string", "'", "olleh", "banana", "aaabnn"]

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")


def _tokenizer(with_template=True):
    vocab = {t: i for i, t in enumerate(VOCAB)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=t,
                                   eos_token="[EOS]", pad_token="[PAD]",
                                   unk_token="[UNK]")
    if with_template:
        fast.chat_template = CHAT_TEMPLATE
    return fast


def _qwen_tokenizer():
    """The pinned Qwen tokenizer, when it is cached locally (tests stay
    offline; CI without the cache skips, a cached host qualifies)."""
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
            pytest.skip("pinned Qwen tokenizer not cached locally — "
                        "download Qwen/Qwen2.5-0.5B-Instruct to qualify "
                        "with the real tokenizer")


TASKS = [
    ("Reverse the string 'hello'.", "olleh"),
    ("Sort the characters of 'banana'.", "aaabnn"),
    ("Transform 'x y' with unusual punctuation: ;:'\"", "y x"),
    ("A deliberately long prompt " + "with repeated context " * 40 +
     "that exceeds any reasonable budget.", "ok"),
]


def _assert_parity(tokenizer, output=None):
    contract = build_prompt_contract(tokenizer, output)
    for task, response in TASKS:
        gen_ids = render_generation_input(tokenizer, task, contract)
        enc = render_training_example(tokenizer, task, response, contract,
                                      max_length=4096)
        prefix = enc["input_ids"][:enc["prompt_tokens"]]
        assert prefix == gen_ids, (
            "training prompt tokens differ from inference input tokens "
            f"for task {task!r}")
        assert enc["input_ids"][:len(gen_ids)] == gen_ids


def test_stub_tokenizer_parity():
    _assert_parity(_tokenizer())


def test_stub_tokenizer_parity_with_answer_instruction():
    _assert_parity(_tokenizer(), answer_only_output_contract())


def test_pinned_qwen_tokenizer_parity():
    _assert_parity(_qwen_tokenizer())


def test_pinned_qwen_tokenizer_parity_with_answer_instruction():
    _assert_parity(_qwen_tokenizer(), answer_only_output_contract())


def test_historical_mismatch_is_real():
    """The 3A failure class: raw tokenization is NOT the rendered
    surface, for the stub tokenizer and for the pinned Qwen tokenizer."""
    for tok in (_tokenizer(), _qwen_tokenizer()):
        contract = build_prompt_contract(tok)
        task = "Reverse the string 'hello'."
        raw = list(tok(task, add_special_tokens=False)["input_ids"])
        rendered = render_generation_input(tok, task, contract)
        assert raw != rendered, (
            f"{type(tok).__name__}: raw and rendered tokenization are "
            "identical — this tokenizer has no chat surface to miss")
        # the rendered surface is the chat template's, not the raw text's
        assert render_prompt_text(tok, task, contract) != task


def _load_runner():
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location(
        "run_campaign1_under_test", ROOT / "scripts" / "run_campaign1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _RecordingModel:
    """Minimal model stub: records the input ids generation receives
    and appends three generated tokens to them."""

    def __init__(self, gen_ids):
        self._param = torch.zeros(1)
        self.gen_ids = list(gen_ids)
        self.seen_input_ids = None

    def parameters(self):
        yield self._param

    def generate(self, **batch):
        self.seen_input_ids = [int(t) for t in batch["input_ids"][0]]
        return torch.tensor([self.seen_input_ids + self.gen_ids],
                            dtype=torch.long)


def test_runner_generate_renders_through_the_contract():
    """If a future runner tokenizes raw prompts again, this fails."""
    runner = _load_runner()
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    task = "ask hello"
    model = _RecordingModel([3, 4, 5])
    text, nt, eos_hit = runner.generate(model, tok, task, 8)
    expected_ids = render_generation_input(tok, task, contract)
    assert model.seen_input_ids == expected_ids
    assert nt == 3
    assert eos_hit is False


def test_runner_generate_honors_explicit_contract():
    runner = _load_runner()
    tok = _tokenizer()
    contract = build_prompt_contract(tok, answer_only_output_contract())
    model = _RecordingModel([1, 2, 3])
    runner.generate(model, tok, "ask me", 8, contract)
    assert model.seen_input_ids == render_generation_input(
        tok, "ask me", contract)
    assert model.seen_input_ids != render_generation_input(
        tok, "ask me", build_prompt_contract(tok))
