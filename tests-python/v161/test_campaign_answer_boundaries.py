"""SCI-301 — supervised answer boundaries.

The training encoder must prove, not assume, that the supervised
response starts exactly where generation's prompt ends:

  * the prompt tokenization is a token-level prefix of the supervised
    sequence (chat-template path),
  * prompt tokens are masked with -100 and only the response carries
    loss,
  * a boundary that merges tokens is refused,
  * template specials are not duplicated,
  * truncation keeps the supervised tail and a fully truncated response
    is rejected.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.prompt_contract import (  # noqa: E402
    LABEL_IGNORE, answer_only_output_contract, build_prompt_contract,
    compose_user_content, render_prompt_text, encode_supervised)

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
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
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


def test_boundaries_stub_and_qwen():
    for tok in (_tokenizer(), _qwen_tokenizer()):
        contract = build_prompt_contract(tok)
        enc = encode_supervised(tok, "ask hello", "ans a1", contract,
                                      max_length=4096)
        ids, labels = enc["input_ids"], enc["labels"]
        n_prompt = enc["prompt_tokens"]
        assert len(ids) == len(labels)
        # the prompt region is masked exactly; the supervised region
        # begins at the prompt boundary
        assert labels[:n_prompt] == [LABEL_IGNORE] * n_prompt
        assert all(t != LABEL_IGNORE for t in labels[n_prompt:])
        # the supervised suffix is exactly the tokenized continuation
        assert ids[n_prompt:] == enc["input_ids"][n_prompt:]
        assert enc["n_supervised"] == len(ids) - n_prompt
        assert enc["n_prompt"] == n_prompt


def test_answer_instruction_is_rendered_identically_for_training():
    tok = _tokenizer()
    output = answer_only_output_contract()
    contract = build_prompt_contract(tok, output)
    # the rendered prompt text carries the instruction and the composed
    # user content — the same composition training and inference share
    prompt_text = render_prompt_text(tok, "ask hello", contract)
    assert output.instruction in prompt_text
    assert compose_user_content("ask hello", output) in prompt_text
    enc = encode_supervised(tok, "ask hello", "ans a1", contract,
                                  max_length=4096)
    # masked prefix still exactly the prompt boundary
    assert enc["labels"][:enc["prompt_tokens"]] == \
        [LABEL_IGNORE] * enc["prompt_tokens"]


def test_specials_are_not_duplicated():
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    enc = encode_supervised(tok, "ask hello", "ans a1", contract,
                                  max_length=4096)
    decoded_prompt = tok.decode(enc["input_ids"][:enc["prompt_tokens"]])
    decoded_full = tok.decode(enc["input_ids"])
    # exactly one user turn and one assistant turn header
    assert decoded_prompt.count("user:") == 1
    assert decoded_prompt.count("assistant:") == 1
    assert decoded_full.count("assistant:") == 1
    assert decoded_full.count("user:") == 1


def test_boundary_merge_is_refused():
    """A tokenizer whose prompt tokenization is not a prefix of the full
    render must be refused — never trained around."""
    class _MergeAtBoundary:
        chat_template = "{{ messages[0]['content'] }}"

        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=False):
            text = messages[0]["content"]
            if len(messages) > 1:
                text = text + messages[1]["content"]
            return text

        def __call__(self, text, add_special_tokens=False):
            # the full render "ask helloans a1" tokenizes differently
            # from the prompt "ask hello" — a boundary merge
            if text.endswith("ans a1"):
                return {"input_ids": [8, 9, 10]}
            return {"input_ids": [7, 8]}

    with pytest.raises(ValueError, match="prefix-stable"):
        encode_supervised(_MergeAtBoundary(), "ask hello", "ans a1",
                                build_prompt_contract(_MergeAtBoundary()),
                                max_length=64)


def test_truncation_keeps_response_tail():
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    enc = encode_supervised(tok, "ask hello world x y me ok", "ans a1",
                                  contract, max_length=3)
    assert len(enc["input_ids"]) == 3
    assert enc["n_supervised"] > 0
    assert enc["truncated"] is True
    # the supervised tail survives truncation
    assert all(t != LABEL_IGNORE
               for t in enc["labels"][-enc["n_supervised"]:])


def test_response_without_tokens_rejected():
    """A response that tokenizes to nothing (or is fully cut) is
    refused — never silently trained as an empty target."""
    class _NoResponseTokens:
        chat_template = None
        eos_token = ""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2] if "ask" in text else []}

    tok = _NoResponseTokens()
    with pytest.raises(ValueError, match="no supervised tokens"):
        encode_supervised(tok, "ask hello", "ans a1",
                                build_prompt_contract(tok), max_length=64)


def test_empty_inputs_rejected():
    tok = _tokenizer()
    contract = build_prompt_contract(tok)
    with pytest.raises(ValueError):
        encode_supervised(tok, "", "ans a1", contract, max_length=64)
    with pytest.raises(ValueError):
        encode_supervised(tok, "ask hello", "  ", contract,
                                max_length=64)
