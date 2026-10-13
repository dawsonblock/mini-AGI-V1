"""v16.6 prompt contract — one tokenization surface for training and
inference (SCI-301).

Campaign 3A trained LoRA adapters on chat-template-rendered
prompt/response pairs (``peft_trainer.supervised_tokens``) but evaluated
them on raw prompts (``run_campaign1.generate`` tokenized the task text
directly). For a chat model the two token sequences are different
surfaces — the model was optimized for one input format and measured
under another — which is a source-confirmed contributor to Campaign 3A's
near-zero incremental transfer.

This module is the single rendering contract for every path that feeds
tokens to the model:

  * training examples (``encode_supervised``),
  * baseline, adapter, replay-practice and security inference
    (``encode_generation``),
  * diagnostic scoring of the generation contract
    (``generation_diagnostics``).

Training and inference render the SAME user content through the SAME
chat-template mechanism, and the training encoder proves the prompt
token sequence is a token-level prefix of the supervised sequence — the
boundary is verified, never assumed. The contract's digests (tokenizer
identity, chat template, output contract) are recorded in signed
per-arm metrics so a campaign's input surface is auditable.

The output contract (``OutputContract``) freezes the answer-only
generation surface — instruction, token limit, EOS stopping — before
confirmation data is evaluated. The historical default (no instruction,
greedy, EOS stop) preserves pre-3B behavior for replays; new campaigns
opt into the answer-only contract through their config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from egai.common.canonical import digest, sha256_bytes

LABEL_IGNORE = -100

PROMPT_CONTRACT_SCHEMA = "mini-agi-v16.6-prompt-contract-v1"
OUTPUT_CONTRACT_SCHEMA = "mini-agi-v16.6-answer-output-contract-v1"

#: The answer-only instruction the 3B diagnostic freezes (SCI-303).
#: It is deliberately explicit: strict exact-match scoring is only
#: attainable when the model emits the transformed result and nothing
#: else.
ANSWER_ONLY_INSTRUCTION = ("Return only the transformed result. "
                           "Do not provide an explanation.")


@dataclass(frozen=True)
class OutputContract:
    """The frozen output surface of one campaign.

    ``instruction`` is prepended to every task prompt (empty string =
    historical behavior). ``max_new_tokens`` and ``stop_on_eos`` are the
    generation limits the runner must use; the contract's digest is
    recorded in signed metrics so a change to any of them is visible.
    """
    instruction: str = ""
    max_new_tokens: int = 32
    stop_on_eos: bool = True
    schema: str = OUTPUT_CONTRACT_SCHEMA

    def __post_init__(self):
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        if int(self.max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not self.stop_on_eos:
            raise ValueError("stop_on_eos=false is not supported: a "
                             "generation that ignores the end-of-turn "
                             "token cannot be scored under the answer "
                             "contract")

    @property
    def digest(self) -> str:
        return digest(self)


#: Historical default — no instruction, greedy, EOS stop. Campaigns
#: that predate the contract replay under exactly this surface.
DEFAULT_OUTPUT_CONTRACT = OutputContract()


def answer_only_output_contract(max_new_tokens: int = 64) -> OutputContract:
    """The preregistered 3B diagnostic output contract."""
    return OutputContract(instruction=ANSWER_ONLY_INSTRUCTION,
                          max_new_tokens=int(max_new_tokens),
                          stop_on_eos=True)


def output_contract_from_config(cfg: Mapping[str, Any],
                                max_new_tokens: int) -> OutputContract:
    """Build the output contract from a campaign config mapping.

    Absent ``output_contract`` keeps the historical surface. An explicit
    ``max_new_tokens`` must agree with the effective protocol/config
    value — a contract that silently disagrees with the signed
    generation limit would be unverifiable.
    """
    oc = dict(cfg.get("output_contract") or {})
    limit = int(oc.get("max_new_tokens", max_new_tokens))
    if limit != int(max_new_tokens):
        raise ValueError(
            f"output_contract.max_new_tokens={limit} disagrees with the "
            f"effective max_new_tokens={int(max_new_tokens)} — freeze one "
            "value")
    return OutputContract(instruction=str(oc.get("instruction", "")),
                          max_new_tokens=limit,
                          stop_on_eos=bool(oc.get("stop_on_eos", True)))


def template_digest(tokenizer) -> str:
    """Digest of the tokenizer's chat template (empty when none).

    Identical to the campaign plan's ``generation_template_digest``
    computation, so the contract and the signed plan bind the same
    bytes.
    """
    return sha256_bytes(str(getattr(tokenizer, "chat_template", "") or "")
                        .encode())


def tokenizer_identity_digest(tokenizer) -> str:
    """Deterministic identity of the tokenizer surface the contract was
    built against — class, size, special tokens, template."""
    try:
        vocab_size = int(len(tokenizer))
    except TypeError:  # minimal stub tokenizers need not define __len__
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    doc = {"class": tokenizer.__class__.__name__,
           "vocab_size": vocab_size,
           "eos_token": str(getattr(tokenizer, "eos_token", "") or ""),
           "pad_token": str(getattr(tokenizer, "pad_token", "") or ""),
           "template_digest": template_digest(tokenizer)}
    return sha256_bytes(_canonical_json(doc))


@dataclass(frozen=True)
class PromptContract:
    """The immutable rendering contract bound to one campaign run.

    Field naming follows the 3B plan's PromptContract shape; ``schema``
    is this repository's spelling of the plan's ``version``.
    ``max_input_tokens`` is the supervised-sequence budget (0 =
    unspecified); ``max_new_tokens`` mirrors the output contract's
    generation limit.
    """
    schema: str
    tokenizer_digest: str
    template_digest: str
    add_generation_prompt: bool
    output: OutputContract
    max_input_tokens: int = 0
    max_new_tokens: int = 0

    @property
    def output_contract_digest(self) -> str:
        return self.output.digest

    @property
    def digest(self) -> str:
        return digest(self)


def build_prompt_contract(tokenizer, output: OutputContract | None = None,
                          add_generation_prompt: bool = True,
                          max_input_tokens: int = 0) -> PromptContract:
    """Build the contract for one tokenizer/output pair."""
    output = output or DEFAULT_OUTPUT_CONTRACT
    return PromptContract(
        schema=PROMPT_CONTRACT_SCHEMA,
        tokenizer_digest=tokenizer_identity_digest(tokenizer),
        template_digest=template_digest(tokenizer),
        add_generation_prompt=bool(add_generation_prompt),
        output=output,
        max_input_tokens=int(max_input_tokens),
        max_new_tokens=int(output.max_new_tokens))


def _canonical_json(doc: Mapping[str, Any]) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def compose_user_content(task_prompt: str, output: OutputContract) -> str:
    """The exact user message both training and inference render.

    The instruction (when present) is separated from the task by a blank
    line; the separator is part of the contract surface.
    """
    task = str(task_prompt)
    if output.instruction:
        return f"{output.instruction}\n\n{task}"
    return task


def _chat_template(tokenizer) -> str:
    return str(getattr(tokenizer, "chat_template", "") or "")


def render_messages(task_prompt: str,
                    contract: PromptContract) -> list[dict]:
    """The message structure every path renders (plan: render_messages).

    The answer instruction is composed into the user content — the
    frozen calibration surface (docs/science/SCORING_CALIBRATION.md)
    binds this rendering; a system-message variant would be a new,
    separately calibrated contract.
    """
    return [{"role": "user",
             "content": compose_user_content(task_prompt, contract.output)}]


def render_prompt_text(tokenizer, task_prompt: str,
                       contract: PromptContract) -> str:
    """The model-facing prompt text for one task.

    With a chat template this is the rendered user turn (plus the
    generation prompt when the contract asks for it) — the exact surface
    ``encode_generation`` tokenizes and ``encode_supervised`` uses as
    the supervised boundary.
    """
    messages = render_messages(task_prompt, contract)
    if _chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=contract.add_generation_prompt,
            tokenize=False)
    return messages[0]["content"]


def encode_generation(tokenizer, task_prompt: str,
                      contract: PromptContract) -> list[int]:
    """The exact input token sequence generation consumes (plan:
    encode_generation).

    This is the single inference-side entry point: baseline, adapter,
    replay-practice and security paths all call it, so no runner can
    evaluate a raw prompt while training renders a chat surface.
    """
    text = render_prompt_text(tokenizer, task_prompt, contract)
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("prompt rendered to zero tokens")
    return ids


def encode_supervised(tokenizer, task_prompt: str, response: str,
                      contract: PromptContract, max_length: int) -> dict:
    """Encode one supervised {task, response} pair under the contract
    (plan: encode_supervised).

    Chat-template tokenizers are encoded STRICTLY: the full text is
    tokenized once and the prompt tokenization must be a token-level
    prefix of it — a boundary that merges tokens is refused, never
    silently trained around. Prompt tokens are masked to -100 and only
    the response (including the template's end-of-turn token) carries
    loss. Template-less tokenizers keep the historical separate-
    tokenization semantics (response + EOS), which predates the chat
    surface and is pinned by the v16.2.1 tests.

    Malformed examples (empty prompt/response, or a response fully
    truncated by ``max_length``) raise — callers decide whether that
    invalidates the run.
    """
    if not str(task_prompt) or not str(response).strip():
        raise ValueError("empty prompt or response")
    if int(max_length) <= 0:
        raise ValueError("max_length must be positive")
    response = str(response)
    messages = render_messages(task_prompt, contract)
    if _chat_template(tokenizer):
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=contract.add_generation_prompt,
            tokenize=False)
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": response}],
            tokenize=False)
        if not full_text.startswith(prompt_text):
            raise ValueError("chat template render is not prefix-consistent")
        p_ids = list(tokenizer(prompt_text,
                               add_special_tokens=False)["input_ids"])
        full_ids = list(tokenizer(full_text,
                                  add_special_tokens=False)["input_ids"])
        if full_ids[:len(p_ids)] != p_ids:
            raise ValueError(
                "tokenization is not prefix-stable at the prompt/response "
                "boundary — the supervised sequence would not start where "
                "generation's prompt ends")
        input_ids = full_ids
        supervised_from = len(p_ids)
    else:
        prompt_text = messages[0]["content"]
        eos = str(getattr(tokenizer, "eos_token", "") or "")
        p_ids = list(tokenizer(prompt_text,
                               add_special_tokens=False)["input_ids"])
        r_ids = list(tokenizer(response + eos,
                               add_special_tokens=False)["input_ids"])
        input_ids = p_ids + r_ids
        supervised_from = len(p_ids)
    if supervised_from >= len(input_ids):
        raise ValueError("response produced no supervised tokens")
    labels = [LABEL_IGNORE] * supervised_from + \
        input_ids[supervised_from:]
    truncated = False
    if len(input_ids) > int(max_length):
        # keep the response tail: drop leading prompt context, never the
        # supervised suffix
        overflow = len(input_ids) - int(max_length)
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        truncated = True
    n_sup = sum(1 for t in labels if t != LABEL_IGNORE)
    if n_sup == 0:
        raise ValueError("response fully truncated by max_length — "
                         "no supervised tokens remain")
    supervised_ids = [t for t in labels if t != LABEL_IGNORE]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_supervised = eos_id is not None and int(eos_id) in supervised_ids
    eos_str = str(getattr(tokenizer, "eos_token", "") or "")
    if eos_str and eos_str in input_ids and not eos_supervised:
        raise ValueError(
            "the rendered end-of-turn token is not supervised — the "
            "model would not be trained to stop")
    return {"input_ids": input_ids, "labels": labels,
            "n_prompt": len(labels) - n_sup, "n_supervised": n_sup,
            "prompt_tokens": len(p_ids),
            "response_tokens": len(input_ids) - supervised_from,
            "truncated": truncated,
            "eos_supervised": eos_supervised}


def _normalize(text: str) -> str:
    return str(text).strip().casefold()


def generation_diagnostics(expected: str, prediction: str, *,
                           generated_tokens: int, max_new_tokens: int,
                           eos_hit: bool | None = None) -> dict:
    """Per-prediction diagnostics for the answer contract (SCI-303).

    Diagnostics are recorded alongside — never instead of — the strict
    exact-match score: a correct prefix followed by a contradictory
    answer must not receive primary correctness credit. ``eos_hit``
    should be the precise observation from generation (last token was
    the EOS id); when absent it is inferred as "stopped before the
    token limit".
    """
    exp = _normalize(expected)
    pred = _normalize(prediction)
    lines = [ln for ln in str(prediction).strip().splitlines() if ln.strip()]
    if eos_hit is None:
        eos_hit = int(generated_tokens) < int(max_new_tokens)
    if eos_hit:
        stop_reason = "end_of_message"
    elif int(generated_tokens) >= int(max_new_tokens):
        stop_reason = "max_tokens"
    else:
        stop_reason = "unknown"
    return {
        "strict_exact_match": float(pred == exp),
        "correct_first_line": float(bool(lines) and _normalize(lines[0]) == exp),
        "correct_answer_prefix": float(bool(exp) and pred.startswith(exp)
                                       and pred != exp),
        "valid_answer_format": float(bool(pred) and len(lines) == 1),
        "eos_termination": float(bool(eos_hit)),
        "max_token_truncation": float(int(generated_tokens)
                                      >= int(max_new_tokens)),
        "generated_tokens": int(generated_tokens),
        "stop_reason": stop_reason,
    }
