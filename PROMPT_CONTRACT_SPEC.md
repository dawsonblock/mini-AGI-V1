# Prompt contract — specification (SCI-301)

Status: **implemented and verified**. Module:
`src-python/minagi/v161/prompt_contract.py`. Tests:
`test_campaign_prompt_parity.py`, `test_campaign_answer_boundaries.py`,
`test_campaign_eos_behavior.py`, `test_campaign_output_scoring.py`,
`test_positive_control_learning.py`.

This is the authoritative specification for the Campaign 3B source
release. `docs/science/PROMPT_CONTRACT_SPEC.md` is the implementation-
time snapshot of the same contract and is retained unmodified.

## The defect this closes

Campaign 3A trained LoRA adapters on chat-template-rendered
prompt/response pairs (`peft_trainer.supervised_tokens`) but evaluated
them on raw prompts (`run_campaign1.generate` tokenized the task text
directly). For a chat model those are different input surfaces: the
adapter was optimized for tokens it never saw at evaluation time. The
mismatch is demonstrated by
`test_campaign_prompt_parity.py::test_historical_mismatch_is_real` for
both a stub tokenizer and the pinned Qwen tokenizer.

## Required contract elements

The master specification requires the contract to fix eleven things.
Each is bound to a named field or operation, so a reviewer can check
the mapping rather than take it on trust.

| required element | where it lives |
|---|---|
| model identity | campaign plan (`model_id`, `model_revision`) + `model_digest` over the resolved snapshot |
| tokenizer identity | `PromptContract.tokenizer_digest` — tokenizer class, vocab size, special tokens and template digest |
| tokenizer / chat-template digest | `PromptContract.template_digest` = sha256 of the chat template; identical computation to the plan's `generation_template_digest` |
| system and user message construction | `render_messages` — one user turn; the answer instruction is composed into the user content by `compose_user_content` (instruction, blank line, task). A system-message variant would be a **new**, separately calibrated contract |
| assistant-generation prefix | `PromptContract.add_generation_prompt` — rendered by `render_prompt_text` |
| response-only training targets | `encode_supervised` masks every prompt token to `-100` |
| EOS and end-of-message handling | the template's end-of-turn token is supervised; a rendered end marker that is not supervised is refused |
| input and output token limits | `PromptContract.max_input_tokens` (supervised budget) and `OutputContract.max_new_tokens` (generation limit) |
| padding and truncation behaviour | over-budget sequences keep the response tail; a fully truncated response is refused; padding is excluded from loss |
| expected answer format | `OutputContract.instruction` — the frozen answer-only surface |
| configuration digest | `PromptContract.digest` / `OutputContract.digest`, recorded in signed metrics and the training receipt |

## The contract

`PromptContract` (frozen, digestible):

| field | meaning |
|---|---|
| `schema` | `mini-agi-v16.6-prompt-contract-v1` |
| `tokenizer_digest` | class + vocab size + special tokens + template digest |
| `template_digest` | sha256 of the tokenizer's chat template (identical to the plan's `generation_template_digest`) |
| `add_generation_prompt` | render the assistant generation prefix |
| `output` | the `OutputContract` in force |
| `max_input_tokens` | supervised-sequence budget (0 = unspecified) |
| `max_new_tokens` | mirrors the output contract's generation limit |

`OutputContract` (frozen, digestible): `instruction`, `max_new_tokens`,
`stop_on_eos` (false is refused — a generation that ignores the
end-of-turn token cannot be scored under the answer contract). Two
surfaces exist:

- **historical default** — no instruction, greedy, EOS stop (campaigns
  that predate the contract replay under exactly this surface);
- **answer-only** — `answer_only_output_contract()`: "Return only the
  transformed result. Do not provide an explanation." — the surface
  frozen by `docs/science/SCORING_CALIBRATION.md`.

Campaign configs opt in through `output_contract:` (instruction /
max_new_tokens / stop_on_eos); an explicit `max_new_tokens` that
disagrees with the effective protocol value is refused.

## Operations

| operation | responsibility |
|---|---|
| `render_messages` | the message structure both paths render (the instruction is composed into the user content) |
| `render_prompt_text` | the model-facing prompt text (chat template + generation prefix) |
| `encode_generation` | the exact input token sequence generation consumes — the single inference-side entry point |
| `encode_supervised` | full training ids + response-only loss mask |
| `generation_diagnostics` | strict EM, first line, answer prefix, format validity, EOS/truncation and `stop_reason` |

## Guarantees (verified, not assumed)

1. **Token-level parity** — the training prompt prefix equals the
   inference input tokens. Verified for a stub tokenizer and the pinned
   Qwen tokenizer (revision `7ae55760…`); a future runner that
   tokenizes raw prompts fails
   `test_runner_generate_renders_through_the_contract`.
2. **Strict boundaries** — the chat-template path tokenizes the full
   render once and requires the prompt tokenization to be a token-level
   prefix of it; a boundary that merges tokens is refused
   (`test_boundary_merge_is_refused`).
3. **Exact masking** — prompt tokens are `-100`; only the response (and
   the template's end-of-turn token) carries loss; padding is excluded.
4. **End-of-turn supervision** — the template's end-of-message token is
   supervised, and a rendered end marker that is not supervised is
   refused. (The tiny-model learnability test shows the consequence:
   without end-of-turn supervision the adapter learns the mapping but
   never stops.)
5. **Truncation policy** — over-budget sequences keep the response
   tail; a response that would be fully truncated is refused, never
   silently trained as a prompt-less example.
6. **Diagnostics are diagnostics** — first-line and prefix credit are
   recorded alongside strict exact match, never instead of it.

## Boundary effects are handled, not assumed

Independently tokenized strings cannot always be concatenated to
reproduce the tokenization of the complete conversation: a chat
template can merge tokens across the prompt/response boundary (for
example a trailing space and a leading word character). The contract
therefore does **not** concatenate `tokenize(prompt)` with
`tokenize(response)` on the chat-template path. It tokenizes the full
render once and requires the prompt tokenization to be a token-level
prefix of it, refusing the example when the boundary merges. The
prefix assertion `training_input_ids[:prompt_length] ==
inference_prompt_input_ids` is the same check expressed against the
inference surface.

## Where the identity is bound

- signed per-arm metrics: `prompt_contract_digest`,
  `output_contract_digest`, aggregate `generation_diagnostics`;
- per-row `DIAGNOSTICS-<arm>.jsonl` (enumerated by the seed's
  `COMMIT_MANIFEST`);
- the training receipt: `prompt_contract_digest`,
  `output_contract_digest`;
- the campaign plan's `generation_template_digest` uses the same
  computation as `template_digest`.

## Fail-closed behaviour

A prompt contract that differs between training and evaluation cannot
be silently used. `encode_supervised` refuses a boundary merge, refuses
a fully truncated response, and refuses an unsupervised end marker;
`encode_generation` refuses a zero-token render; `OutputContract`
refuses `stop_on_eos=False`; `output_contract_from_config` refuses a
contract whose `max_new_tokens` disagrees with the signed generation
limit.

## Test coverage

| area | test |
|---|---|
| parity, both tokenizers | `test_campaign_prompt_parity.py` |
| Unicode, empty outputs, punctuation, truncation, multi-turn, special tokens | `test_campaign_answer_boundaries.py` |
| EOS and generation limits | `test_campaign_eos_behavior.py` |
| strict vs diagnostic scoring | `test_campaign_output_scoring.py` |
| real training parity on the pinned model | `test_positive_control_learning.py` |

## Not in this change

The plan/protocol schemas are deliberately untouched: adding fields to
the versioned dataclasses would change historical digests and break
Campaign 3A verification. The 3B preregistration binds the frozen
output contract through the config and the signed metrics above.
