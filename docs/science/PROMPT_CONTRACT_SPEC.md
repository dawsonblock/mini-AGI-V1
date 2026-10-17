# Prompt contract — specification (SCI-301)

Status: **implemented and verified**. Module:
`src-python/minagi/v161/prompt_contract.py`. Tests:
`test_campaign_prompt_parity.py`, `test_campaign_answer_boundaries.py`,
`test_campaign_eos_behavior.py`, `test_campaign_output_scoring.py`.

## The defect this closes

Campaign 3A trained LoRA adapters on chat-template-rendered
prompt/response pairs (`peft_trainer.supervised_tokens`) but evaluated
them on raw prompts (`run_campaign1.generate` tokenized the task text
directly). For a chat model those are different input surfaces: the
adapter was optimized for tokens it never saw at evaluation time. The
mismatch is demonstrated by
`test_campaign_prompt_parity.py::test_historical_mismatch_is_real` for
both a stub tokenizer and the pinned Qwen tokenizer.

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

## Where the identity is bound

- signed per-arm metrics: `prompt_contract_digest`,
  `output_contract_digest`, aggregate `generation_diagnostics`;
- per-row `DIAGNOSTICS-<arm>.jsonl` (enumerated by the seed's
  `COMMIT_MANIFEST`);
- the training receipt: `prompt_contract_digest`,
  `output_contract_digest`;
- the campaign plan's `generation_template_digest` uses the same
  computation as `template_digest`.

## Not in this change

The plan/protocol schemas are deliberately untouched: adding fields to
the versioned dataclasses would change historical digests and break
Campaign 3A verification. The 3B preregistration binds the frozen
output contract through the config and the signed metrics above.
