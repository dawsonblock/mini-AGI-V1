# Campaign 3B — Scoring / generation calibration (Step 4, exit criterion A2)

Status: **complete**. Output-format failures are measured on development
rows, the answer-only contract is selected and frozen, and the raw
surface's failure mode is quantified.

## Method

`scripts/calibrate_generation.py` compares a **predetermined** set of
greedy configurations against the frozen pinned model on development
data only:

- model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775` (dtype float32 on CPU;
  greedy decoding throughout — no sampling),
- rows: 12 of the 24 `validation` rows of `configs/campaign3_tasks.jsonl`
  (the script refuses every non-development split — the sealed final
  holdout and the hidden/retention/security partitions are never read),
- configurations (fixed in the script, never searched):
  the historical raw-prompt 32-token surface (measurement baseline),
  contract rendering at 32 / 64 / 128 tokens, and the answer-only
  contract at 64 tokens.

Reproduce:

```sh
python3 scripts/calibrate_generation.py \
    --config configs/campaign3a.yaml --split validation --limit 12 \
    --output GENERATION_CALIBRATION.json
```

## Result

| config | strict EM | first line | answer prefix | valid format | EOS | truncation | mean tokens |
|---|---|---|---|---|---|---|---|
| historical-raw-32 | 0.000 | 0.167 | 0.167 | 0.000 | 0.000 | 1.000 | 32.0 |
| contract-32 | 0.000 | 0.000 | 0.000 | 0.083 | 0.083 | 0.917 | 30.6 |
| contract-64 | 0.000 | 0.000 | 0.000 | 0.083 | 0.083 | 0.917 | 59.9 |
| contract-128 | 0.000 | 0.000 | 0.000 | 0.083 | 0.833 | 0.167 | 92.8 |
| **answer-only-64 (selected)** | **0.167** | 0.167 | 0.000 | **1.000** | **1.000** | **0.000** | **3.2** |

Reading:

- The historical raw surface truncates 100% of generations and never
  emits the end-of-turn token — the 3A evaluation never gave the model a
  chance to produce a scorable answer.
- Rendering the chat contract alone fixes the input surface but the
  model still rarely produces a single-line answer or stops (0.083);
  a larger budget mostly buys longer explanations (at 128 tokens, EOS
  rises to 0.833 but 92.8 mean tokens are spent).
- The **answer-only instruction** is the decisive lever: valid format
  and EOS termination reach 1.000 with no truncation and 3.2 mean
  generated tokens, and strict exact match rises from 0.000 to 0.167 on
  development rows. Strict EM remains the primary metric; the
  improvement here is an output-contract effect, not evidence of
  learning.

Selection rule (deterministic, recorded): among contract
configurations — highest mean strict exact match, then highest mean
valid-answer-format rate, then fewest mean generated tokens.

## Frozen output contract

The selected contract, to be bound into the next campaign's config:

```yaml
output_contract:
  instruction: "Return only the transformed result. Do not provide an explanation."
  max_new_tokens: 64
  stop_on_eos: true
```

The contract's digest is recorded in signed per-arm metrics
(`output_contract_digest`) and the training receipt, so any change to
the instruction, the limit or the stop policy is visible in evidence.
This is an engineering freeze for the 3B-D diagnostic; it does not
amend Campaign 3A's historical decision or thresholds.

## Machine-readable record

Produced by the run above (`dtype float32`, 2026-10-13):

```json
{
  "schema": "mini-agi-v16.6-generation-calibration-v1",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
  "dtype": "float32",
  "decoding": "greedy",
  "dataset": "configs/campaign3_tasks.jsonl",
  "split": "validation",
  "rows": 12,
  "selected": "answer-only-64",
  "selection_rule": "among contract configurations: highest mean strict exact match, then highest mean valid-answer-format rate, then fewest mean generated tokens",
  "configs": [
    {"name": "historical-raw-32", "raw_prompt": true, "strict_exact_match": 0.0, "correct_first_line": 0.167, "correct_answer_prefix": 0.167, "valid_answer_format": 0.0, "eos_termination": 0.0, "max_token_truncation": 1.0, "mean_generated_tokens": 32.0},
    {"name": "contract-32", "raw_prompt": false, "strict_exact_match": 0.0, "correct_first_line": 0.0, "correct_answer_prefix": 0.0, "valid_answer_format": 0.083, "eos_termination": 0.083, "max_token_truncation": 0.917, "mean_generated_tokens": 30.6},
    {"name": "contract-64", "raw_prompt": false, "strict_exact_match": 0.0, "correct_first_line": 0.0, "correct_answer_prefix": 0.0, "valid_answer_format": 0.083, "eos_termination": 0.083, "max_token_truncation": 0.917, "mean_generated_tokens": 59.9},
    {"name": "contract-128", "raw_prompt": false, "strict_exact_match": 0.0, "correct_first_line": 0.0, "correct_answer_prefix": 0.0, "valid_answer_format": 0.083, "eos_termination": 0.833, "max_token_truncation": 0.167, "mean_generated_tokens": 92.8},
    {"name": "answer-only-64", "raw_prompt": false, "strict_exact_match": 0.167, "correct_first_line": 0.167, "correct_answer_prefix": 0.0, "valid_answer_format": 1.0, "eos_termination": 1.0, "max_token_truncation": 0.0, "mean_generated_tokens": 3.2}
  ]
}
```

## Boundary

This calibration measures the **output contract** on development data.
It is not a learning result: no adapter is involved, and the strict
exact-match rate of the frozen model remains low. Proving that LoRA can
learn the intended behavior is the positive-control experiment
(Phase C) and is explicitly out of scope here.
