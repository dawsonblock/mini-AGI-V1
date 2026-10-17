# v4.0.3-r2 — Execution Provenance

## The Diagnostics-9 finding

v4.0.3-r1 closed the *measurement* boundary: an authorized evaluator
recomputes committed predictions and signs
`independent-evaluation-v1`. The remaining gap — the evaluator
authenticated evidence the **research process generated**. A run could
claim adapter X produced outputs Y; the evaluator checked Y scored as
reported, but never established that X on model M actually produced Y.

## The two-claim separation

Execution provenance and training provenance are deliberately distinct
verification lanes — conflating them was the original defect:

| Claim | Question | Evidence |
|---|---|---|
| Execution provenance | Does THIS model+adapter produce THESE outputs under THESE conditions? | `model-execution-receipt-v1` — evaluator-owned load + inference + score |
| Training provenance | Was this adapter trained as claimed (algorithm, config, lineage)? | `TRAINING_RECEIPT.json`, source revision, config, artifact lineage — verified separately |

A valid execution receipt proves the artifact produced the outputs; it
says nothing about how the artifact was made. One flag must never
assert both.

## What the executor does

`independent_model_executor.execute_evaluation`:

1. Receives frozen artifact references (model id+revision, adapter dir,
   evaluator-controlled task set).
2. Loads `HFLoadSpec(model_id, revision)` through
   `hf_runtime.load_causal_lm(..., purpose="research")` — the
   evaluator's recorded opt-in — and `load_tokenizer`; builds the
   canonical prompt contract (`build_prompt_contract` +
   `output_contract_from_config`).
3. Refuses incompatible artifacts: `adapter_artifact_digest_of` must
   resolve, and `adapter_config.base_model_name_or_path` must equal
   the declared model.
4. Executes every task via `encode_generation` (the same
   contract-bound encoding the run pipeline uses — never raw prompts),
   deterministic `generate` (`do_sample=False`, `pad=eos`,
   `max_new_tokens` from the output contract).
5. Scores each output with the declared oracle (`exact-match-v2` →
   `exact_match_strict`).
6. Issues `model-execution-receipt-v1` binding: execution_id,
   model_id, model_revision, tokenizer_digest, adapter_digest,
   dataset_digest (id+prompt_sha256+expected_sha256 commitment —
   answers stay private), generation_config_digest, scoring_version,
   execution_environment_digest (python/torch/transformers/peft/
   platform/device), outputs_digest, scores_digest, execution_mode,
   evaluator_authority_id — Ed25519 over the canonical body.

## Modes and the trust boundary

| execution_mode | Meaning | Real-model confirmation? |
|---|---|---|
| `real_model` | evaluator loaded base + adapter, ran HF inference | yes |
| `frozen_baseline` | evaluator loaded the bare model (adapter_digest = null, enforced) | yes |
| `simulated` | injected mock loader — a TEST seam | **never** (`require_real` default refuses) |

A test artifact of arbitrary bytes (`b"weights"`-style fixtures) can
receive a *simulated* receipt — and is refused for real-model
confirmation. The real lane requires actual `load_causal_lm` success.

## Receipt verification

`verify_execution_receipt` checks, in order: schema + binding fields →
status → mode legality (simulated refused when `require_real`;
baseline must carry null adapter; real must carry adapter bytes) →
attested model identity vs `expected_model` → `evaluation`-role
authorization → Ed25519 signature → recomputed binding over supplied
artifacts (adapter bytes, task set, generation config, predictions
file, scores, environment). `allow_detached` permits signature+role
only — never trusted import.

## What is NOT proven (stated plainly)

- **Key custody**: the signature proves an `evaluation`-role key
  signed — separation of that key from the research agent is
  operational (OS identity, sealed storage), attested by
  `evaluator_custody` checks, not by crypto.
- **Environment equivalence**: `execution_environment_digest` records
  provenance; it does not prove two environments bit-match.
- **Training lineage**: deliberately out of scope — see the
  two-claim separation above.
