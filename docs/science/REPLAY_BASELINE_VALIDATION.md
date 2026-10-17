# Replay baseline validation (SCI-302)

Status: **implemented and verified**. Module: `src-python/minagi/v161/arms.py`.
Tests: `test_replay_nonempty.py`, `test_replay_provenance.py`.

## The defect this closes

Campaign 3A's L5 grounded-replay baseline admitted a practice trace only
when the model's own prediction scored exactly 1.0 — and no practice
prediction ever did. Every packaged `REPLAY.json` in the pushed
evidence (`results/campaign3a-v166/arms/campaign3a-v166/seed-*/L5/REPLAY.json`)
contains `"traces": []`, so the campaign's primary comparison
(L6 − L5) never had a functioning grounded-replay comparator.

## What is preserved

`GroundedReplayArm` (L5a) is unchanged: a self-verified replay of
practice traces, hidden labels never consulted. Its zero-trace outcome
in 3A is a recorded negative result, not something to fill with
unverifiable model predictions.

## What is added — L5b, the verified replay baseline

`OracleVerifiedReplayArm` (`arm_id = "L5b"`) admits every **TRAIN** row
whose membership and content verify; the answer is the row's
established training label, bound by digest:

```python
VerifiedReplayRecord(task_id, family_id, prompt_digest, answer_digest,
                     split_digest, verifier_digest, verification_status)
```

Rules enforced by `build_state` / `verify_state`:

- non-train rows (hidden / validation / holdout) are **refused** and
  never persisted — their labels cannot be retrieved into any prompt;
- duplicate task ids, duplicate prompts, conflicting answers for one
  prompt, excluded ids and empty rows are refused **with reasons**;
- an optional trusted verifier may reject rows; the verification policy
  is recorded as `verifier_digest` (the default policy is the
  established-train-labels digest);
- retrieval is budgeted (`max_block_chars`) and deterministic;
- every `augment()` call is logged: hit task ids and scores,
  injected/trimmed characters **and tokens** (when a counter is
  installed), the augmented-prompt digest and the budget;
- `verify_state()` re-derives every persisted digest and raises on
  tampering.

## Evidence

Each replay-using arm publishes `REPLAY_MANIFEST-<arm>.json` inside the
seed's atomic evidence commit: records, rejected rows, split/verifier
digests, retrieval statistics. The signed per-arm metrics carry the same
summary under `replay`.

## The matched comparison — L7

L7 trains the same adapter as L6 and is **evaluated under the same
verified-replay context L5b uses**, so the primary 3B comparison

    Δ = Score(L7) − Score(L5b)

varies only the neural adaptation (both sides have the replay context).
`RESULT.json` reports `delta_ft_neural_L7_minus_L5b` alongside the
historical `delta_ft_neural_L6_minus_L5`.

## Verified behavior (Step 9 table)

| case | test |
|---|---|
| verified examples supplied → buffer nonempty | `test_verified_examples_make_the_buffer_nonempty` |
| relevant prompt → eligible examples retrieved | `test_relevant_prompt_retrieves_eligible_examples` |
| no eligible example → explicit no-hit | `test_no_eligible_example_records_an_explicit_no_hit` |
| over-budget example → deterministic trim | `test_context_budget_trims_deterministically` |
| tampered state → digest verification fails | `test_tampered_state_fails_digest_verification` |
| token costs recorded | `test_token_costs_are_recorded_when_a_counter_is_set` |
| hidden labels refused, never persisted | `test_non_train_rows_are_refused_and_never_persisted` |
| duplicates/conflicts refused with reasons | `test_duplicates_and_conflicts_are_refused` |
| custom verifier enforced and recorded | `test_custom_verifier_is_enforced_and_recorded` |

## Boundary

A functioning replay baseline is not automatically an *effective* one.
L5b's retrieval is lexical and budgeted; whether it helps is measured by
the campaign, and the 3B gates compare against it explicitly rather than
assuming a fixed baseline.
