# Replay baseline validation (SCI-302)

Status: **implemented, verified, and observed operating**. Module:
`src-python/minagi/v161/arms.py`. Tests: `test_replay_nonempty.py`,
`test_replay_provenance.py`, `test_positive_control_learning.py`.

This is the authoritative validation record for the Campaign 3B source
release. `docs/science/REPLAY_BASELINE_VALIDATION.md` is the
implementation-time snapshot of the same mechanism and is retained
unmodified.

## The defect this closes

Campaign 3A's L5 grounded-replay baseline admitted a practice trace only
when the model's own prediction scored exactly 1.0 — and no practice
prediction ever did. Every packaged `REPLAY.json` in the pushed evidence
(`results/campaign3a-v166/arms/campaign3a-v166/seed-*/L5/REPLAY.json`)
contains `"traces": []`. The campaign's primary comparison (L6 − L5)
therefore never had a functioning grounded-replay comparator: the
"grounded replay" arm was the bare model wearing a replay arm's name.

## What is preserved

`GroundedReplayArm` (L5a) is **unchanged**: a self-verified replay of
practice traces, hidden labels never consulted. Its zero-trace outcome
in 3A is a recorded negative result, not something to fill with
unverifiable model predictions.

When L5a admits nothing it is reported as **INACTIVE**, not as an
operational grounded-replay baseline:

```
arm_state: L5a  traces: 0  status: INACTIVE (self-verified replay admitted no traces)
```

The campaign runner records the trace count in the arm's signed metrics
and the `REPLAY_MANIFEST-L5a.json`, so an INACTIVE arm is visible in
evidence rather than inferred from a suspiciously flat score.

## What is added — L5b, the oracle-verified training replay

`OracleVerifiedReplayArm` (`arm_id = "L5b"`) admits every **TRAIN** row
whose membership and content verify; the answer is the row's
established training label, bound by digest:

```python
VerifiedReplayRecord(task_id, family_id, prompt_digest, answer_digest,
                     split_digest, verifier_digest, verification_status)
```

Every admitted record therefore carries the fields the master
specification requires: task and family identity, training-split
identity, prompt and answer digests, verifier identity, verification
result, evidence provenance and retrieval eligibility.

Rules enforced by `build_state` / `verify_state`:

- non-train rows (hidden / validation / holdout / sealed) are
  **refused** and never persisted — their labels cannot be retrieved
  into any prompt;
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

## Required metrics

The master specification requires explicit metrics for replay count,
family coverage, retrieval hits, misses, context cost and actual prompt
augmentation. `retrieval_stats()` reports all of them:

| required metric | field |
|---|---|
| replay count | `records` (+ `rejected` with reasons) |
| family coverage | `retrieved_task_ids` joined to family via the corpus; per-family counts in the manifest |
| retrieval hits | `hit_queries` |
| retrieval misses | `no_hit_queries` |
| context cost | `injected_chars`, `trimmed_chars`, `injected_tokens`, `prompt_tokens`, `augmented_tokens` |
| actual prompt augmentation | the per-query log's `augmented_prompt_digest` plus the injected counts — a retrieval that does not change the prompt is visible as zero injected characters |

## Verified behaviour (Step 9 table)

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

## Observed operation (executed diagnostic)

The Step 13 positive control ran the arm on the pinned model:

| quantity | 3A | 3B (observed) |
|---|---|---|
| admitted records | 0 | **30 / 30** |
| retrieval queries | 0 | 20 |
| retrieval hits | 0 | **20 / 20** |
| no-hit queries | 0 | 0 |
| injected tokens | 0 | **760** |

L5b was the strongest hidden-block arm in that run (0.583 strict exact
match vs 0.417 frozen), which is exactly the kind of result a working
non-neural comparator is supposed to be able to produce — and the
reason the neural comparison must be made against it rather than
against the frozen model.

## Leakage prevention

- Only `train`-split rows are admissible; the split is verified against
  the signed partition digest.
- Hidden, validation, holdout and sealed-confirmation labels are never
  loaded into the store, never persisted, and cannot be retrieved.
- `test_replay_provenance.py` asserts the refusal path and that no
  out-of-split record reaches disk.
- The retrieval index is built from admitted records only; a query
  cannot reach a row that was not admitted.

## The matched comparison — L7

L7 trains the same adapter as L6 and is **evaluated under the same
verified-replay context L5b uses**, so the primary 3B comparison

    Δ = Score(L7) − Score(L5b)

varies only the neural adaptation (both sides have the replay context).
`RESULT.json` reports `delta_ft_neural_L7_minus_L5b` alongside the
historical `delta_ft_neural_L6_minus_L5`. The extra inference tokens and
compute the replay context costs are recorded per arm in
`cost_per_correct` and the efficiency block.

## Boundary

A functioning replay baseline is not automatically an *effective* one.
L5b's retrieval is lexical and budgeted; whether it helps is measured by
the campaign, and the 3B gates compare against it explicitly rather than
assuming a fixed baseline. In the executed diagnostic it helped
(0.417 → 0.583 hidden); that is one seed on a small corpus, and it is
reported as such.
