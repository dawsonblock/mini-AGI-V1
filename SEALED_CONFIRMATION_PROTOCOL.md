# Sealed confirmation protocol (Steps 20-22)

Status: **implemented and executed end to end on synthetic fixtures.
Real confirmation: NOT RUN** (no independently held sealed set or
evaluator custody was available to this effort). The workflow is
executable; it is not claimed to have confirmed anything.

## The defect this closes

Campaign 3A's qualifier checked that a final-holdout **manifest digest**
matched the one bound in the plan. It never loaded the holdout, never
generated a prediction and never scored anything. A digest match is
evidence that a file is the file it claims to be; it is not evidence
about a model. The master specification states the rule directly: *a
valid dataset digest alone must never be treated as completed
confirmation.*

## Custody model

The confirmation set is **evaluator custody**. The evaluator runs the
tool on its own host, under the evaluation and qualification keys that
never touched the training worker. The research worker receives only
aggregates: the signed record carries counts, families and the manifest
digest — never a sealed prompt or answer.

Three controls, each with tests:

1. **Research-custody denial.** `MINAGI_RESEARCH_WORKER` marks a
   research-side process; `load_sealed` refuses outright when the marker
   is set, *before touching the filesystem*, so the refusal is a custody
   decision rather than a missing-file error. The marker parses
   conservatively — only explicit falsy spellings mean "not a worker".
2. **Sealed-set location rule.** A sealed set living under the research
   storage is refused: research workers hold write access to that tree,
   so a sealed file inside it is readable through the same path the
   worker already uses.
3. **No leakage into evidence.** Tests assert the signed record and the
   append-only ledger contain no sealed prompt or expected answer, and
   that the sealed block carries only counts, families and the digest.

These are guards alongside the operational separation, not a
replacement for it.

## Workflow

```
freeze submission
  → verify eligibility
  → verify candidate artifacts
  → load sealed rows
  → execute candidates (real generation)
  → score under frozen rules
  → qualification (uncertainty, retention, safety)
  → sign
  → preserve evidence (append-only ledger)
```

### 1. Freeze

The submission binds what is being confirmed: campaign id, plan digest,
model id and revision, model digest, tokenizer digest, prompt-contract
digest, input budget and the frozen output contract. A missing field is
refused (`require_submission_fields`), and the loader enforces the same
check as the flow entry point.

The freeze step also refuses a research record that is absent, not
`QUALIFIED`, or carrying integrity failures. **A candidate that was not
scientifically qualified cannot reach the sealed lane at all.**

### 2. Verify candidate artifacts

Every per-seed adapter and arm-state directory the confirmation loads
must match the digest the signed research receipt bound. Adapter paths
are derived from the storage, never taken from the submission, so a
submission cannot point the evaluator at unverified bytes. One
substituted byte refuses before any generation.

### 3. Load sealed rows

The sealed manifest digest must equal the plan's bound
`final_holdout_digest`, and the sealed design must carry the required
blocks and family coverage. The loaded model's identity is re-verified
against the signed plan before any generation.

### 4. Execute

Real greedy generation under the frozen answer-only contract, using the
same prompt contract the research run used — the rebuild compares
digests, so a contract that cannot be reproduced refuses rather than
silently evaluating a different surface. Replay/retrieval arms are
reconstructed from their persisted state so the sealed run reproduces
the research-time context (L7 uses the same L5b context the research
comparison used).

### 5. Score under frozen rules

Strict exact match on hidden rows, the plan's retention scorer on
retention rows, and the **governed safety suite** on security rows —
scored through the shared `score_row` dispatch so a governed probe is
evaluated by its own allowlisted declarative predicate. The final check
is not weaker than the research gate it exists to re-verify.

### 6. Qualification

Applied to the sealed predictions only:

| gate | rule |
|---|---|
| primary comparison | `Score(L7) − Score(L5b)` (or `L6 − L5`), mean over paired sealed rows |
| effect size | must clear `minimum_neural_incremental_ft` |
| uncertainty | family-clustered bootstrap CI lower bound must clear `min_delta_ft_ci_lower` |
| family analysis | per-family deltas, leave-one-family-out, worst family |
| retention | drop vs baseline must not exceed `retention_max_drop` |
| security | pass rate must clear `security_min_pass_rate` |
| governed safety | eight mandatory categories, fail-closed (see `SAFETY_AND_RETENTION_REPORT.md`) |

An unmeasurable primary comparison — no paired sealed rows — is itself a
refusal reason, not a pass.

**Delayed retention on the sealed lane.** The sealed holdout does not
contain the seed-end delayed probes; those re-measure the reloaded
artifact at seed end and are a research-lane mechanism. Rather than
silently skipping a mandatory category, the gate records an explicit
`inherited` entry with its justification, and inheritance applies *only*
when the category is UNDEFINED. A category that was measured and failed
stays blocked — inheritance is for absent evidence, not for bad news.

### 7. Sign and preserve

A signed `QUALIFIED` or `REFUSED` decision, appended to an append-only
confirmation ledger with a monotonic sequence. The ledger is append-only
across attempts, so a failed candidate cannot be silently re-tuned
against the same sealed labels: every attempt is recorded.

## Verified behaviour

| case | test |
|---|---|
| sealed confirmation executes, scores and signs | `test_sealed_confirmation_executes_scores_and_signs` |
| non-qualified / missing research record refused | `test_refuses_non_qualified_research_record`, `test_refuses_missing_research_record` |
| wrong sealed set refused | `test_refuses_wrong_sealed_set` |
| insufficient coverage refused | `test_refuses_insufficient_coverage` |
| identity mismatch refused | `test_refuses_identity_mismatch` |
| gate failure is a *signed* refusal | `test_gate_failure_is_signed_refusal` |
| evaluator keys required | `test_confirmation_requires_evaluator_keys` |
| submission must bind model identity | `test_submission_requires_model_identity` |
| candidate artifacts verified against receipts | `test_candidate_artifacts_are_verified_against_receipts` |
| replay context reconstructed | `test_replay_arm_context_is_reconstructed` |
| no sealed content in record or ledger | `test_record_contains_no_sealed_answers_or_prompts`, `test_ledger_contains_no_sealed_content` |
| ledger append-only across attempts | `test_ledger_is_append_only_across_attempts` |
| research worker denied sealed access | `test_research_worker_cannot_load_sealed_rows`, `test_custody_guard_fires_before_the_filesystem` |
| sealed set under research storage refused | `test_sealed_set_inside_research_storage_is_refused` |
| governed suite enforced on the sealed lane | `test_sealed_confirmation_enforces_the_governed_suite`, `test_sealed_confirmation_refuses_a_governed_violation` |
| legacy sealed set labelled, not passed | `test_sealed_confirmation_labels_a_legacy_sealed_set` |

## Honest status

The workflow was executed end to end against **synthetic fixtures**
built by the tests. No independently held sealed confirmation set and
no independent evaluator custody were available to this effort, so:

- **Real confirmation: NOT RUN.**
- The sealed lane is **not** reported as passed, and no candidate is
  reported as confirmed.
- The synthetic runs demonstrate that the machinery executes, scores,
  gates, refuses and signs — nothing more.

This is the correct outcome under the master specification: *if sealed
data or independent evaluator custody is unavailable, implement the
workflow and run synthetic end-to-end tests, but report real
confirmation as NOT RUN, not PASS.*
