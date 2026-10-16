# V4.1.0 — benchmark & task-content audit

Pre-execution audit of the campaign4a corpus and the evaluation path.
Verdict: **task definitions, expected answers, and split design are
correct**; every check below was executed programmatically against the
committed files, not assumed.

## Function ↔ expected agreement (all rows, all surfaces)

Every corpus row whose family carries a generator function was
re-derived and compared byte-for-byte with the committed `expected`:

| surface | rows checked | mismatches |
|---|---|---|
| campaign4a corpus (train/validation/within_holdout/hidden/compositional/rehearsal) | 1,085 | 0 |
| sealed holdout `campaign4a_holdout.jsonl` | 100 | 0 |
| historical `campaign3e_tasks.jsonl` (reproduction truth) | 373 | 0 |

Instruction text was reviewed against each function body for all 8
trained families, both new hidden families (hid4-*), all 4 new
compositional families (order of composition matches the "X, then Y"
wording), all 5 rehearsal families, and all 5 sealed-holdout families
(including odd-length `ho4-swap-halves` middle-character semantics).

All 8 `TASK_B` variant phrasings describe the identical operation —
`within_holdout` therefore measures instruction-surface robustness, not
a different task (64/256 rows carry `variant: "B"`).

## Arithmetic / factual answers

Every parseable arithmetic retention and rehearsal probe was verified
programmatically (add/subtract/multiply/divide/square/cube/sqrt/
power/percent forms): **0 errors**. The factual QA answers (capitals,
chemical symbols, calendar/geometry constants) were hand-checked —
conventional answers used throughout (Nile, Sahara, Mercury).

## Input-word disjointness (empirical, not just declared)

| pair | overlap |
|---|---|
| train ∩ within_holdout | 0 |
| validation ∩ within_holdout | 0 |
| within_holdout ∩ {hidden, compositional, rehearsal, holdout} | 0 |
| train ∩ {hidden, rehearsal, holdout} | 0 |

`DatasetPartitionSetV4` additionally enforces sample-id disjointness
across every partition, family-disjointness of hidden/compositional/
final_holdout vs within_holdout, within_holdout ⊆ train families, and
rehearsal-family disjointness from every measured surface.

## Scoring-path checks

- `security` rows carry governed `verify` specs; polarity verified
  against `evaluators.declarative_score` — `not(contains_none)` flags a
  violation iff a forbidden marker appears; `not(regex_fullmatch)` flags
  iff the output drifts off the answer-only shape.
- `SAFETY_EXT` expected fields follow the frozen v1-suite convention
  (uppercased transform — cf. `saf-inj-1` "AMMAG"). `expected` is
  documentary for governed probes; the verify spec is what scores.
- The fit runner's default rehearsal pool whitelist is
  `(train, validation, rehearsal)` — within_holdout/retention/hidden/
  security can never contaminate training (asserted in
  test_fit_runner_integrity and test_dataset_independence).
- `CONFIG.json` records `within_holdout_rows`/`retention_rows`, so the
  reconciler's expected-file set sizes the new surfaces exactly.
- `rehearsal_pool.{verified_pairs,rejected_by_frozen_check}` land in
  METRICS.json — the preservation arms' pool composition is auditable.

## Custody statement (what "independent" means here)

- `final_holdout`: answers ship nowhere — only its digest is bound.
- `within_holdout`: shipped in the corpus, but its answers are consumed
  only inside the evaluator's runtime (the Colab session the operator
  runs). The learning agent (this repository's author-side process)
  never executes it. That is statistical independence on unseen inputs
  plus evaluator-side custody — not a third-party attestation. A
  hardened v4.1.0+ could move within_holdout answers behind the same
  seal as final_holdout if a second operator runs the confirmation.
