# Scoring and statistics report

Status: **implemented and verified**. Modules:
`src-python/minagi/v161/prompt_contract.py`,
`src-python/minagi/v161/stats.py`. Tests:
`test_campaign_output_scoring.py`, `test_false_activation_undefined.py`,
`test_family_cluster_statistics.py`, `test_v165_qualifier_chain.py`.

## Part 1 — Scoring

### Strict exact match is primary; diagnostics never replace it

The master specification is explicit: *do not silently replace strict
exact match with prefix scoring.* The implementation keeps strict
exact match as the score that gates qualification and records the
diagnostics beside it.

`generation_diagnostics` records, per prediction:

| diagnostic | meaning |
|---|---|
| `strict_exact_match` | normalized full-string equality — the primary metric |
| `correct_first_line` | the first non-empty line equals the expected answer |
| `correct_answer_prefix` | the expected answer is a prefix of the prediction |
| `valid_answer_format` | the prediction satisfies the frozen answer-only surface |
| `generated_tokens` | length actually produced |
| `max_token_truncation` | the generation ran to the limit without stopping |
| `eos_hit` / `stop_reason` | `end_of_message` vs `max_tokens` |

A prediction that is a correct prefix followed by a contradictory
answer receives **no** primary correctness credit — the prefix is
recorded as a diagnostic only. That is the failure mode the Campaign 3A
surface exhibited in full: the historical raw prompt truncated 100% of
generations and never emitted the end-of-turn token
(`docs/science/SCORING_CALIBRATION.md`), so first-line and prefix
credit were the only signals available and strict exact match was 0.000
for the frozen model.

### The frozen output contract

```yaml
output_contract:
  instruction: "Return only the transformed result. Do not provide an explanation."
  max_new_tokens: 64
  stop_on_eos: true
```

Selected by a **predetermined** comparison on development rows only
(no search, no sealed data), by the deterministic rule: highest mean
strict exact match → highest mean valid-format rate → fewest mean
generated tokens. The selection and the losing configurations are both
recorded in `docs/science/SCORING_CALIBRATION.md`.

The contract's digest travels in signed per-arm metrics
(`output_contract_digest`), in the training receipt, and in the sealed
confirmation submission, so a change to the instruction, the limit or
the stop policy is visible in evidence rather than inferred.

### Protocol versioning

A scorer or generation-policy change requires a **new scientific
protocol version**. It does not retroactively change Campaign 3A's
decision, its thresholds or its recorded scores. The historical default
output contract (no instruction) remains available precisely so that
3A-era records replay under the surface they were produced on.

## Part 2 — Statistics

### SCI-305 — an undefined rate is not zero

`false_activation_rate` conditions on the baseline-correct population
(rows the frozen model answered correctly), matching the preregistered
bound "off-target harmful flips ≤ 1% of previously correct baseline
cases". The original implementation divided by *all* pairs, diluting
the rate by rows the baseline already failed.

When no baseline row is correct, the denominator is empty. The function
returns:

```
{"false_activation_rate": None, "conditional_regression_rate": None,
 "n_baseline_correct": 0, "n_false_activation": <count>,
 "status": "NOT_ESTIMABLE"}
```

It never returns `0.0`, which would claim demonstrated absence of
off-target flips from evidence that cannot measure them. The qualifier
treats `NOT_ESTIMABLE` as a reason to refuse:

> false-activation rate NOT_ESTIMABLE for N/M seed(s) — no
> baseline-correct hidden population; absence of off-target flips is
> not demonstrated

The same rule is applied to the comparative retention category in the
governed safety suite: an empty baseline-correct denominator is
UNDEFINED and blocks.

### Repeated seeds are not independent task samples

Ten seeds evaluating the same task templates are not ten independent
samples of task difficulty. The uncertainty unit is therefore the
**task family**:

| function | what it resamples |
|---|---|
| `bootstrap_ci` | rows (seed-level uncertainty) |
| `cluster_bootstrap_ci` | clusters — task families — pooling their members |
| `family_delta_analysis` | per-family paired deltas, leave-one-family-out means, worst family, families-positive fraction |

`family_delta_analysis` reports `family_row_counts` alongside the
per-family means so a family carrying most of the rows is visible
rather than hidden inside a pooled average.

### What the qualifier reports

- paired baseline comparisons (L6−L5, L7−L5b, and each arm vs L1);
- per-family strict accuracy and per-family deltas;
- seed-level bootstrap CI over per-seed deltas;
- family-aware (cluster) bootstrap CI where the plan declares
  `cluster_unit: family`;
- leave-one-family-out sensitivity and worst-family degradation;
- positive-seed counts (`n_seeds_positive_ft` vs the preregistered
  `min_seeds_positive_ft`);
- overall off-target regression rates (conditional regression, retention
  drop, security drop vs L1);
- improvement per unit of compute (`cost_per_correct`, wall/train
  seconds, generated tokens, trainable parameters, arm-state bytes).

### Rejecting over-claimed transfer

A claim of broad transfer is rejected when the apparent gain depends on
one family or on repeated successes on one task. Concretely:

- `families_positive` / `fraction_families_positive` must clear the
  preregistered bar, not just the pooled mean;
- the cluster bootstrap CI lower bound must exceed
  `min_delta_ft_ci_lower` — resampling families, not rows;
- `worst_family_delta` is reported and a negative worst family is
  visible in the record even when the pooled mean is positive;
- the sample size (number of independent task families) is fixed before
  final confirmation and carried in the signed plan.

### What the executed diagnostic showed

On the Step 13 positive control the statistics did their job: the
family-aware view is what distinguishes "LoRA learns the trained
families" (validation 0.222 → 0.444) from "LoRA transfers to held-out
families" (0.167 vs 0.417 frozen). The pooled hidden number alone would
have been reported as a single negative delta; the family breakdown
says the degradation is concentrated in the held-out families, which is
the actionable finding. See `LEARNABILITY_TEST_REPORT.md`.

## Boundary

These are engineering instruments. A statistical method passing its
tests is not evidence that any adapter learned anything — that is what
the campaign evidence is for, and the executed campaign evidence is
negative.
