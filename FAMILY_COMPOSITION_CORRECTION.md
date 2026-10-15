# Campaign 3B.2 — Correction: breadth was confounded with family composition

> **Superseded scoring (3B.3, RD-18).** Accuracy numbers in this
> document were produced by `exact-match-v1` (case-insensitive), which
> accepted wrong-case outputs on case-transformation tasks. Treat every
> accuracy/delta below as **provisional**; strict `exact-match-v2`
> rescores live in `results/campaign3b2/RESCORE_SUMMARY.json`, each
> run's `RESCORE-STRICT.json`, and `CAMPAIGN3B_RESCORE_REPORT.md`.
> Corrections in brief: frozen hidden 0.417 -> 0.333; L5b replay
> advantage = one net row (p = 1.0), not two; rank-ladder magnitudes
> are additionally confounded by alpha/r covariation (RD-13).

Status: **the published "breadth is the binding constraint" conclusion is
WITHDRAWN.** The dominant variable is whether `pc-rot1` is in the
training set. 28 correct-label runs, perfect separation.

This correction was prompted by external review, which warned that
"family breadth is also associated with which particular transformations
are included" and asked for family composition to be balanced or varied.
That warning was correct, and the effect it exposed is **larger than the
effect I published.**

## What was published

`MATCHED_TRAINING_EXPERIMENT.md` reported:

| breadth | runs | mean | above frozen (0.417) |
|---|---|---|---|
| 2 families | 7 | 0.476 | 5/7 |
| 4 families | 5 | 0.200 | 0/5 |
| 6 families | 4 | 0.250 | 0/4 |

and concluded "breadth is the constraint."

## The confound

The 2-family condition was **always** `pc-upper,pc-rev`. The 4-family
condition was that pair **plus** `pc-rot1,pc-sort`. The 6-family
condition was all six. So breadth and composition moved together: every
condition with breadth > 2 contained `pc-rot1`, and the 2-family
condition never did.

## The test

Hold everything at the winning 2-family configuration (5 examples per
family, 300 steps, rank 16, same model, same seeds) and vary **only**
which two families:

| families | n | mean | range | above frozen |
|---|---|---|---|---|
| pc-upper, pc-rev | 3 | **0.500** | 0.500–0.500 | 3/3 |
| pc-swap, pc-delim | 3 | **0.444** | 0.417–0.500 | 1/3 |
| pc-upper, pc-rot1 | 3 | 0.222 | 0.167–0.250 | 0/3 |
| pc-rot1, pc-sort | 3 | **0.111** | 0.000–0.167 | 0/3 |

**Spread across pairs: 0.389.** The published breadth effect was 0.226.
The composition spread is **1.7× larger than the effect it was
confounded with.**

## The decisive test — breadth without `pc-rot1`

If breadth were the driver, high breadth without `pc-rot1` would still
fail. It does not:

| condition | n | mean | range | above frozen |
|---|---|---|---|---|
| 4 families, no rot1 | 3 | **0.528** | 0.500–0.583 | **3/3** |
| 5 families, no rot1 | 3 | 0.417 | 0.333–0.583 | 1/3 |
| 4 families, with rot1 | 5 | 0.200 | 0.083–0.250 | 0/5 |
| 6 families, with rot1 | 4 | 0.250 | 0.250–0.250 | 0/4 |

**Same breadth (4), same examples, same steps: 0.528 without `pc-rot1`
versus 0.200 with it.** Breadth is exonerated at 4 families.

## The corrected picture

| group | n | mean | range | above frozen |
|---|---|---|---|---|
| **contains `pc-rot1`** | 17 | **0.181** | 0.000–0.250 | **0 / 17** |
| **without `pc-rot1`** | 19 | **0.474** | 0.250–0.583 | **13 / 19** |

**Perfect separation.** Every run containing `pc-rot1` scores at or below
0.250; every run without it scores at or above 0.250. The boundary is
exactly 0.250.

### Breadth has no clean independent effect

Within the no-`pc-rot1` subset: 2 families 0.474 (n=13), **4 families
0.528 (n=3)**, 5 families 0.417 (n=3). Not monotone — 4 families beats
2. There is no breadth decline to explain once composition is
controlled.

Within the `pc-rot1` subset, breadth *helps* slightly (1 family 0.042,
2 families 0.167, 4 families 0.200, 6 families 0.250): adding other
families **dilutes** `pc-rot1`'s share of the training signal. It never
rescues the condition — 0/17 above frozen at any breadth.

### `pc-rot1` alone is catastrophic

| condition | hidden |
|---|---|
| `pc-rot1` alone (1 family) | 0.000, 0.083 → mean **0.042** |
| `pc-rot1` + 1 other | 0.111–0.222 |
| `pc-rot1` + 3 others | 0.200 |
| `pc-rot1` + 5 others | 0.250 |

Training on `pc-rot1` alone destroys almost all held-out performance.
The family is not merely unhelpful; it is actively harmful, and its harm
is diluted — not eliminated — by adding other families.

## What this means

**Withdrawn:** "breadth is the binding constraint."

**Replaced by:** *training-set composition dominates; specifically,
including `pc-rot1` caps held-out performance at 0.250 regardless of
breadth, and `pc-rot1` alone reduces it to 0.042.* Breadth has no clean
independent effect once composition is held fixed.

**Unchanged:** correct supervision still beats matched shuffled controls
in four of five pairs; exposure and step count still do nothing; the
campaign verdict is still REFUSED.

## Why this matters beyond the correction

The mechanism is now a **specific, testable property of one
transformation** rather than a vague breadth effect. That is a much
better position for the next experiment:

- `pc-rot1` is "rotate each letter forward by one (z wraps to a)". It is
  the only family whose output is **character-wise and positionally
  local** — every character changes, and the mapping is the same at every
  position.
- A plausible hypothesis: an adapter trained on a per-character
  positional shift learns a low-entropy, position-independent edit that
  then dominates the output distribution on any input. `pc-upper` is
  also character-wise but only changes *case*, which the tokenizer may
  not separate cleanly.
- That is a **hypothesis**, not a finding. Testing it means running the
  other character-wise families individually — `pc-upper`, `pc-sort`,
  `pc-rev` — which is cheap and is the obvious next experiment.

## Reproduce

```sh
R="--run-root results/campaign3b2/runs"
# composition ladder — 2 families, vary which
python3 scripts/positive_control_fit.py --families pc-upper,pc-rev  --examples-per-family 5 --steps 300 --seed 0 --run-id pcupper_pcrev-s0 $R
python3 scripts/positive_control_fit.py --families pc-rot1,pc-sort  --examples-per-family 5 --steps 300 --seed 0 --run-id pcrot1_pcsort-s0 $R
# decisive test — breadth WITHOUT pc-rot1
python3 scripts/positive_control_fit.py --families pc-upper,pc-rev,pc-swap,pc-delim --examples-per-family 5 --steps 300 --seed 0 --run-id 4fam_norot1-s0 $R
# is pc-rot1 harmful alone?
python3 scripts/positive_control_fit.py --families pc-rot1 --examples-per-family 5 --steps 300 --seed 0 --run-id rot1only-s0 $R
```

Summary: `results/campaign3b2/family-composition.json`.
