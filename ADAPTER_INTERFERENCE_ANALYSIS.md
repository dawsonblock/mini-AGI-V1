# Campaign 3B.2 — Adapter interference analysis (Stage 2, steps 7–10)

Status: **steps 7–9 executed. Step 10 partially resolved, with one
hypothesis ruled out by measurement.** Every number below is produced by
a committed tool from preserved evidence, not from a hand-maintained
summary.

Two new tools:

| tool | what it does |
|---|---|
| `scripts/analysis/adapter_interference.py` | reproduces the paired comparison from preserved per-row predictions and classifies every task |
| `scripts/analysis/adapter_counterfactual.py` | adapter-on/off on **one** model instance with **one** tokenized input per task |

## Step 7 — Reproduction from preserved predictions

`adapter_interference.py` reads the campaign's `PREDICTIONS-<ARM>.jsonl`
files and the committed corpus. L7 vs L5b on the 12 held-out tasks:

| quantity | value |
|---|---|
| baseline (L5b) accuracy | 0.583 |
| treatment (L7) accuracy | 0.083 |
| preserved-correct | 1 |
| newly-correct | **0** |
| destroyed-correct | **6** |
| still-incorrect | 5 |
| **D** = destroyed / baseline-correct | **0.857** [ESTIMATED] |
| inputs identical | **12 / 12** |

The Campaign 3B.1 claim — six losses, no wins — **reproduces exactly**
from the preserved predictions. The `inputs identical: 12/12` column is
the control: L5b and L7 share the replay context, so any difference is
the adapter.

Contrast cases, same tool:

| comparison | baseline | treatment | destroyed | gained | D | inputs identical |
|---|---|---|---|---|---|---|
| L7 vs L5b | 0.583 | 0.083 | 6 | 0 | 0.857 | 12/12 |
| L6 vs L1 | 0.417 | 0.167 | 3 | 0 | 0.600 | 12/12 |
| NC vs L1 | 0.417 | 0.250 | 2 | 0 | 0.400 | 12/12 |
| L6 vs L5b | 0.583 | 0.167 | 5 | 0 | 0.714 | **0/12** |

The last row is expected and worth reading: L6 is evaluated *without* the
replay context L5b has, so its prompts genuinely differ. That is why the
counterfactual below matters.

**`newly-correct` is zero in every comparison.** Across all four
comparisons the adapter never fixes a task the baseline got wrong.

### Per-family destruction (L7 vs L5b)

| family | n | baseline correct | destroyed | gained |
|---|---|---|---|---|
| pc-len | 3 | 3 | **3** | 0 |
| pc-vowels | 3 | 3 | 2 | 0 |
| pc-dup | 3 | 1 | 1 | 0 |
| pc-mirror | 3 | 0 | 0 | 0 |

## Step 8 — Adapter-on/off counterfactual

The strongest control in this campaign. One frozen base-model instance,
one tokenization per task reused by every condition, identical decoding
from the frozen contract, adapter toggled via peft's `disable_adapter`.

**The toggle is faithful:** `adapter-off-after` reproduces
`adapter-off-before` **exactly** (same accuracy, 0 destroyed, 0 gained).
If it had not, the tool reports FAILED and refuses to draw conclusions.
So the interference is not a model-reload, tokenization or ordering
artifact.

| condition | accuracy | destroyed | gained | D |
|---|---|---|---|---|
| adapter-off-before | 0.417 | 0 | 0 | 0.000 |
| adapter-off-after | 0.417 | 0 | 0 | 0.000 |
| **adapter-on[campaign-L6]** | 0.167 | **3** | **0** | **0.600** |
| adapter-on[campaign-NC] | 0.250 | 2 | 0 | 0.400 |
| **adapter-on[fit-run1]** | **0.583** | 2 | **4** | 0.400 |
| adapter-on[fit-run2] | 0.250 | 3 | 1 | 0.600 |

Reading:

- The campaign adapter destroys 3 of 5 baseline-correct tasks and gains
  none — the campaign finding, reproduced on a single instance.
- **`fit-run1` beats the baseline** (0.583 vs 0.417, +4 gained against
  2 destroyed). A well-fit narrow adapter is a net improvement under the
  same counterfactual.
- The shuffled-label control also destroys 2. Learning something is not
  required to damage the model — which is why every arm needs a control.

### The `pc-len` diagnostic family

| condition | pc-len correct (baseline 3/3) |
|---|---|
| adapter-off-before / after | 3 / 3 |
| campaign-L6 | **1 / 3** |
| fit-run1 | **1 / 3** |
| fit-run2 | **1 / 3** |
| campaign-NC | **3 / 3** |

Character counting is damaged by **every trained adapter**, including
the one that is a net improvement, and preserved by the control that
learned nothing. So capability preservation and net accuracy are
**separable**: `fit-run1` improves overall while still breaking a
specific capability. Any claim of the form "adaptation preserves
existing abilities" has to answer this row.

## Step 9 — Diagnostics

### Task-family confusion

Does the adapter emit a *trained* family's transformation on an
unrelated task? Measured against the corpus's own transformation
functions, imported from the generator so the check cannot drift.

| comparison | exact matches | casefold-only |
|---|---|---|
| L7 vs L5b | **2** | 5 |
| L6 vs L1 | 1 | — |
| NC vs L1 | 0 | — |

**A correction to my own earlier reporting.** I previously wrote "6 of 12
tasks emit an exact trained transformation". That was casefolded, and
casefolded comparison conflates distinct transformations —
`pc-sort("egg")` is `"egg"`, which casefolds equal to
`pc-upper("egg")` = `"EGG"`. The tool now separates exact from
casefold-only matches. The accurate statement is **2 exact, 5
casefold-only**. Scoring stays casefolded (the harness does); the
*diagnosis* does not, because over-counting a confusion signal is its
own error.

Examples of genuine confusion:

| task | asked | emitted | is |
|---|---|---|---|
| pc-hd-pc-dup-1 | duplicate `ice` | `ICE` | `pc-upper` |
| pc-hd-pc-mirror-0 | mirror `egg` | `egg` | `pc-sort` |

So the adapter sometimes applies a *trained* transformation to a task
that asked for a different one. But it is **2 of 12, not 6** — the
over-application story explains a minority of the damage, not most of
it.

### Output-format drift

**Zero.** `n_broke = 0` for L7 vs L5b. The adapter does not break the
answer-only contract: its outputs are still short, single-line,
answer-shaped. The failure is **content**, not format — which rules out
the "the adapter stopped following the output contract" hypothesis.

### Adapter weight geometry

Per-tensor L2, split by projection (A is random-initialised and carries
no learning; B is zero-initialised, so B is the informative column):

| adapter | A L2 | B L2 | B/A | destroyed | gained |
|---|---|---|---|---|---|
| campaign-L6 | 22.708 | 1.420 | 0.0625 | 3 | 0 |
| campaign-NC | 22.743 | 1.453 | 0.0639 | 2 | 0 |
| fit-run1 | 22.717 | 1.440 | 0.0634 | 2 | **4** |
| fit-run2 | 22.808 | **2.043** | **0.0896** | 3 | 1 |

Two readings, one of which rules a hypothesis out:

- `fit-run2` carries a **42% larger** learned update than the others.
- **But update magnitude does not predict destruction.** `campaign-L6`
  (B = 1.420) and `fit-run1` (B = 1.440) are within 1.5% of each other
  and yet one gains 0 while the other gains 4.

**So the interference is not an update-size problem.** Same-size
updates produce opposite outcomes; the difference is in what the update
*encodes*. This is direct evidence against the "lower the learning rate"
remedy (step 16 E4) as a general fix — the update is already small.

## Step 10 — Separating the failure modes

| observation | interpretation | status |
|---|---|---|
| training accuracy poor | optimisation / training-contract issue | **ruled out** — both breadths reach 1.000 fit |
| training high, within-family weak | generalisation problem | not observed — within-family rises 0.222 → 0.444 |
| within-family strong, unrelated tasks degrade | interference or specialisation | **observed** — the headline |
| LoRA alone works, LoRA + replay fails | interaction with in-context examples | **observed** — L5b 0.583 vs L7 0.083 on identical inputs |
| base correct, adapted wrong | capability regression | **observed** — D = 0.857, `pc-len` 3/3 → 1/3 |
| correct adapter no better than shuffled control | learning evidence inadequate | **observed at 6 families** (0.250 vs 0.250); **not** observed at 2 (0.583 vs 0.000) |
| update magnitude drives damage | — | **ruled out** — same B norm, opposite outcomes |
| output-contract breakage | — | **ruled out** — format drift 0 |

Two hypotheses are now closed by measurement: update size, and output
format. One is confirmed as a *minor* contributor: trained-transformation
over-application (2/12 exact).

## What remains open

- **Why `pc-len` specifically.** It is the sharpest and most consistent
  casualty, damaged by every trained adapter and preserved by the
  control. Character counting is not a transformation of the input
  string, so an adapter whose update is dominated by
  string-transformation behaviour plausibly displaces it. That is a
  hypothesis, not a finding.
- **The `newly-correct = 0` result.** No comparison produced a single
  gain on the held-out set. `fit-run1` gains 4 under the counterfactual,
  so gains are possible — the discrepancy between that and the campaign
  arms needs explaining.
- **Step 10 is not fully closed** because the counterfactual and the
  campaign disagree about whether gains occur, and that disagreement is
  itself unexplained.

## Reproduce

```sh
# step 7 — from preserved predictions
python3 scripts/analysis/adapter_interference.py \
    --predictions-dir <run>/seed-0 --treatment L7 --baseline L5b \
    --output INTERFERENCE.json

# step 8 — counterfactual, one instance, matched token ids
python3 scripts/analysis/adapter_counterfactual.py \
    --adapter L6=<path> --adapter NC=<path> \
    --adapter fit-run1=results/campaign3b2/runs/run1/adapter \
    --adapter fit-run2=results/campaign3b2/runs/run2/adapter \
    --output COUNTERFACTUAL.json
```

Tests: `test_adapter_interference_analysis.py` (7) pins the four-way
classification, the `NOT_ESTIMABLE` rule, missing-prediction reporting,
input-identity detection, exact-vs-casefold confusion, and format drift.
