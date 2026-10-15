# Campaign 3B.1 — Step 4 fit experiment

Status: **executed.** Six runs, all on the pinned
`Qwen/Qwen2.5-0.5B-Instruct` revision `7ae55760…`, CPU, one seed, rank 16,
the frozen prompt and output contracts. Every claim below is a recorded
run, reproducible with `scripts/positive_control_fit.py`.

> **Superseded scoring (3B.3).** All accuracies in this document were
> produced by the case-insensitive `exact-match-v1` scorer, which
> accepted wrong-case outputs. Every number below is **provisional**;
> strict `exact-match-v2` rescores live in each run's
> `RESCORE-STRICT.json` overlay and `results/campaign3b2/RESCORE_SUMMARY.json`.
> Headline corrections: frozen hidden 0.417 → 0.333; run1 hidden
> 0.583 → 0.333; the "adapter beats frozen" claim does not survive
> strict scoring.

The 3B diagnostic never cleared the 95% fit gate (0.767), so its
cross-family failure might have been an under-fitting artifact. Step 4
separates that from a genuine interference property — and the answer is
more interesting than either.

## Result

Six runs. The four correct-label runs form a clean 2×2 (breadth ×
steps), so the two variables are separated rather than confounded.

| run | train families | steps | labels | subset fit | hidden (frozen 0.417) | broken | gained |
|---|---|---|---|---|---|---|---|
| **run1** | 2 | 300 | correct | **1.000** | **0.583** | 2 | **4** |
| run4 | 2 | 600 | correct | 1.000 | **0.583** | 2 | 4 |
| nc1 | 2 | 300 | shuffled | 0.000 | **0.000** | 5 | 0 |
| run3 | 6 | 300 | correct | 1.000 | 0.250 | 3 | 1 |
| run2 | 6 | 600 | correct | 1.000 | 0.250 | 3 | 1 |
| nc2 | 6 | 600 | shuffled | 0.000 | **0.250** | 3 | 1 |

The 2×2, holding labels and fit constant at 1.000:

| | 300 steps | 600 steps |
|---|---|---|
| **2 families** | 0.583 | 0.583 |
| **6 families** | 0.250 | 0.250 |

**Steps have no effect at either breadth. Breadth is the variable.**
Doubling the training duration moves the held-out score by 0.000 in both
rows, while going from 2 to 6 transformations moves it by −0.333 at both
step counts.

For reference, the Campaign 3B campaign arms on the same hidden block:
frozen 0.417 · L5b verified replay 0.583 · L6 0.167 · L7 0.083 · NC 0.250.

## Finding 1 — fixed-rank LoRA can fit *and* positively transfer

At **2 families / 10 examples / 300 steps** the adapter reaches
**1.000 subset accuracy** and **improves held-out families from 0.417 to
0.583** — it gains 4 rows the frozen model missed and breaks 2 it had
right. It also breaks 2 of the 3 `pc-len` rows, so interference is not
gone; the net is positive.

This is the first **controlled positive** result in the campaign: the
matched negative control on the same prompts, steps and schedule scores
**0.000** on the same rows.

| | subset fit | hidden |
|---|---|---|
| correct labels | 1.000 | **0.583** |
| shuffled labels | 0.000 | **0.000** |
| **separation** | | **+0.583** |

A 0.583 separation against a matched shuffled-label control is learning,
not procedure. **So the answer to Step 4's question is yes: fixed-rank
LoRA reliably fits a small verified subset, and a fitted narrow adapter
generalizes positively.**

## Finding 2 — breadth, not fit, is the binding constraint

At **6 families / 30 examples / 600 steps** the adapter also reaches
**1.000 subset accuracy** — but the held-out block **drops to 0.250**,
and the matched shuffled-label control scores **exactly the same 0.250**.

| | subset fit | hidden |
|---|---|---|
| correct labels | 1.000 | 0.250 |
| shuffled labels | 0.000 | 0.250 |
| **separation** | | **0.000** |

**No measurable benefit from supervision at 6 transformations.** The
correctly-trained adapter performs on held-out families exactly as well
as one that learned nothing.

One honesty note on that number: the two runs are *not* row-for-row
identical — they get different subsets right and land on the same total
(3/12 each). So the correct statement is "no measurable advantage", not
"identical mechanism". The aggregate is 0.250 both ways; the rows differ.

## Finding 3 — the 3B negative result decomposes

| effect | evidence |
|---|---|
| part **under-fitting** | 6 families at fit 0.767 → hidden 0.167; at fit 1.000 → 0.250 |
| part **supervision-independent damage** | at 6 families, correct (0.250) = shuffled (0.250) — the loss is the adapter's presence, not what it learned |
| part **genuine capacity limit** | at 2 families the same rank and schedule generalizes *positively* (+0.583 over control), and duration is irrelevant (300 = 600 steps) |

So the Campaign 3B "adapter destroys held-out families" is real, but the
mechanism is not over-fitting to labels and not wrong supervision. At
rank 16 across six transformations the adapter damages the base model
**regardless of what it is trained on**; given a narrower job it helps.

## What this changes about the next step

The 3B.1 plan's step 6 suggested preservation mechanisms — frozen
reference regularization, replay during training, constrained updates.
This result sharpens which of those is worth trying and rules one
approach out:

- **Adding more training data will not help.** At 6 families supervision
  already makes no measurable difference; more of it is more of the same.
- **Training longer will not help.** The 2×2 shows duration has no
  effect at either breadth — 300 and 600 steps land on identical scores.
- **Better labels will not help.** The control *is* the better-labels
  comparison, and it ties.
- **The lever is capacity and regularization**: a smaller effective
  update, per-family capacity, or an explicit preservation term. The
  2-family result shows the pipeline can learn and transfer — the
  question is how to keep that while covering six transformations.

## Limits

- **One seed, one rank (16), one learning rate, 12 hidden rows.**
  Everything here is a single draw; treat the direction as
  well-supported and the magnitudes as provisional.
- **`nc1` scoring 0.000 is itself notable** — the shuffled adapter
  destroys *everything*, breaking all 5 frozen-correct rows. That is
  consistent with a model confidently applying a wrong mapping, but it
  is one run.
- **Example count is not a confound, and the reason is checkable.** The
  trainer samples training indices *with replacement*
  (`peft_trainer.py`: `rnd.randrange(len(encoded))` per microbatch), so
  300 steps draw 300 examples from the pool regardless of pool size. The
  2-family run therefore gives each of its 10 examples ~30 repetitions;
  the 6-family run gives each of its 30 examples ~10. The 6-family
  condition is the one with *fewer* repetitions per example, so it
  cannot be the "more data" condition. The variable that differs is the
  number of distinct transformations the adapter must represent.
- **The Campaign 3B primary verdict is unchanged.** Neural adaptation
  still loses to the verified-replay baseline on the held-out families
  (L7 − L5b = −0.500). This experiment explains *why*, and shows the
  pipeline is capable of a positive result under a narrower task set —
  it does not overturn the campaign's REFUSED decision.

## Reproducibility repair (SCI-402 / SCI-403)

External review found two defects in the runner that produced the table
above. Both were real and both are fixed; the experiment was then re-run
in full.

**SCI-402 — adapters were destroyed.** Every run wrote its adapter to
``<output>.parent/adapter``. All six runs shared one output directory,
so five of the six adapters were overwritten and lost — only the JSON
summaries survived. The numbers above were still reproducible from the
summaries, but the artifacts behind them were not. Runs now own an
immutable directory built in staging and atomically published:

```
results/campaign3b2/runs/<run-id>/
    CONFIG.json  TRAINING_RECEIPT.json  ADAPTER_MANIFEST.json
    adapter/     PREDICTIONS-{FROZEN,ADAPTED,SUBSET}.jsonl
    METRICS.json RUN_COMMIT.json
```

An existing destination is refused, a failed run is recorded as FAILED
rather than dropped, every file is hashed into ``RUN_COMMIT.json``, and
the adapter must reload from disk and reproduce the in-memory outputs
before the run publishes.

**SCI-403 — the generation budget was not the contract's.** The runner
accepted ``--max-new-tokens`` independently, and generation used the CLI
value while the contract recorded its own. The two happened to agree
(both 16) in every executed run, so no result here is invalid — but the
mechanism would have diverged silently. The flag is gone; the budget
comes from ``contract.output.max_new_tokens``, and the effective
decoding configuration is recorded in every run.

### The re-run reproduces every number

All six runs were re-executed under the repaired runner:

| run | fit | hidden | destroyed | gained | vs original |
|---|---|---|---|---|---|
| run1 | 1.000 | 0.583 | 2 | 4 | **identical** |
| run4 | 1.000 | 0.583 | 2 | 4 | **identical** |
| nc1 | 0.000 | 0.000 | 5 | 0 | **identical** |
| run3 | 1.000 | 0.250 | 3 | 1 | **identical** |
| run2 | 1.000 | 0.250 | 3 | 1 | **identical** |
| nc2 | 0.000 | 0.250 | 3 | 1 | **identical** |

All six adapters now exist with **distinct** weight hashes, every run
carries 12 hashed files, and every adapter is reload-verified. So the
findings above are reproduced from complete evidence rather than from
summaries of destroyed artifacts.

## Reproduce

```sh
F2=pc-upper,pc-rev
F6=pc-rev,pc-upper,pc-rot1,pc-sort,pc-swap,pc-delim
R="--run-root results/campaign3b2/runs"

python3 scripts/positive_control_fit.py --families $F2 --steps 300 --run-id run1 $R
python3 scripts/positive_control_fit.py --families $F2 --steps 600 --run-id run4 $R
python3 scripts/positive_control_fit.py --families $F2 --steps 300 --shuffle-labels --run-id nc1 $R
python3 scripts/positive_control_fit.py --families $F6 --steps 300 --run-id run3 $R
python3 scripts/positive_control_fit.py --families $F6 --steps 600 --run-id run2 $R
python3 scripts/positive_control_fit.py --families $F6 --steps 600 --shuffle-labels --run-id nc2 $R
```

The script exits 0 only when the fit gate passes; `nc1`/`nc2` exit 2,
which is the expected negative-control outcome. Re-running with an
existing `--run-id` is refused: a committed run is never modified.
