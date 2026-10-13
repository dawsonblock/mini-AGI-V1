# Campaign 3B.1 — Step 4 fit experiment

Status: **executed.** Four runs, all on the pinned
`Qwen/Qwen2.5-0.5B-Instruct` revision `7ae55760…`, CPU, one seed, rank 16,
the frozen prompt and output contracts. Every claim below is a recorded
run, reproducible with `scripts/positive_control_fit.py`.

The 3B diagnostic never cleared the 95% fit gate (0.767), so its
cross-family failure might have been an under-fitting artifact. Step 4
separates that from a genuine interference property — and the answer is
more interesting than either.

## Result

| run | train families | steps | labels | subset fit | hidden (frozen 0.417) | broken | gained |
|---|---|---|---|---|---|---|---|
| **run1** | 2 | 300 | correct | **1.000** | **0.583** | 2 | **4** |
| nc1 | 2 | 300 | shuffled | 0.000 | **0.000** | 5 | 0 |
| run2 | 6 | 600 | correct | 1.000 | 0.250 | 3 | 1 |
| nc2 | 6 | 600 | shuffled | 0.000 | **0.250** | 3 | 1 |

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
| part **genuine capacity limit** | at 2 families the same rank and schedule generalizes *positively* (+0.583 over control) |

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
- **2 vs 6 families also changes example count** (10 vs 30) and step
  count (300 vs 600). The experiment does not isolate breadth from data
  volume; a 6-family run at 300 steps, or a 2-family run at 600, would.
  That is the obvious next ablation and it is cheap.
- **The Campaign 3B primary verdict is unchanged.** Neural adaptation
  still loses to the verified-replay baseline on the held-out families
  (L7 − L5b = −0.500). This experiment explains *why*, and shows the
  pipeline is capable of a positive result under a narrower task set —
  it does not overturn the campaign's REFUSED decision.

## Reproduce

```sh
python3 scripts/positive_control_fit.py \
    --families pc-upper,pc-rev --steps 300 \
    --output run1.json
python3 scripts/positive_control_fit.py \
    --families pc-upper,pc-rev --steps 300 --shuffle-labels \
    --output nc1.json
python3 scripts/positive_control_fit.py \
    --families pc-rev,pc-upper,pc-rot1,pc-sort,pc-swap,pc-delim \
    --steps 600 --output run2.json
python3 scripts/positive_control_fit.py \
    --families pc-rev,pc-upper,pc-rot1,pc-sort,pc-swap,pc-delim \
    --steps 600 --shuffle-labels --output nc2.json
```

All four exit 0 only when the fit gate passes; `nc1`/`nc2` exit 2, which
is the expected negative-control outcome.
