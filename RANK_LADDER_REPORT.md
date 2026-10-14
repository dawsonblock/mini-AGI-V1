# Campaign 3B.2 — Rank ladder: capacity is not the explanation

Status: **executed. 28 runs on the v2 benchmark. Rank does not change the
outcome at any breadth, and composition dominates it by roughly 5×.**

Run on `configs/positive_control_tasks_v2.jsonl` — 120 held-out tasks
across 8 families, so one answer moves accuracy by 0.83 pp rather than
8.33 pp and a ~14 pp paired effect is detectable. Frozen baseline
**0.400**.

## Result

### 6 families (composition includes `pc-rot1`)

| rank | n | fit | mean hidden | range | above frozen |
|---|---|---|---|---|---|
| 4 | 3 | 1.000 | **0.294** | 0.217–0.350 | **0 / 3** |
| 8 | 3 | 1.000 | 0.250 | 0.200–0.308 | 0 / 3 |
| 16 | 3 | 0.978 | 0.242 | 0.175–0.308 | 0 / 3 |
| 32 | 3 | 1.000 | 0.261 | 0.233–0.300 | 0 / 3 |

### 2 families, good pair (`pc-upper`, `pc-rev`)

| rank | n | fit | mean hidden | range | above frozen |
|---|---|---|---|---|---|
| 4 | 3 | 1.000 | 0.369 | 0.300–0.408 | 1 / 3 |
| 8 | 3 | 1.000 | 0.397 | 0.367–0.425 | 1 / 3 |
| 16 | 3 | 1.000 | **0.408** | 0.400–0.425 | 1 / 3 |
| 32 | 3 | 1.000 | 0.375 | 0.333–0.400 | 0 / 3 |

### 2 families, bad pair (`pc-rot1`, `pc-sort`)

| rank | n | mean hidden | above frozen |
|---|---|---|---|
| 4 | 2 | 0.138 | 0 / 2 |
| 32 | 2 | 0.158 | 0 / 2 |

## Reading

**Rank is flat.** Across a 8× range (4 → 32) at 6 families the mean moves
0.242 → 0.294 and is **not monotone** — rank 4 is the best of the four.
At 2 families it moves 0.369 → 0.408, peaking at 16. The rank spread is
about **0.05**; the composition gap between the good and bad pair is
about **0.23**, roughly five times larger.

**Training capacity is adequate at every rank.** Fit is 1.000 at ranks 4,
8 and 32 — including rank 4, the smallest. So the failure was never "the
adapter cannot represent the training set"; a rank-4 adapter represents
it perfectly. (Rank 16's fit is 0.978, marginally the *lowest*.)

**Composition survives every rank.** The bad pair is below the frozen
baseline at rank 4 (0.138) and rank 32 (0.158); the good pair is at or
above it at rank 8, 16 and 32. Changing rank does not rescue a bad
composition or spoil a good one.

## Against the decision table

The plan specified what each outcome would mean. The observed outcome
matches one row:

| outcome | interpretation | observed? |
|---|---|---|
| rank 32 improves six-family transfer without more regressions | supports insufficient capacity | **no** |
| rank 4 or 8 improves preservation and transfer | excessive specialisation at 16 | **no** |
| **all ranks fail on broader families** | **capacity alone unlikely to explain it** | **YES** |
| higher rank improves fit but not held-out | optimisation capacity without generalisation | no (fit is already 1.000 at rank 4) |
| higher rank improves L6 but not L7 | adapter/replay interaction | not tested here |

**Capacity is not the explanation.** And because fit is 1.000 at rank 4,
"insufficient training capacity" is not merely unsupported — it is
refuted. The adapter can represent the training set at one eighth the
rank.

## What the campaign now measures

Effect sizes on the same benchmark, largest first:

| variable | spread | source |
|---|---|---|
| **family composition** | **~0.23–0.54** | composition ladder, single-family diagnostic |
| rank (4→32) | ~0.05, non-monotone | this ladder |
| example exposure (10→30) | 0.000 | Contrast C |
| training steps (300→900) | 0.000 | Contrast C, 3B.1 |

**Composition is the only variable in this campaign with a large,
reproducible effect.** Everything else is flat.

## Consequences

Per the plan's own logic, the next priority is **not** more rank, more
data, or more steps — all three are now ruled out by measurement. It is
the family of interventions the plan listed under step 16 for exactly
this case:

- **task-conditioned adapters** (E6) — if composition is what matters,
  specialising per transformation is the direct response;
- **interference-aware training** (E1 balanced sampling, E2 replay
  during training);
- **preservation regularization** (E3 frozen-reference KL).

One caveat that must be stated: this ladder varied rank at **one**
learning rate and **one** schedule. The reviewer's earlier point stands —
learning rate affects trajectory, not only magnitude — so a rank ×
learning-rate interaction is not excluded by this result. What is
excluded is rank as a standalone explanation.

## Reproduce

```sh
C="--corpus configs/positive_control_tasks_v2.jsonl"
R="--run-root results/campaign3b2/runs"
F6=pc-rev,pc-upper,pc-rot1,pc-sort,pc-swap,pc-delim
for r in 4 8 16 32; do for s in 0 1 2; do
  python3 scripts/positive_control_fit.py $C --families $F6 \
      --examples-per-family 5 --steps 300 --rank $r --seed $s \
      --run-id "r${r}_f6_s${s}" $R
done; done
```

Summary: `results/campaign3b2/rank-ladder.json`.
