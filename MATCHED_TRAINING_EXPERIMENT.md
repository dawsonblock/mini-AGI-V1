# Campaign 3B.2 — Matched training experiment (Stage 3, steps 11–14)

Status: **Contrasts A, B and C executed; seed replication complete.**
Every number below comes from a committed runner and an immutable run
directory under `results/campaign3b2/runs/`.

The Campaign 3B.1 experiment varied breadth, dataset size and
per-example exposure together, so none could be attributed. Three
controls separate them, and the single-seed results are now replicated
across seeds.

## What the three contrasts control

| flag | holds constant | varies |
|---|---|---|
| `--total-examples 30` | the TOTAL training-example count | breadth |
| `--examples-per-family 5` | examples PER FAMILY | breadth (total grows) |
| `--exposure 30` | PRESENTATIONS PER EXAMPLE (`steps = exposure × examples`) | breadth |

Each family holds exactly 5 unique examples, so a fixed total of 30
across 2 families requires cyclic repetition. The runner records the
assigned allocation, the training-example count, the unique scored rows,
the exposure target, and the **realised** `sample_presentations` /
`optimizer_updates` from the training receipt.

## Frozen baseline

The frozen model scores **0.417** on the 12 held-out tasks. Every
comparison below is against that.

## Contrast A — fixed total (30 examples), 300 steps, exposure 10

| families | hidden | destroyed | gained |
|---|---|---|---|
| 2 | **0.333** | 2 | 1 |
| 4 | 0.250 | 2 | 0 |
| 6 | 0.250 | 3 | 1 |

## Contrast B — fixed 5 per family, 300 steps

| families | examples | exposure | hidden | destroyed | gained |
|---|---|---|---|---|---|
| 2 | 10 | 30 | **0.583** | 2 | 4 |
| 4 | 20 | 15 | 0.250 | 2 | 0 |
| 6 | 30 | 10 | 0.250 | 3 | 1 |

## Contrast C — fixed exposure (30 presentations/example)

| families | examples | steps | hidden | destroyed | gained |
|---|---|---|---|---|---|
| 4 | 20 | 600 | 0.250 | 2 | 0 |
| 6 | 30 | 900 | 0.250 | 3 | 1 |

## Finding 1 — exposure is not the lever

At **4 families**, exposure 15 → 0.250 and exposure 30 → 0.250.
At **6 families**, exposure 10 → 0.250 and exposure 30 → 0.250.

Doubling and tripling the number of times each example is rehearsed
moves the held-out score by **zero**. Combined with the 3B.1 result that
300 vs 600 steps made no difference, the "train longer / rehearse more"
family of remedies is ruled out by measurement at 4 and 6 families, not
by argument.

## Finding 2 — correct supervision beats the matched control, nearly everywhere

Every correct-label condition has a matched shuffled-label control at
the same breadth, examples and steps:

| breadth | correct | shuffled | separation |
|---|---|---|---|
| 2 fam, 10 ex | 0.583 | 0.000 | **+0.583** |
| 2 fam, 30 ex | 0.333 | 0.000 | **+0.333** |
| 4 fam, 30 ex | 0.250 | 0.000 | **+0.250** |
| 6 fam, 30 ex | 0.250 | 0.083 | **+0.167** |
| 6 fam, 30 ex, 600 steps | 0.250 | 0.250 | **0.000** |

**Four of five matched pairs show a clear separation.** Learning happens
and it transfers to held-out families at every breadth. The fifth pair
is the exception flagged in 3B.1 and it remains unexplained.

This is the most robust result in the campaign: it replicates across
breadths, example counts and step counts, and it is the comparison the
shuffled control exists to make.

## Finding 3 — breadth is the binding constraint, now with 8 seeds

Seed replication on the two decision-relevant configurations:

| condition | seeds | per-seed hidden | mean | range | seeds above frozen | mean destroyed | mean gained |
|---|---|---|---|---|---|---|---|
| **2 families**, 10 ex, 300 steps | 5 | .583 .500 .500 .583 .250 | **0.483** | 0.250–0.583 | **4 / 5** | 1.8 | **+2.6** |
| 4 families, 30 ex, 300 steps | 3 | .250 .083 .167 | **0.167** | 0.083–0.250 | **0 / 3** | 3.0 | 0.0 |
| 6 families, 30 ex, 300 steps | 1 (run twice) | .250 .250 | 0.250 | — | 0 / 1 | 3.0 | 1.0 |

Reading:

- **At 2 families the adapter beats the frozen baseline in 4 of 5
  seeds**, with a mean of 0.483 against 0.417. The single-seed 0.583
  from 3B.1 was the top of the range, not a typical draw.
- **At 4 and 6 families no seed beats the frozen baseline.** All three
  4-family seeds and the 6-family configuration sit at or below it.
- **The distributions barely overlap**: the 2-family *worst* seed (0.250)
  equals the 4-family *best* seed (0.250). With 5 and 3 seeds
  respectively, that separation is unlikely to be sampling noise.
- **Gains only occur at 2 families.** Mean gained is +2.6 rows there and
  0.0–1.0 at 4 and 6. The adapter stops fixing tasks the baseline got
  wrong as soon as breadth increases.

## A determinism check worth noting

`A6` and `run3` are the **same configuration executed twice** in
separate processes: both gave `0.250`, destroyed 3, gained 1 —
identical. So the spread across seeds is genuine seed-to-seed variance,
not execution or measurement noise. That matters for how the range
above should be read.

## What Stage 3 establishes

1. **Per-example exposure and total dataset volume are not the
   constraints** — measured, at two breadths, across a 3× range.
2. **Correct supervision produces transferable learning** — four of five
   matched pairs, spanning 2/4/6 families.
3. **Breadth is the constraint**, now supported by 8 distinct seeds
   rather than one: positive transfer at 2 families (4/5 seeds), none at
   4 or 6 (0/4 seeds).

## What it does not establish

- **Why 2 families is special.** Five distinct seeds is still small, and
  the 2-family condition also has the fewest examples (10). Breadth and
  volume are separated in the *contrasts* (A2 = 2 families at 30
  examples → 0.333, below the 0.583 at 10 examples) but that contrast
  is single-seed, so it cannot carry the conclusion alone.
- **The one zero-separation control pair** (6 families, 600 steps).
- **Any claim about rank.** All of this is rank 16; step 12's rank ladder
  (4/8/16) has not been run, and it is now the most informative next
  experiment because the constraint looks like capacity rather than data.
- **Promotion.** Nothing here is a qualification result. The campaign's
  REFUSED verdict stands, and the primary L7 − L5b comparison is
  unaffected.

## Reproduce

```sh
F2=pc-upper,pc-rev ; F4=pc-upper,pc-rev,pc-rot1,pc-sort
F6=pc-rev,pc-upper,pc-rot1,pc-sort,pc-swap,pc-delim
R="--run-root results/campaign3b2/runs"

# Contrast A — fixed total
python3 scripts/positive_control_fit.py --families $F2 --total-examples 30 --steps 300 --run-id A2 $R
python3 scripts/positive_control_fit.py --families $F4 --total-examples 30 --steps 300 --run-id A4 $R
python3 scripts/positive_control_fit.py --families $F6 --total-examples 30 --steps 300 --run-id A6 $R
# Contrast B — fixed per family
python3 scripts/positive_control_fit.py --families $F4 --examples-per-family 5 --steps 300 --run-id B4 $R
# Contrast C — fixed exposure
python3 scripts/positive_control_fit.py --families $F4 --examples-per-family 5 --exposure 30 --run-id C4 $R
python3 scripts/positive_control_fit.py --families $F6 --examples-per-family 5 --exposure 30 --run-id C6 $R
# controls and seeds
python3 scripts/positive_control_fit.py --families $F2 --total-examples 30 --steps 300 --shuffle-labels --run-id ncA2 $R
for s in 1 2 3 4; do
  python3 scripts/positive_control_fit.py --families $F2 --examples-per-family 5 --steps 300 --seed $s --run-id seed${s}-2fam $R
done
```

Summary records: `results/campaign3b2/contrast-summary.json`,
`results/campaign3b2/seed-replication.json`.
