# Campaign 3B.2 — Single-family diagnostic

Status: **executed. One hypothesis refuted by its own test, one corpus
defect found.** 10 new runs (5 non-`pc-rot1` families × 2 seeds) plus the
2 existing `pc-rot1`-only runs.

The composition correction showed that `pc-rot1` presence dominates
held-out performance. This diagnostic asks *why* by training on each
family alone, so the family is the only variable.

## Result

One family, 5 examples, 300 steps, rank 16. Frozen baseline **0.417**.

| family | scores | mean | vs frozen |
|---|---|---|---|
| `pc-upper` | 0.667, 0.500 | **0.583** | **+0.166** |
| `pc-delim` | 0.333, 0.417 | 0.375 | −0.042 |
| `pc-rev` | 0.250, 0.500 | 0.375 | −0.042 |
| `pc-swap` | 0.250, 0.417 | 0.333 | −0.084 |
| `pc-sort` | 0.083, 0.250 | 0.167 | −0.250 |
| **`pc-rot1`** | 0.000, 0.083 | **0.042** | **−0.375** |

**Single-family outcomes span 0.042 to 0.583 — a 0.542 range.** Family
identity is a larger variable than anything else measured in this
campaign, including breadth (0.226) and exposure (0.000).

Two things follow immediately:

- **`pc-upper` alone beats the frozen baseline** (0.583 vs 0.417). A
  single-family adapter can positively transfer, so "adaptation hurts"
  was never the right summary — it depends entirely on what it is
  trained on.
- **`pc-rot1` alone is catastrophic** (0.042). Training on that one
  family destroys almost all held-out performance.

## A hypothesis I proposed and then refuted

I predicted that `pc-rot1` was toxic because **every character changes
via a position-independent mapping** — the only family with that
property.

`pc-upper` also changes every character (case) and is the **best** single
family at 0.583. So "every character changes" is not the toxic property,
and the prediction is **refuted by its own test**.

What the ordering actually tracks is not yet established. A plausible
reading, offered as a hypothesis and not a finding:

| family | output vs input | mean |
|---|---|---|
| `pc-upper` | same letters, different case | 0.583 |
| `pc-delim` / `pc-rev` / `pc-swap` | a permutation of the input | 0.375 / 0.375 / 0.333 |
| `pc-sort` | a canonicalised permutation | 0.167 |
| `pc-rot1` | **different letters** | 0.042 |

The ordering follows how far the output is from the input's surface
form — but `pc-sort` (permutation) is much worse than `pc-rev`
(permutation), so surface distance alone does not explain it. **No
mechanism is claimed.**

## A corpus defect, found by trying to interpret the above

`pc-rev` and `pc-swap` are **the same function on the training data**.
Every training word is three letters long, and for a three-letter word
"swap the first and last characters" **is** "reverse the word":

```
pc-rev("cat")  = "tac"      pc-swap("cat") = "tac"
pc-rev("dog")  = "god"      pc-swap("dog") = "god"
```

All five training words agree. So the corpus declares **six train
families but carries five distinct functions**, and their near-identical
scores (0.375 vs 0.333) are not independent evidence about each other.

**Consequence for interpretation:** every "breadth" count over the train
families overstates by one, and any claim that treats `pc-rev` and
`pc-swap` as separate families is inflated.

**Not fixed, deliberately.** v2 keeps the training block byte-identical
to v1 so that v1- and v2-trained runs see the same training condition and
are comparable. Fixing the collision would silently change every training
condition in the campaign. The defect is pinned by a test that asserts
*exactly* the known collision — so a new one fails the build, and a
silent fix also fails it — and recorded in `REMAINING_DEFECTS.md`. A
future corpus should use varied word lengths.

## What this changes about the recommended next experiment

The rank ladder is still the right next experiment, but this sharpens it
in two ways:

1. **It must be composition-controlled.** With family identity spanning
   0.542 and expected rank effects a few points, an uncontrolled ladder
   would mostly re-measure which families were in the set.
2. **It needs the v2 benchmark.** At n=12 a paired comparison detects
   only a ~44 pp effect; v2's 120 tasks detect ~14 pp. A ladder run on
   v1 could not resolve the effect it is looking for even if it exists.

Both are now in place: `configs/positive_control_tasks_v2.jsonl`
(120 held-out tasks, 8 families, byte-identical training block) and the
`--rank` / `--examples-per-family` / `--total-examples` controls.

## Reproduce

```sh
R="--run-root results/campaign3b2/runs"
for fam in pc-upper pc-rev pc-sort pc-swap pc-delim; do
  python3 scripts/positive_control_fit.py --families $fam \
      --examples-per-family 5 --steps 300 --seed 0 --run-id single_${fam}-s0 $R
done
```

Summary: `results/campaign3b2/single-family.json`.
