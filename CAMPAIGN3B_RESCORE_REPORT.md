# Campaign 3B — strict rescore report (3B.3 P0)

Status: **executed and reproducible.** Every number below is recomputed
from retained prediction artifacts by `scripts/analysis/rescore_strict.py`
under the corrected oracle `exact-match-v2` (case-sensitive, task-aware).
Nothing is re-generated; the overlays sit beside the original artifacts
and each run's `RESCORE-STRICT.json` binds per-row v1/v2 scores.

## The defect

`exact_match` compared `.casefold()` on both prediction and expected
answer. On a benchmark where the task IS a case transformation
(`pc-vowels`, `pc-lastupper`), wrong-case outputs scored correct. The
same casefold lived in three places: `src-python/minagi/v161/evaluators.py`,
the fit runner's local scorer, and `prompt_contract.py`'s diagnostic
field named `strict_exact_match` — which was not strict.

## Blast radius

| quantity | casefold (v1) | strict (v2) |
|---|---|---|
| pooled frozen hidden mean | 0.411 | **0.297** |
| pooled adapted hidden mean | 0.311 | **0.172** |
| adapted `pc-vowels` family | 0.634 | **0.020** |
| adapted `pc-lastupper` family | 0.713 | **0.000** |
| retained runs rescored | — | 80 |
| scored rows flipping | — | ~1,302 / 9,419 |

## Campaign 3B arms, strict

| arm | v1 | v2 | vs L1 (strict, paired, n=12) |
|---|---|---|---|
| L1 frozen | 0.417 | **0.333** | — |
| L5 | 0.417 | 0.333 | gained 0 / destroyed 0 / net 0 |
| **L5b replay** | 0.583 | **0.417** | gained 1 / destroyed 0 / **net +1, sign-test p = 1.0** |
| L6 LoRA | 0.167 | 0.167 | gained 0 / destroyed 2 / net −2, p = 0.5 |
| L7 LoRA+replay | 0.083 | 0.083 | gained 1 / destroyed 4 / net −3, p = 0.375 |
| NC shuffled | 0.250 | 0.250 | gained 0 / destroyed 1 / net −1, p = 1.0 |

## Corrections this forces

1. **The frozen baseline was inflated.** 0.417 → 0.333 (one wrong-case
   row). Every v1-derived comparison shifted with it.
2. **The replay advantage is one net row, not two.** L5b − L1 strict =
   +0.083, `gained=1, destroyed=0`, sign-test p = 1.0. Under the v1
   oracle the same comparison was +2 net, p = 0.50. Either way there is
   **no statistical evidence of superiority** at n=12 — a promising
   direction, not a finding.
3. **The 3B.1 "positive transfer" headline collapses.** run1/run4
   0.583 → 0.333: the apparent above-frozen transfer on
   `pc-vowels`/`pc-lastupper` was substantially a scoring artifact.
4. **The destruction signal survives strict scoring and sharpens:**
   L7 destroys all 4 frozen-correct hidden rows (strict), L6 destroys
   2 of 4. Capability destruction is real evidence; the gain claims
   were not.
5. **Prior documents are marked provisional**, not silently rewritten —
   each carries a banner pointing here.

## Provenance

| artifact | contents |
|---|---|
| `results/campaign3b2/RESCORE_SUMMARY.json` | campaign arms + pooled stats |
| `results/campaign3b2/runs/*/RESCORE-STRICT.json` | per-run overlay: v1, v2, flipped rows, per-family |
| `scripts/analysis/rescore_strict.py` | the scorer, deterministic |
| `src-python/minagi/v161/evaluators.py` | `exact_match_v2`; `exact_match` retained for provenance only |
