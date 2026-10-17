# V4.1.0 — capability-acquisition development matrix report

Development-lane execution of `configs/capability_acquisition/intervention_matrix.yaml`
on the campaign4a corpus, under `development.yaml` (seeds {0,1}), on this host
(CPU, torch 2.10, peft 0.20, transformers 4.57; `Qwen/Qwen2.5-0.5B-Instruct` @
`7ae55760…`). Every cell ran through `scripts/positive_control_fit.py` under the
v4.0.3 run-reservation lane (atomic claims, ownership-bound staging, committed
RUN_COMMIT inventory) and independently re-verified with
`minagi.v161.evidence_reconciliation.verify_run` — **all 8 committed cells:
VERIFIED, zero mismatches**.

Evidence root: `results/v410_dev/` (gitignored; not source history).
Analysis artifact: `results/v410_dev/CONFIRMATION_ANALYSIS-v410-dev.json`.

## Executed cells (8) and screened cells (2)

| cell | status |
|---|---|
| fixed-s0, fixed-s1 | committed, VERIFIED |
| rehearsal-s0, rehearsal-s1 | committed, VERIFIED |
| kl-s0, kl-s1 | committed, VERIFIED |
| shuffled-s0, shuffled-s1 | committed, VERIFIED |
| both-s0, both-s1 | **screened — not executed** |

`both` (rehearsal 0.25 + KL 0.1) was screened out and reports `SCREENED` in the
analysis artifact: its rehearsal component alone fails the fit gate at fraction
0.25 on both seeds, and its KL component is separately shown non-protective and
~17× the compute. Running it could not change the per-arm verdicts. This is a
development-lane screening decision, recorded verbatim in the analysis artifact
(`--screened-arms`/`--screened-reason`, which are refused without a reason), not
an edit to the preregistered matrix — the confirmation lane must decide the arm
set explicitly, and a screened cell still counts as a missing evidence gap
(`complete: false`).

## Gate evaluation (gates from confirmation.yaml)

Security regressions below are **paired per-row net-new violations**: a
governed probe counts only if the adapted model violates it and the frozen
model passed it. The frozen baseline itself violates 5/30 governed probes —
absolute counts are reported alongside but are not the gate quantity.

| arm | fit all seeds | G5 within_holdout Δ (family-cluster 95% CI) | G6 retention max destruction | security net-new regressions (adapted abs.) | all gates |
|---|---|---|---|---|---|
| fixed | pass (0.988 / 0.975) | **+0.3125 [0.147, 0.475] PASS** | 0.156 **FAIL** | 4 (5, 7) **FAIL** | FAIL |
| kl | pass (0.963 / 1.000) | **+0.2949 [0.141, 0.445] PASS** | 0.219 **FAIL** | 5 (7, 6) **FAIL** | FAIL |
| rehearsal | **fail** (0.938 / 0.850) | +0.2363 [0.098, 0.377] | 0.625 **FAIL** | 4 (6, 6) **FAIL** | FAIL |
| shuffled (NC) | fail by design (0.0) | −0.078 [−0.188, 0.004] null | 0.844 | 0 (0, 0) degenerate | FAIL |
| both | not executed | — | — | — | SCREENED |

## Findings

1. **Acquisition replicates.** Within-family generalization on the
   *independent* `within_holdout` surface (256 rows, inputs disjoint from train
   and validation) is +0.20…+0.32 across all three parametric arms on both
   seeds; the family-clustered bootstrap CI for fixed and kl excludes 0.
   The earlier campaign3e result was not a single-seed artifact.

2. **No declared arm preserves retention reliably.** Per-seed destruction on
   the 108-row retention surface scatters wildly:

   | arm | s0 | s1 |
   |---|---|---|
   | fixed | 3/32 (0.094) | 5/32 (0.156) |
   | rehearsal | 0/32 (0.000) | 20/32 (0.625) |
   | kl | 7/32 (0.219) | 3/32 (0.094) |
   | shuffled | 27/32 (0.844) | 12/32 (0.375) |

   Retention outcome is a seed-level lottery under every declared mechanism —
   no arm keeps all seeds ≤ 0.10, and the scatter means single-seed results
   (in either direction) are not evidence of a working mechanism.

3. **The rehearsal pool is starved by design.** The default verified pool is
   built by frozen-model strict-oracle regeneration over non-trained-family
   rows; on campaign4a only **2 of 80 reh4 rows survive** (78 rejected) because
   reh4 is transform-family text the frozen 0.5B model cannot already do.
   Rehearsing two arbitrary pairs 160× is not capability preservation; the s0
   zero-destruction result was luck, not mechanism.

4. **KL preservation did not protect the retention surface** and costs ~17×
   the compute (train_seconds ~5,600–5,700 vs ~335–500): at weight 0.1 on 12
   generic preservation prompts it regularizes those prompts' continuations,
   not the probe distribution retention measures — s0 destruction was *worse
   than the fixed baseline* (0.219 vs 0.094).

5. **Negative control is clean.** Shuffled-label training is unlearnable
   (fit 0.0, within_holdout Δ ≈ −0.08 on both seeds) yet still destroys
   37–84% of frozen-correct retention answers — gradient noise alone damages
   the surface; destruction is not specific to real acquisition.

6. **Hidden transfer remains absent** (adapted hidden ≈ 0.01–0.02, no arm
   acquires cross-family behavior) — acquisition is within-family only.

7. **Security gate is a paired regression bound — resolved.** The declared
   `security_max_violations: 0` was ambiguous between an absolute adapted
   count (which the frozen baseline itself fails at 5/30 governed probes)
   and the campaign convention of a frozen-relative regression bound. It is
   now implemented and documented as the latter: the analyzer derives
   net-new violations per row from the sidecar's stored frozen+adapted
   outputs (never trusting a lane rollup field), and unevaluated/unpaired
   evidence fails closed. Under the corrected reading every parametric arm
   still fails (fixed 4, kl 5, rehearsal 4 net-new); notably fixed-s0 shows
   the case pairing catches — adapted and frozen violate the *same count*
   (5) but on *different rows* (1 regressed, 1 repaired). The shuffled
   control's 0 is degenerate (it emits no marker-shaped output at all).

## Standing vs the v4.1.0 claim

- **G5 (generalization): demonstrated at dev scale** — real acquisition on
  unseen same-family inputs, independently recomputed, replicating on 2 seeds.
- **G6 (retention): NOT demonstrated** — every declared arm fails the
  preregistered bound on at least one seed; the preservation mechanism is
  unproven.
- **Qualification: all runs EVALUATION_VERIFIED / NOT_QUALIFIED** — derived
  strictly from evidence (`qualified_outcome.derive_qualified_outcome`), no
  independent evaluator custody exists on this host, and none is claimed.
- The confirmatory lane (5 seeds, sealed final_holdout, evaluator custody)
  is **not ready to run**: it would record failures, not decide a live
  question. A retention-preserving intervention must first survive the dev
  lane.

## Follow-up probes executed in this lane (post-matrix dev iteration)

The starved-pool finding (item 3) motivated a concrete test: build a rehearsal
corpus from capabilities the frozen model verifiably *has*. A pool file of 136
campaign3 train/validation/retention rows (all families outside tr3-*, none on
campaign4a measurement surfaces) was written to
`results/v410_dev/rehearsal_pool_c3.jsonl`; frozen-correct regeneration admits
**27 verified pairs** (109 rejected) — 13.5× the declared corpus's pool.

Three candidate arms ran the same verified pipeline (2 seeds each):

| candidate config | seed | fit | who Δ | retention destruction |
|---|---|---|---|---|
| rehearsal 0.25, pool=27 | s0 | 0.900 FAIL | +0.242 | 1/32 (0.031) |
| rehearsal 0.25, pool=27 | s1 | 0.938 FAIL | +0.223 | 2/32 (0.063) |
| rehearsal 0.10, pool=27 | s0 | 0.988 PASS | +0.262 | 1/32 (0.031) |
| rehearsal 0.10, pool=27 | s1 | 0.912 FAIL | +0.301 | 4/32 (0.125) |
| rehearsal 0.10, pool=27, steps 720 | s0 | 0.988 PASS | +0.246 | 1/32 (0.031) |
| rehearsal 0.10, pool=27, steps 720 | s1 | 0.963 PASS | +0.301 | 6/32 (0.188) |

(all six committed runs independently VERIFIED, zero mismatches.)

### What the probes establish

- **Pool size controls retention, replicated:** on seed 1, the same
  intervention at the same fraction destroys 20/32 with a 2-pair pool vs
  2/32 with a 27-pair pool. Verified-pair rehearsal is a real mechanism —
  when the pool contains capabilities the model actually has.
- **Fit margin is a budget problem, fixable:** steps 720 at fraction 0.10
  restores task presentations to ~648 (≈ the fixed arm's 640) and passes the
  fit gate on both seeds (0.988 / 0.963).
- **Retention is still not inside the gate on seed 1.** Every configuration
  scatters around the 0.10 bound on that seed (0.063–0.188); destruction
  counts of 2–6/32 are within noise of each other, and nothing yet separates
  "protected" from "lucky".

### Dev-lane verdict

A candidate arm *shape* now exists — compensated-budget rehearsal over an
enlarged frozen-correct pool — that beats every arm in the declared matrix and
clears all gates on seed 0. It does **not** yet clear them on seed 1, so it is
not a confirmation-ready arm. The honest next dev iterations are a bigger and
more *diverse* verified pool (hundreds of pairs, not tens), pool composition
matched to the retention surface's task types, or a lower rehearsal fraction
with the same budget compensation. The confirmatory lane remains unrun and
unearned.

## Candidate dev iterations (remaining)

- Rehearsal pool at 100+ verified pairs across heterogeneous frozen-correct
  task types (QA, format, lexical) — the pool-size result suggests coverage,
  not count alone, is the binding constraint.
- rehearsal_fraction 0.05 with steps ≥ 720 — less diversion, same budget.
- Weight-space interpolation toward frozen post-training (model averaging)
  evaluated under identical gates.
- Lower learning rate / lower rank to probe the acquisition–forgetting
  tradeoff boundary.

None of these is a claim; each is a proposed dev-lane arm whose evidence must
pass through the same verified pipeline.
