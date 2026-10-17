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

`both` (rehearsal 0.25 + KL 0.1) was screened out and is honestly MISSING in the
analysis artifact: its rehearsal component alone fails the fit gate at fraction
0.25 on both seeds, and its KL component is separately shown non-protective and
~17× the compute. Running it could not change the per-arm verdicts. This is a
development-lane screening decision, recorded here, not an edit to the
preregistered matrix — the confirmation lane must decide the arm set explicitly.

## Gate evaluation (gates from confirmation.yaml)

| arm | fit all seeds | G5 within_holdout Δ (family-cluster 95% CI) | G6 retention max destruction | security regressions | all gates |
|---|---|---|---|---|---|
| fixed | pass (0.988 / 0.975) | **+0.3125 [0.147, 0.475] PASS** | 0.156 **FAIL** | 12 | FAIL |
| kl | pass (0.963 / 1.000) | **+0.2949 [0.141, 0.445] PASS** | 0.219 **FAIL** | 13 | FAIL |
| rehearsal | **fail** (0.938 / 0.850) | +0.2363 [0.098, 0.377] | 0.625 **FAIL** | 12 | FAIL |
| shuffled (NC) | fail by design (0.0) | −0.078 [−0.188, 0.004] null | 0.844 | 0 (degenerate) | FAIL |
| both | not executed | — | — | — | MISSING |

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

7. **Security-gate tension to resolve before confirmation:** the declared
   `security_max_violations: 0` reads as an absolute in the analyzer, but the
   frozen baseline itself already violates 5/33 — as written the gate is
   unreachable by any model including the base. Adapted-vs-frozen net
   regressions were +0…+2 per arm. The gate needs a stated interpretation
   (absolute vs net-new) before the confirmatory protocol freezes.

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

## Candidate dev iterations (not yet run — require a new dev sweep)

- Rehearsal over a pool of capabilities the frozen model verifiably *has*
  (e.g., general-QA/format pairs passing frozen-correct regeneration at high
  rate), not transform families it lacks; measure pool size vs destruction.
- Lower rehearsal fraction (0.05–0.10) to recover fit margin — only worth
  testing once the pool is big enough to matter.
- Lower exposure/LR to find whether a milder adaptation exists inside the
  acquisition window before the forgetting regime.
- Post-adaptation repair (e.g., weight-space interpolation toward frozen)
  evaluated with the same gates.

None of these is a claim; each is a proposed dev-lane arm whose evidence must
pass through the same verified pipeline.
