# Campaign 3B.3 — scientific calibration and preservation

Status: **P0–P5 executed.** This campaign corrected the evidence
before believing it, then measured whether preservation mechanisms
change the verdict. They do not — under the strict oracle, no
adaptation condition beats frozen inference.

## P0 — corrected oracle and full rescore

`exact-match-v1` casefolded both sides; on a benchmark whose tasks ARE
case transformations that silently accepted wrong-case outputs.
`exact_match_strict` (exact-match-v2) is now bound into the campaign
runner, qualifier, sealed confirmation, colab runner, fit runner,
adversarial checks and the `strict_exact_match` diagnostic — the v1
function remains only for provenance, labelled as such.

Every retained prediction rescored (`CAMPAIGN3B_RESCORE_REPORT.md`):
~1,302 / 9,419 rows flip; frozen hidden 0.411 → 0.297 pooled; the
3B.1 "positive transfer" headline does not survive. All casefold-era
documents carry supersession banners rather than silent edits.

## P1 — uncertainty-aware intervention selection

The selector's states are machinery, not prose — each is a reachable
decision branch pinned by `tests-python/v161/test_intervention_states.py`:

- `E5_rank_or_target_modules` → **CONFOUNDED** (alpha=32 held fixed, so
  alpha/r covaried inversely with rank; the sweep cannot isolate
  capacity). Rank allocation, alpha/r sweeps, and dynamic LoRA remain
  open mechanisms — explicitly listed under `preserves`.
- `E2_replay_during_training` → **RECOMMENDED**, defined as
  rehearsal-in-gradients, explicitly distinct from L7's failed
  inference-time replay.
- `selection_complete` / `unmeasured_levers` / `completeness_warning`
  keep an evidence-poor checkout from pretending otherwise.

## P2 — rehearsal in the gradient stream

`rehearsal_fraction` interleaves *verified-correct* pairs into the
optimizer's microbatch schedule (every round(1/f)th microbatch), with
the same response-only loss as task pairs. The corpus is the frozen
model's own correct answers — verified at run time, bound into the
receipt by digest, per-source sample order recorded. Refuses to run
with an empty/unprovided pool.

## P3 — frozen-reference KL preservation

`kl_weight` adds mean KL(frozen ‖ adapted) over a preservation prompt
set every `kl_every`-th microbatch, computed against the same weights
with the adapter disabled. The receipt binds the reference parameter
digest — "the frozen reference" is a measured artifact, not a name.

## P5 — production confirmation attempt (honest NOT_QUALIFIED)

`scripts/validation/attempt_production_confirmation.py` ran the real
production path against the real campaign evidence on this host
(`P5_CONFIRMATION_ATTEMPT.json`):

- custody report: `independent_isolation_established: false` — uid 501
  owns both sealed data and research storage (machine-verified);
- confirmation refused at eligibility: the research record is REFUSED,
  so there is no qualified candidate to confirm.

P5 stays **open** until a surviving candidate meets an evaluator on a
distinct OS identity against a fresh sealed benchmark.

## P4 — matched multi-seed comparison: EXECUTED, no winner

`scripts/analysis/campaign3b3_multiseed.py` — four conditions
(fixed / rehearsal / KL / both) × seeds {0,1,2}, matched corpus
(pc-upper + pc-rev train), contract (16 max tokens), rank 16,
alpha 32, optimizer, 300 steps; strict oracle throughout. All 12
cells passed the 95% fit gate. Results:
`results/campaign3b3/MULTISEED_COMPARISON.json` (machine-produced).

| condition | hidden acc | gained/destroyed | net per seed | sign p | train s |
|---|---|---|---|---|---|
| fixed | 0.333 | 4/4 | [0, 0, 0] | 1.000 | 110 |
| rehearsal (P2) | 0.278 | 4/6 | [0, −1, −1] | 0.754 | 98 |
| KL (P3) | 0.333 | 5/5 | [+1, −1, 0] | 1.000 | 475 |
| both | 0.361 | 7/6 | [+1, −1, +1] | 1.000 | 457 |
| **frozen baseline** | **0.333** | — | — | — | 0 |
| replay L5b (ref) | 0.417 | +1/−0 in 12 | single seed | 1.000 | 0 |

**What the comparison establishes:**

1. **No adaptation condition beats frozen inference.** Fixed lands
   exactly on frozen (net 0 every seed); rehearsal is *worse* than
   frozen (0.278); KL averages net 0; rehearsal+KL nets +0.33 mean —
   2 of 3 seeds +1, sign-test p = 1.0.
2. **The recommended intervention underperformed.** E2 rehearsal
   (RECOMMENDED by the selector) *increased* destruction: −0.67 mean
   net. Gradient mass on verified-correct rows did not protect the
   held-out families at this scale — a genuinely informative negative.
3. **Preservation costs real compute for no measured benefit.**
   KL evaluation (batched, every 4th microbatch) multiplied train
   time ×4.3 (475s vs 110s) with zero net gain.
4. **Resolution is honest, not claimed.** Hidden = 12 rows × 3 seeds;
   small effects are underpowered by construction. The report carries
   magnitudes, and no condition is claimed positive — `p = 1.0` is
   what n=36-dependent-cells buys.

**Corrected verdict vs. the assessment's expectation:** the
hypothesis that the dominant problem is destructive weight updates
(which preservation should fix) is *not* supported at this scale —
rehearsal made destruction slightly worse and KL was neutral. The
problem is not merely drift; it is that at n=10 training examples the
adapter learns nothing that transfers to the hidden families under
the strict oracle, preservation or not.

## Environment hazard recorded

Sibling checkouts of this repository are importable via editable-install
finder hooks on this host. Scripts here anchor `sys.path` to their own
tree, but ad-hoc `python -c` from the wrong cwd can silently import a
sibling's `minagi` — observed during this campaign. Keep repo tooling
path-anchored; treat bare-interpreter imports of `minagi` as suspect
outside `src-python`.

The same leak reaches **IDE linting**: Pylint launched with the sibling
checkout as workspace root resolves `minagi` to that tree's stale
modules and reports `E0611`/`E1123` on 3B.3 symbols
(`exact_match_strict`, `rehearsal_fraction`, `kl_weight`,
`rehearsal_examples`, `kl_prompts`, `run_confirmation(mode=)`). These
are resolution artifacts, not code defects — the same pylint version
rates all six flagged files 10.00/10 from this worktree, ruff is clean,
and the P4 sweep executed with exactly those call signatures. The
worktree `pyproject.toml` init-hook repairs path resolution whenever
pylint loads this tree's config; the failure mode is a launch cwd in a
different checkout.
