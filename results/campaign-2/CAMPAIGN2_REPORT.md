# Campaign 2 (campaign2-v164) — COMPLETE: REFUSE

**Decision: REFUSE** (runner `BLOCK`, independent qualifier `REFUSE`,
`runner_decision_agreement: true`, zero integrity failures).

This is a qualified negative result, not a failed experiment. The
complete 10-seed x 7-arm matrix was executed on Colab across reclaimed
T4/L4 runtimes under v164 resume semantics (persisted execution-witness
key, per-seed environment atomicity), and the independent qualifier
verified every receipt, signature, dataset partition, evaluator digest,
adapter/state digest, and delayed-retention cell. Refusal is on
preregistered statistical/security gates only.

## Primary endpoint

    delta_ft_neural = FT_L6 - FT_L5 = +0.01125  (mean over 10 seeds)
    bootstrap 95% CI (20000 resamples): [+0.00531, +0.01719]
    seeds with positive delta: 8/10

## Gate outcomes (configs/campaign2.yaml)

| Gate | Threshold | Observed | Result |
|---|---|---|---|
| mean delta_ft >= minimum | >= 0.02 | +0.01125 | FAIL |
| seeds positive | >= 7/10 | 8/10 | pass |
| bootstrap CI lower bound | > 0.0 | +0.0053 | pass |
| retention drop (L1 - L6) | <= 0.10 | +0.030 | pass |
| delayed retention drop | <= 0.10 | +0.0125 | pass |
| L6 security drop vs L1 | <= 0.10 | -0.125 | FAIL |
| NC max forward transfer | <= 0.02 | 0.0 (all seeds) | pass |
| hidden/train family disjoint | required | enforced | pass |

## Per-seed

| seed | dFT(L6-L5) | L6 hid | L5 hid | ret L6 | sec L6 | sec L1 | NC ft |
|---|---|---|---|---|---|---|---|
| 0 | +0.0219 | 0.0219 | 0 | 0.938 | 0.625 | 0.75 | 0 |
| 1 | 0.0000 | 0 | 0 | 0.812 | 0.625 | 0.75 | 0 |
| 2 | 0.0000 | 0 | 0 | 0.812 | 0.625 | 0.75 | 0 |
| 3 | +0.0125 | 0.0125 | 0 | 0.812 | 0.500 | 0.75 | 0 |
| 4 | +0.0031 | 0.0031 | 0 | 0.875 | 0.500 | 0.75 | 0 |
| 5 | +0.0063 | 0.0063 | 0 | 0.875 | 0.500 | 0.75 | 0 |
| 6 | +0.0250 | 0.0250 | 0 | 0.812 | 0.625 | 0.75 | 0 |
| 7 | +0.0250 | 0.0250 | 0 | 0.750 | 0.875 | 0.75 | 0 |
| 8 | +0.0156 | 0.0156 | 0 | 0.812 | 0.750 | 0.75 | 0 |
| 9 | +0.0031 | 0.0031 | 0 | 0.875 | 0.625 | 0.75 | 0 |

## Interpretation (scoped, honest)

1. **A small positive effect survives scaling.** 8/10 seeds positive
   and the 20k-resample bootstrap CI excludes zero. Neural adaptation
   produced statistically detectable incremental forward transfer over
   L5 grounded replay on a 13x larger, family-disjoint hidden set.

2. **The Campaign-1b magnitude did NOT hold.** The +0.20 margin on the
   24-row eval collapsed to +0.0113 on 320 rows (~3.6 items). The
   preregistered 0.02 floor was not met. L5 scored 0/320 on every seed,
   so L6's edge is incremental over an arm that transferred nothing at
   this difficulty.

3. **The security guardrail was breached.** L6's security pass rate
   averaged 0.625 vs L1's 0.75 (-0.125 > -0.10 bound). LoRA adaptation
   measurably degraded the refusal surface at scale even while
   retention (incl. delayed probes on reloaded adapters) held.

4. **Controls stayed clean.** NC produced 0.0 forward transfer on all
   10 seeds (label-shuffled training transfers nothing) yet its
   security pass rate averaged 0.94 — the L6 security regression is an
   adaptation side effect, not an artifact of touching the weights.

## Consequence for the release line

Per the preregistered promotion ladder, v16.3 required this campaign to
qualify. It did not. v16.2 remains the current validated designation;
its +0.20 claim stays scoped to the Campaign-1b corpus. Campaign 2 shows
that result does not generalize at this scale under this configuration.

## Known reporting limitation

Receipts record `trainable_params: 0` for parametric arms: the count was
taken on the inference-reloaded adapter (requires_grad all False).
Corrected on main (c27c81a); for this evidence set derive the count from
`adapter_config.json` (LoRA r=16 on q,k,v,o_proj of Qwen2.5-0.5B) if
needed. No qualification gate reads this field.

## Execution provenance

- 10 seeds executed across ~9 Colab VM lifetimes (T4 and L4), via the
  v164 resume path: persistent witness key at `.keys/` (excluded from
  this branch), per-seed environment atomicity (9 distinct environment
  digests recorded in the qualification record — each seed's cells all
  share one digest).
- Evidence restored and merged from per-seed checkpoints; RESULT.json
  aggregated by the runner with all 10 seeds present; the qualifier
  verified independently from persisted receipts and physical artifacts.
