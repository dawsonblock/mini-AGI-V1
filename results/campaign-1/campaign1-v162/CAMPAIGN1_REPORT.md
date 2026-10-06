# Campaign 1 — Six-Arm × Five-Seed Report (v16.2)

**Campaign:** `campaign1-v162` — executed on a fresh Colab T4 from
`main@ae937a927aee6e002bc0d74c907e66c3266dfbdf` (Commit A2). Plan was
created and Ed25519-signed *before* any model evaluation.

**Empirical question:** does neural adaptation produce repeatable
incremental forward transfer beyond L5 grounded replay?

## Result

| Arm | hidden EM (mean, 5 seeds) | FT vs L1 | retention | security regressions |
|-----|---------------------------|----------|-----------|----------------------|
| L1 frozen            | 0.000 | —     | 0.0 | 1.0  |
| L2 retrieval         | 0.000 | 0.000 | 0.0 | 3.0  |
| L3 semantic-memory   | 0.000 | 0.000 | 0.0 | 3.0  |
| L4 skills            | 0.000 | 0.000 | 0.0 | 5.0  |
| L5 grounded-replay   | 0.000 | 0.000 | 0.0 | 1.0  |
| L6 neural-adapter    | 0.200 | +0.20 | 0.0 | 1.2  |
| NC negative control  | 0.000 | 0.000 | 0.0 | 2.4  |

**Primary comparison — ΔFT_neural = mean(L6 − L5) = +0.200**
(per-seed: 0.125–0.292, σ = 0.068, n = 5)

**Negative control: clean.** NC showed exactly 0.0 forward transfer on
every seed — the L6 gain is attributable to the learning signal, not to
the training pipeline or LoRA mechanics alone.

**Decision: REFUSE / BLOCK** — independent artifact-only qualification
re-verified all 35 receipts, all 10 adapter digests, all arm-state
digests, and the signed plan, and reproduced the runner's decision.
The campaign fails two preregistered floors:

1. `minimum_retention: 0.95` — all arms score 0.0 on retention probes.
   Inspection shows this is a **scorer calibration artifact, not
   catastrophic forgetting**: the frozen base model (L1) also scores
   0.0 because it answers in prose ("The capital city of France is
   Paris.") while the scorer requires exact match ("Paris"). L6 causes
   no measurable degradation vs L1 — but the retention instrument did
   not measure retention as intended.
2. `require_zero_security_regressions` — 83 total flagged outputs
   across arms. The base model echoes literal `<policy-bypass>`-style
   markers when lured (L1 = 1.0/seed baseline). This is pre-existing
   non-compliance of the 0.5B instruct model, not an
   adaptation-induced regression; L6 adds no material increase
   (1.2/seed) vs frozen (1.0/seed).

## What was actually demonstrated

- Neural LoRA adaptation produced **repeatable** positive forward
  transfer (+0.20 hidden exact-match) on family-disjoint transform
  tasks — the adapter's hits concentrate on `hid-repeat`
  (`word-hyphen-word`), the family nearest the trained distribution.
- Non-parametric arms (retrieval, semantic memory, skills, grounded
  replay) produced **zero** forward transfer at this scale — exemplar
  injection alone did not let the 0.5B model execute unseen transforms.
- The full preregistered evidence chain held: signed plan → 35 signed
  receipts → digest-bound adapters/states → independent qualification.
- GPU nondeterminism policy held as designed: adapter digests are
  per-runtime unique; equivalence was evaluated statistically.

## Secondary finding

- **NC security regression (2.4/seed) exceeded L1 (1.0/seed)** — the
  corrupted-label adapter is measurably *less safe* than the frozen
  baseline (more literal-marker echo), while L6 (1.2/seed) ≈ L1.
  Corrupted supervision degraded safety without producing transfer.
- Post-review, `qualify_campaign1.py` was hardened (main@86af18c) to
  recompute the dataset partition from the committed corpus and check
  model/tokenizer digest consistency; re-qualification of this bundle
  reproduces REFUSE (`reproduction/REQUALIFICATION_V2.json`).

## Limitations (honest)

- Retention measurement needs a calibrated scorer (normalized/contain
  match or judge) before retention claims can be made either way.
- Security floor (zero echoes of literal markers) is arguably too
  strict for an untuned instruct model; both are artifacts of the
  preregistered thresholds doing their job — reported, not adjusted.
- 0.20 ΔFT on 24 hidden rows = ~5 extra correct answers/seed; small
  absolute effect. Meaningful as a mechanism demonstration, not as a
  capability claim.
- Single model (Qwen2.5-0.5B), single T4 class, one corpus.

## Not demonstrated

No claim of AGI, recursive self-improvement, or generalized continual
learning. Campaign 1 produced one reproducible incremental-transfer
signal under preregistered gates that were not met overall.
