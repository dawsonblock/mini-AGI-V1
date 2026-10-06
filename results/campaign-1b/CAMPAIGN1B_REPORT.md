# Campaign 1b Report — `campaign1b-v163`

**Decision: QUALIFIED** (runner `PASS`, independent qualifier `QUALIFIED`, agreement true)

Campaign 1b is a NEW preregistered experiment. Campaign 1 (`campaign1-v162`)
remains untouched as historical evidence — its REFUSE under the hardened
qualifier stands.

## Preregistered plan (signed before any evaluation)

| Field | Value |
|---|---|
| plan digest | `sha256:aa0b028a5821d4f9d6878b04018af9ab390b8c11b196b5aee26439ab7de471c7` |
| schema | `mini-agi-v16.3-colab-campaign-plan-v1` |
| model | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` (immutable commit) |
| model_digest | `sha256:2dd8a7c8df74e1f60214556116edb81c62d5d4c075bf17ab67f79438cffc4f32` |
| tokenizer_digest | `sha256:c7db022fae80ee836c03b3bddcfd69c4523b4966bc76002c3aa1404a6fdebf31` |
| generation_template_digest | `sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| dataset_partition_digest | `sha256:d5fab38ed65ecb9bf7f19380975c9921ea6dc8e00357febb6c8ccf8024c32dc7` |
| scorer_artifact_digest | `sha256:babec1c1796cfa7cf6574fe8fe484f1f4804784258aace404b81e467553220e2` (containment_match, calibrated) |
| retention_artifact_digest | `sha256:3629619ea59ea1936fce34c5cb2678bcca7dc23b00f4a9b18c21a7af22859b46` (binds calibration record) |
| security_artifact_digest | `sha256:3ac68eb5e95dd5581cac483237661f5f326381d6f4ea43c77baa04e843171ffe` |
| seeds | [0, 1, 2, 3, 4] |
| arms | ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'NC'] |

### Frozen thresholds

- `minimum_neural_incremental_ft` = 0.02 (primary endpoint ΔFT_L6 = FT_L6 − FT_L5)
- `min_seeds_positive_ft` = 4 (per-seed confidence criterion)
- `negative_control_max_ft` = 0.02
- `retention_max_drop` = 0.1 (L6 retention ≥ L1 − 0.1)
- `security_min_pass_rate` = 0.5 (absolute floor)
- `security_max_drop_vs_L1` = 0.1 (relative guardrail)
- reproduction semantics: `statistical-equivalence`, adapter bitwise required: False, tolerance 0.05

## Results (Runtime A, Colab T4, source `d0d8cb2`)

| Arm | hidden EM | retention (containment) | security pass-rate |
|---|---|---|---|
| L1 frozen | 0.000 | 0.833 | 0.833 |
| L2 retrieval | 0.000 | 0.833 | 0.500 |
| L3 semantic-memory | 0.000 | 0.833 | 0.500 |
| L4 skills | 0.000 | 0.750 | 0.167 |
| L5 grounded-replay | 0.000 | 0.833 | 0.833 |
| **L6 neural adapter** | **0.200** | 0.867 | 0.800 |
| NC negative control | 0.000 | 0.717 | 0.600 |

### Primary endpoint

- **ΔFT_neural (L6 − L5) = +0.200** — mean across seeds
- per-seed ΔFT: [0.167, 0.292, 0.167, 0.125, 0.25]
- positive seeds: **5/5** (required ≥ 4)
- NC forward transfer: 0.0 (violation: False)

### Hard constraints — all satisfied

- ΔFT_L6 = 0.20 > 0.02 ✓
- retention: L6 0.867 ≥ L1 0.833 − 0.1 = 0.733 ✓ (no measurable forgetting;
  calibrated containment scorer grades prose answers correctly — Campaign 1's
  0.0 was a scorer artifact)
- security: L6 pass-rate 0.80 ≥ floor 0.5 ✓;
  drop vs L1 = 0.033 ≤ 0.1 ✓
- NC: ft 0.0 ≤ 0.02 ✓ — corrupted supervision still
  produces no transfer, isolating L6's gain as learning-signal-driven

## Interpretation

ΔFT_neural reproduced at +0.20 — identical to Campaign 1's point estimate, now
under a stronger evidence standard: calibrated scorer, immutable model/tokenizer/
template identities, symmetric receipts, dual security rule, per-seed confidence
criterion (5/5 positive vs ≥4 required).

Residual caveats:
- absolute scale is small (0.20 EM on a 24-row hidden set ≈ 5 items; the
  confidence criterion is seed-count-based, not a full CI)
- L4 skills show poor security pass-rate (0.167) — measured, not gated; the
  gates apply to the L6 candidate
- NC security (0.60) still trails L1 (0.833) — corrupted supervision carries a
  safety cost even when it buys no transfer
- hidden EM uses exact match on transform outputs (appropriate there);
  retention uses the calibrated containment scorer

PASS is experimental qualification only; not production promotion authority.
