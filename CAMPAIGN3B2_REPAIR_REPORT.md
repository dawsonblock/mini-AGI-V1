# Campaign 3B.2 — repair report

Status: **executed.** This closes the defects the 3B.1/3B.2 sequence was
launched to fix and records what the repaired pipeline measured.

## What was broken and what fixed it

| defect | fix | evidence |
|---|---|---|
| Runner wrote every adapter to a shared `<output>.parent/adapter`, destroying five of six adapters (SCI-402) | immutable per-run directories: staging → hash → reload-verify → atomic publish; failure recorded as `FAILED`, never dropped | `scripts/positive_control_fit.py` `fail()`/publish path; run dirs each carry `ADAPTER_MANIFEST.json` + `RUN_COMMIT.json` |
| Token budget accepted from a flag the contract did not authenticate (SCI-403) | `contract.output.max_new_tokens` is the only source; effective decoding recorded per run | `output_contract_from_config`, `effective_decoding` block in `METRICS.json` |
| Case-insensitive answer scoring (RD-18, found by external assessment) | `exact-match-v2` strict oracle; every retained prediction rescored | `CAMPAIGN3B_RESCORE_REPORT.md`, 80 run overlays |
| Selector could silently stop refusing on missing evidence | `selection_complete` / `unmeasured_levers` reported explicitly | `scripts/analysis/intervention_selection.py` |
| Custody report equated mode-0600 with independent isolation (RD-19) | distinct storage/evaluator owner required; production fails closed | `run_sealed_confirmation.py`, custody tests |

## Rank-ladder reconciliation

The ladder varied rank 4–32 at fixed `alpha=32`, so `alpha/r` (8.0→1.0)
covaried inversely with rank. The non-monotone ordering that produced a
"rank refuted" claim is therefore a **confounded** measurement —
reclassified `CONFOUNDED / low priority`, and the selector now
distinguishes that state from `REFUTED` (see RD-13).

## Verified bounds that survived

- `alpha/r` confound identified and carried into the selector's
  evidence model.
- Composition remains the largest measured lever under strict scoring.
- Destruction (adapted breaking frozen-correct rows) replicates across
  seeds and survives the strict rescore — L7 destroys 4/4.
- The fit gate is real: matched runs reach 1.000 subset fit, so the
  pipeline can produce a fitted adapter on demand.

## What this campaign did NOT establish

- Any positive transfer beyond one net row under strict scoring.
- That rank, rank allocation, or scaling helps — evidence inconclusive.
- Any qualified candidate: nothing here survives promotion review.
