# Campaign 3A (campaign3a-v166) — results

- decision: **REFUSED** (independent qualification; experimental qualification only, never production promotion)
- qualified_at: 2026-10-13T01:30:47.682324+00:00
- seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- qualification record: `QUALIFICATION_RECORD.json` (digest sha256:8049f70d0e7231dd8f27d009c18c9b827cc2b39ef4bb4f99773833417fc0f90b)
- runner agreement: True

## Gate reasons

- mean neural incremental FT (L6-L5) below preregistered minimum
- only 6/10 seeds positive < 7
- family-clustered CI lower bound 0.0000 <= 0.0

## Reconstructed metrics

```json
{
  "best": 0.00625,
  "bootstrap_ci": {
    "alpha": 0.05,
    "lower": 0.0009375000000000002,
    "mean": 0.0021875,
    "n": 10,
    "resamples": 20000,
    "rng_seed": 20241006,
    "upper": 0.0034375000000000005
  },
  "cluster_bootstrap_ci_row_delta": {
    "alpha": 0.05,
    "lower": 0.0,
    "mean": 0.0021875,
    "n": 3200,
    "n_clusters": 8,
    "resamples": 20000,
    "rng_seed": 20241006,
    "upper": 0.0065625
  },
  "conditional_regression_rate_mean": 0.0,
  "false_activation_rate_mean": 0.0,
  "mean": 0.0021875,
  "median": 0.003125,
  "n": 10,
  "retention_delayed": {
    "L1": 0.8125,
    "L6": 0.8125,
    "drop": 0.0
  },
  "stdev": 0.002109214242845478,
  "worst": 0.0
}
```

## Evidence inventory

- campaign evidence + signed receipts: `campaigns/campaign3a-v166/`
- adapters: `adapters/campaign3a-v166/`
- arm states: `arms/campaign3a-v166/`
- signed evaluation bundle: `campaigns/campaign3a-v166/EVALUATION_BUNDLE.json`
- authority ledger: `AUTHORITY_LEDGER.jsonl`
- trust root (public identities): `trust_root.json`

## Provenance notes

- The worker executed all 10 preregistered seeds under the final key generation (plan `sha256:c7570e59…`, witness `sha256:325e8917…`). The final-session download was torn by a repack race; the local tree was reconstructed from the seed archives and salvage.
- `arms/campaign3a-v166/seed-9/L5` was restored from the byte-identical attested state (the L5 replay buffer is identical across all ten seeds; the restored tree re-derives to the receipt's exact `state_digest` `sha256:e1709e61…`).
- The qualification-time trust root was assembled per the separated-custody flow: the worker's plan+execution_witness records plus the evaluation+qualification records held off-worker by the evaluator.

Private keys are deliberately not published; the sealed holdout file stays with the evaluation authority.
