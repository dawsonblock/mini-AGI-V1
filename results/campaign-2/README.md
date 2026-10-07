# Campaign 2 (campaign2-v164) — COMPLETE: REFUSE

Complete 10-seed x 7-arm matrix. Decision: **REFUSE** — see
`CAMPAIGN2_REPORT.md` for the gate-by-gate record and interpretation.

- Runner: `BLOCK` (`campaigns/campaign2-v164/RESULT.json`, n=10)
- Independent qualifier: `REFUSE`
  (`campaigns/campaign2-v164/QUALIFICATION_RECORD.json`),
  `runner_decision_agreement: true`, zero integrity failures.
- Refusal grounds (preregistered): mean delta_ft_neural +0.01125 < 0.02
  floor; L6 security drop -0.125 vs L1 breaches the -0.10 bound.
- Passed: 8/10 seeds positive, bootstrap CI lower bound > 0,
  retention/delayed-retention drops within bounds, NC = 0.0 everywhere.

Note: the execution-witness PRIVATE key lives under the runtime
storage root `.keys/` and is deliberately excluded from this branch
(gitignored). Only the public key + signed receipts are evidence.
