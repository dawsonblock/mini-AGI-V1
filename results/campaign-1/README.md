# results/v16.1-colab-campaign-1 — v16.1 Colab Validation Evidence

Empirical evidence produced by executing the v16.1 release on a clean Google
Colab T4 runtime (2026-10-06), driven headlessly via `google-colab-cli`.

- **Source commit:** see `SOURCE_COMMIT.txt` — the exact `main` commit this
  evidence was produced from. This branch descends directly from it.
- **Scope:** smoke/plumbing validation. These results prove the
  execution → evidence → qualification path works on real hardware with real
  gradient updates. They do **not** demonstrate useful learning, forward
  transfer, or recursive self-improvement.

## Layout

- `SOURCE_COMMIT.txt` — source SHA this evidence descends from
- `CAMPAIGN_PLAN.json` — sealed Qwen smoke campaign plan (digests bound)
- `READINESS_REPORT.json` — Colab doctor output (GPU/CUDA/torch/RAM/disk)
- `EXECUTION_MATRIX_CERTIFICATE.json` — seed×arm completeness + receipt digests
- `environment/COLAB_ENVIRONMENT.json` — probed runtime environment
- `manifests/` — dataset membership manifests (train/validation/hidden/retention)
- `runs/<campaign>/` — plans, dataset proofs, RESULT.json, execution public key,
  and signed per-seed `A0.json`/`A1.json` run receipts
- `adapters/<campaign>/seed-0/` — the actual serialized LoRA adapters
  (`adapter_model.safetensors`, 2.1 MB for Qwen) plus per-dir digest manifests
- `evaluation/` — `EVALUATION_BUNDLE.json`, `metrics.json`
- `qualification/` — independent artifact-only `QUALIFICATION_RECORD.json` per
  campaign (verifier consumes files only, never live objects)
- `security/` — adversarial rejection results + interruption-recovery audit
- `reproduction/REPRODUCTION_RECORD.json` — `status: NOT_RUN` (Phase 18 pending)
- `gates/` — per-phase gate artifacts (release verify, tests, import probe,
  smoke results, adapter-effect report)
- `reports/CAMPAIGN_REPORT.md` — narrative report with actual numbers
