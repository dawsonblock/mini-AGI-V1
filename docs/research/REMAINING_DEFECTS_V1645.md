# Remaining Defects — v16.4.5

Carried open from the v16.4.4 audit:

| ID | Defect | Target |
|---|---|---|
| SEC-005 | Controller accepts unsupported prerequisite evidence | v16.4.5 → carried (mechanism-controller scope, not release-blocking) |
| SEC-006 | Controller can recommend negative utility | carried |
| SEC-007 | Real-model learning and serving qualification incomplete | v16.4.6 — real PyTorch/PEFT/CUDA lifecycle, crash-injection matrix, GPU/Colab qualification |

Known platform limitations (documented in `THREAT_MODEL_V1645.md`):

- macOS does not enforce `RLIMIT_AS`; worker memory limits are
  requested but not guaranteed on this platform.
- No default filesystem sandbox on macOS; `sandbox-exec` only when an
  explicit profile is configured and tested.
- Credential env filtering is name-based; deployments should drop the
  entire inherited environment and whitelist.

Deferred by design:

- Campaign 3A learning-quality diagnosis (v16.5.0) — the evidence now
  ships separately and its REFUSED result is preserved for analysis.
- Controlled recursive-improvement experiments (v17.0.0).
