# Remaining Defects — v16.4.5

All v16.4.5 release gates are closed. SEC-005/006, carried as open in
the v16.4.4 audit mapping, are **closed**: implemented in the
post-v16.4.4 change set (`plasticity.py`, `experiment_budget.py`),
covered by 19 gate tests in `test_v1645_controller_gate.py`, and
manifested/qualified for the first time under this release.

| ID | Defect | Status |
|---|---|---|
| SEC-005 | Controller accepts unsupported prerequisite evidence | **CLOSED** — verified signed attempt receipts per cheaper rung |
| SEC-006 | Controller can recommend negative utility | **CLOSED** — `NO_CHANGE` abstention at/below τ; exhausted-budget refusal |
| SEC-007 | Real-model learning and serving qualification incomplete | **OPEN** — v16.4.6 target. Implemented and exercised on CPU/macOS through `PeftServingBackend` (load→infer→SIGKILL→quarantine e2e); GPU/Colab qualification and the full crash-injection matrix remain platform gates, not code defects |

Locally covered portions of the SEC-007 crash matrix: worker
termination mid-inference, partial/hostile IPC frames, supervisor
restart (cold restoration), revocation updates during preparation,
staged-artifact mutation, sqlite unavailability/corruption.
Outstanding: GPU/Colab real-model runs and production deployment
qualification.

Known platform limitations (documented in `THREAT_MODEL_V1645.md`):

- macOS does not enforce `RLIMIT_AS`; worker memory limits are
  requested but not guaranteed on this platform.
- No default filesystem sandbox on macOS; `sandbox-exec` only when an
  explicit profile is configured and tested.
- Credential env filtering is name-based; deployments should drop the
  entire inherited environment and whitelist.
- `fork` is not a supported multiprocessing start method for
  sqlite-using children on macOS (system libdispatch state).

Deferred by design:

- Campaign 3A learning-quality diagnosis (v16.5.0) — evidence now
  ships separately; REFUSED result preserved.
- Controlled recursive-improvement experiments (v17.0.0).
