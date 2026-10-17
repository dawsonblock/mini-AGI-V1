# Remaining-defects register — v16.4.4

Defects known at the v16.4.4 release point, with severity, status, and
the reproducing test (or the reason none exists yet). All seven
v16.4.4 runtime findings (SEC-301..307) are closed by this release.
Controller correctness (SEC-005/006 + WP11) — the v16.4.5 gate — is
now CLOSED in this post-release change set; real-model learning
qualification (SEC-007) remains the v16.5.0 gate.

| ID | Severity | Defect | Status | Reproducing test |
|---|---|---|---|---|
| SEC-301 | critical | Model published to the router before its signed completion was durable | **CLOSED v16.4.4** — durable commit intent CAS precedes `publish_route`; `routing_observed` is honest post-traffic evidence | `test_activation_supervisor.py`, `test_v1644_runtime_closure.py` |
| SEC-302 | critical | Activation failure cleared the serving pointer instead of restoring the predecessor | **CLOSED v16.4.4** — `reconcile_deployment()` generation-checked CAS → RESTORED or UNAVAILABLE | `test_activation_supervisor.py` failure matrix |
| SEC-303 | critical | Global in-flight counter; `abort()` unloaded before routing stopped; drain timeout returned while requests ran | **CLOSED v16.4.4** — per-activation `RouteLeaseState`; retire→drain→cancel→safe_unload; timeout never unloads | `test_serving_router.py`, `test_v1644_runtime_closure.py::test_quarantine_does_not_unload_live_inference` |
| SEC-304 | high | Production startup did not supply/enforce backend manifest or policy epoch | **CLOSED v16.4.4** — `ProductionRuntimeConfig` + `--production` refuse without signed manifest, measured closure, epoch, fresh revocations, role keys | `test_supervised_service.py`, `service.py --production` path |
| SEC-305 | high | Administrative audit could silently fail | **CLOSED v16.4.4** — durable signed `admin_audit`; failure refuses admin changes; emergency stop excepted + activation blocked until reconcile | `test_v1644_runtime_closure.py` audit suite |
| SEC-306 | high | Unbounded inference params; no prompt token budget, deadline, or concurrency bound | **CLOSED v16.4.4** — `InferenceBudgetPolicyV1` enforced at router (concurrency/queue) and backend (tokens/deadline/cancel) | `test_v1644_runtime_closure.py` budget suite, `test_v1644_real_model_e2e.py` |
| SEC-307 | high | Cold-start recovery reconciled the pointer but never reloaded the model | **CLOSED v16.4.4** — `RecoveryManager`: fresh grant, re-measure, isolated load, probes, verified commit+publish, or UNAVAILABLE | `test_v1644_runtime_closure.py` recovery suite, e2e test |
| SEC-005 | high | Mechanism controller accepts unsupported prerequisite evidence (`prior_attempts` unverified digests) | **CLOSED (post-v16.4.4 change set)** — `SignedAttemptReceipt` carries the WP10 field set under an evaluator Ed25519 signature; `select_mechanism` requires one verified receipt per cheaper rung, bound to the same `evidence_digest`, with outcome ≠ INVALID, verified under a trusted evaluator verifier. Raw digests, unsigned receipts, untrusted signers, foreign-evidence receipts, and wrong-rung receipts are all rejected | `test_v1645_controller_gate.py` — fabricated-digest, unsigned, untrusted-signer, wrong-evidence, INVALID-outcome, wrong-rung, missing-rung, and no-verifier refusal cases |
| SEC-006 | high | Selector may pick highest-ranked intervention with negative expected utility; no `NO_CHANGE`/τ | **CLOSED (post-v16.4.4 change set)** — `NO_CHANGE` is selected whenever the best admissible utility fails to strictly exceed the frozen `ObjectiveWeights.utility_threshold` (τ=0 default); abstention is distinct from the DIAGNOSTIC_EXPERIMENT path, which remains for the nothing-clears-the-floor case | `test_v1645_controller_gate.py` — all-negative, τ-boundary, and abstain-vs-diagnostic cases |
| SEC-007 | medium | Real-model LEARNING qualification incomplete | **OPEN — v16.5.0 gate** — Campaign 3A **executed** 2026-10-10 (Colab T4, all 10 seeds, verified-resume across VM reclaims); runner aggregate **REFUSED** (delta_ft_neural +0.0022 vs gate ≥0.02; 6/10 positive vs ≥7) — evidence in `colab-evidence/campaign3a/`. Formal qualification still needs the evaluator-sealed holdout file + `qualification` key (both authority-side). 3B unsealed (fresh corpus pending), 3C not run | `scripts/validation/qualify_campaign1.py --holdout` (authority) |
| WP11 | medium | Experiment budgets (attempts, eval calls, wall-clock, GPU-seconds, memory, spend) were not defined or enforceable at the controller level | **CLOSED (post-v16.4.4 change set)** — `v161/experiment_budget.py`: frozen `ExperimentBudgetPolicy`, atomic `ExperimentBudgetLedger.charge` (a refused charge consumes nothing), signed `ExhaustionRecord` for breaches, `AttemptExecutor` enforcement path, and `select_mechanism` returns NO_CHANGE when the ledger is exhausted | `test_v1645_controller_gate.py` — exhausted-gate, atomic-charge, executor-refusal, measured-overrun cases |
| RUN-401 | medium | Backends execute in the supervisor process: a SIGKILL-worthy stalled model cannot be terminated without stopping the service. Cancellation is cooperative (`stopping_criteria`/`max_time`), not preemptive | **CLOSED (post-v16.4.4 change set)** — `runtime/worker_backend.py` runs each loaded model in its own interpreter process (`BackendSpec` + private pickle-frame channel); `terminate()` SIGKILLs the worker, pending requests fail fast with `WorkerDied`, and `ServingRouter.retire` escalates drain-timeout+cancel-grace to termination for backends exposing `terminate`. Production (`--production`) defaults to `process` isolation; `--backend-isolation` overrides. Watchdog expiry marks a worker stalled and fails subsequent requests fast | `test_worker_backend.py` — real spawned workers: SIGKILL mid-inference, wedge-terminate, cooperative cancel over the wire, supervisor quarantine reconcile |
| RUN-402 | low | In-flight lease drain relies on cooperative cancellation on timeout; a request that ignores cancellation retains the backend indefinitely (by design — resources are never freed under a live lease) | **CLOSED for process-isolated backends** — termination reclaims the process and the OS frees the model; the lease accounting still refuses to pretend an unreleased lease is gone (`test_terminate_frees_process_even_when_lease_lingers`). For `--backend-isolation inprocess` the interim accepted risk stands unchanged | `test_worker_backend.py`, `test_serving_router.py` drain-timeout tests |
| RUN-403 | low | `admin_audit` emergency exception writes no record while the store is broken (impossible by construction); the `audit_broken` block persisted in memory only — a restart during outage could serve before reconcile | **CLOSED (post-v16.4.4 change set)** — the flag is now a durable file (`audit_broken.json` next to the authority DB) written atomically when an op proceeds without its audit record; a restarted service rehydrates it and keeps `launch` refused until a reconciling audit lands, and that audit carries the recorded outage detail as evidence | `test_audit_outage_flag_survives_restart`, `test_emergency_quarantine_survives_audit_outage` |
| OPS-004 | medium | Role separation not enforced by code: all role keys co-located under one storage root in the dev scaffold | **PARTIAL (post-v16.4.4 change set)** — the scaffold now *supports* separated custody: `authority_bootstrap.py --role R --emit-record F` provisions only role R's key on the operator's own host and emits a public-only trust record for out-of-band trust-root assembly. The default all-roles path remains for the dev harness (documented) | `test_supervised_service.py`; manual: `--role` provisions one key |
| OPS-005 | low | Snapshot immutability is process-level; a same-identity adversary can chmod and rewrite (detected, not prevented) | **ACCEPTED RISK** — stage-time re-verification | `test_v1641_artifact_closure.py` |
| OPS-006 | low | `peer_uid()` was effectively Linux-only — the `getpeereid` branch checked a module attribute that never exists and indexed a scalar return, so macOS always returned `None` (fails closed) | **CLOSED (post-v16.4.4 change set)** — BSD/macOS resolve the real peer uid via `socket.getpeereid()` where exposed and `getpeereid(2)` through libc on Darwin; `SO_PEERCRED` remains the Linux path | `test_peer_uid_proves_local_identity` |
| OPS-007 | low | Event-log truncation detectable via local anchor only | **ACCEPTED RISK** — external anchor is future work | `test_journal_integrity.py` |
| ENV-001 | info | No GPU, CUDA, or native-QW3 qualification in this release — all evidence is CPU/offline | **OPEN — environment** | `VALIDATED_ENVIRONMENT.json`; ctest 28/28 CPU stub |

## Implemented vs qualified (required distinction)

* **Implemented AND tested here**: every SEC-301..307 repair, with
  deterministic fake backends plus a real tiny HF model through the
  real `PeftServingBackend` for the activate→query→rollback→restore
  loop. Post-release, worker-process backend isolation (RUN-401) is
  implemented and exercised with real spawned workers —
  `test_worker_backend.py` covers SIGKILL mid-inference, wedge
  termination on drain timeout, lingering-lease accounting, watchdog
  stall marking, cooperative cancel across the wire, and a full
  supervisor quarantine→terminate→reconcile cycle, and the same
  real tiny HF/PEFT GPT-2 serving through the worker boundary
  (load→infer→SIGKILL→quarantine) on CPU.
* **Implemented, not independently qualified**: `--production` startup
  path is code-complete and startup-refusals are unit-tested, but no
  production deployment was executed. Process isolation is the
  production default and is exercised on CPU with both the simulated
  backend and the real `PeftServingBackend`; it has NOT been exercised
  under a production deployment or on GPU hardware — those remain
  qualification steps, not claims.
* **Implemented AND tested here (post-release controller work)**:
  SEC-005/006 and WP11 are implemented with unit-level evidence in
  `test_v1645_controller_gate.py` — fabricated/unsigned/untrusted/
  foreign-evidence/INVALID/wrong-rung attempt rejection, NO_CHANGE
  abstention at and below τ, exhausted-budget refusal, atomic charge
  accounting, and signed exhaustion records. This is decision-layer
  evidence; no end-to-end campaign has yet run an attempt through the
  gate — that integration is part of Campaign 3 (SEC-007).
* **Not implemented (by plan)**: SEC-007 learning qualification
  (v16.5.0) — Campaigns 3A/3B/3C.
