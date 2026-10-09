# Remaining-defects register — v16.4.4

Defects known at the v16.4.4 release point, with severity, status, and
the reproducing test (or the reason none exists yet). All seven
v16.4.4 runtime findings (SEC-301..307) are closed by this release.
Controller correctness (SEC-005/006) is the v16.4.5 gate; real-model
learning qualification (SEC-007) is the v16.5.0 gate.

| ID | Severity | Defect | Status | Reproducing test |
|---|---|---|---|---|
| SEC-301 | critical | Model published to the router before its signed completion was durable | **CLOSED v16.4.4** — durable commit intent CAS precedes `publish_route`; `routing_observed` is honest post-traffic evidence | `test_activation_supervisor.py`, `test_v1644_runtime_closure.py` |
| SEC-302 | critical | Activation failure cleared the serving pointer instead of restoring the predecessor | **CLOSED v16.4.4** — `reconcile_deployment()` generation-checked CAS → RESTORED or UNAVAILABLE | `test_activation_supervisor.py` failure matrix |
| SEC-303 | critical | Global in-flight counter; `abort()` unloaded before routing stopped; drain timeout returned while requests ran | **CLOSED v16.4.4** — per-activation `RouteLeaseState`; retire→drain→cancel→safe_unload; timeout never unloads | `test_serving_router.py`, `test_v1644_runtime_closure.py::test_quarantine_does_not_unload_live_inference` |
| SEC-304 | high | Production startup did not supply/enforce backend manifest or policy epoch | **CLOSED v16.4.4** — `ProductionRuntimeConfig` + `--production` refuse without signed manifest, measured closure, epoch, fresh revocations, role keys | `test_supervised_service.py`, `service.py --production` path |
| SEC-305 | high | Administrative audit could silently fail | **CLOSED v16.4.4** — durable signed `admin_audit`; failure refuses admin changes; emergency stop excepted + activation blocked until reconcile | `test_v1644_runtime_closure.py` audit suite |
| SEC-306 | high | Unbounded inference params; no prompt token budget, deadline, or concurrency bound | **CLOSED v16.4.4** — `InferenceBudgetPolicyV1` enforced at router (concurrency/queue) and backend (tokens/deadline/cancel) | `test_v1644_runtime_closure.py` budget suite, `test_v1644_real_model_e2e.py` |
| SEC-307 | high | Cold-start recovery reconciled the pointer but never reloaded the model | **CLOSED v16.4.4** — `RecoveryManager`: fresh grant, re-measure, isolated load, probes, verified commit+publish, or UNAVAILABLE | `test_v1644_runtime_closure.py` recovery suite, e2e test |
| SEC-005 | high | Mechanism controller accepts unsupported prerequisite evidence (`prior_attempts` unverified digests) | **OPEN — v16.4.5 gate** | `tests-python/v161/test_v166_mechanism_controller.py` documents current acceptance |
| SEC-006 | high | Selector may pick highest-ranked intervention with negative expected utility; no `NO_CHANGE`/τ | **OPEN — v16.4.5 gate** | `test_v166_mechanism_controller.py` (utility arithmetic) |
| SEC-007 | medium | Real-model LEARNING qualification incomplete (Campaign 3A executing, 3B unsealed, 3C not run) | **OPEN — v16.5.0 gate** — note: real-model *serving* lifecycle is now exercised by `test_v1644_real_model_e2e.py` (tiny real GPT-2, real backend); the open item is qualified-learning evidence, not serving mechanics | Campaign 3 execution |
| RUN-401 | medium | Backends execute in the supervisor process: a SIGKILL-worthy stalled model cannot be terminated without stopping the service. Cancellation is cooperative (`stopping_criteria`/`max_time`), not preemptive | **OPEN — accepted interim risk**; worker-process isolation is the production shape, scheduled post-v16.4.5 | documented in `serving_router.retire`/`cancel_requests`; no test — requires a worker-process backend |
| RUN-402 | low | In-flight lease drain relies on cooperative cancellation on timeout; a request that ignores cancellation retains the backend indefinitely (by design — resources are never freed under a live lease) | **ACCEPTED RISK** — mitigated by budget deadlines; RUN-401 worker isolation is the real fix | `test_serving_router.py` drain-timeout tests |
| RUN-403 | low | `admin_audit` emergency exception writes no record while the store is broken (impossible by construction); the `audit_broken` block persists in memory only — a restart during outage could serve before reconcile | **PARTIAL** — mitigated: restart runs `recover()` which revalidates the store; a fully durable "audit-broken" flag is future work | `test_emergency_quarantine_survives_audit_outage` |
| OPS-004 | medium | Role separation not enforced by code: all role keys co-located under one storage root in the dev scaffold | **PARTIAL** — production path checks key roles; dev scaffold unchanged | `test_supervised_service.py` |
| OPS-005 | low | Snapshot immutability is process-level; a same-identity adversary can chmod and rewrite (detected, not prevented) | **ACCEPTED RISK** — stage-time re-verification | `test_v1641_artifact_closure.py` |
| OPS-006 | low | `peer_uid()` has no macOS transport — service fails closed there | **OPEN — platform coverage** | `test_supervised_service.py` |
| OPS-007 | low | Event-log truncation detectable via local anchor only | **ACCEPTED RISK** — external anchor is future work | `test_journal_integrity.py` |
| ENV-001 | info | No GPU, CUDA, or native-QW3 qualification in this release — all evidence is CPU/offline | **OPEN — environment** | `VALIDATED_ENVIRONMENT.json`; ctest 28/28 CPU stub |

## Implemented vs qualified (required distinction)

* **Implemented AND tested here**: every SEC-301..307 repair, with
  deterministic fake backends plus a real tiny HF model through the
  real `PeftServingBackend` for the activate→query→rollback→restore
  loop.
* **Implemented, not independently qualified**: `--production` startup
  path is code-complete and startup-refusals are unit-tested, but no
  production deployment was executed. Worker-process backend isolation
  (RUN-401) is designed but not implemented. GPU inference is
  unverified (no CUDA host).
* **Not implemented (by plan)**: SEC-005/006 controller correctness
  (v16.4.5), SEC-007 learning qualification (v16.5.0).
