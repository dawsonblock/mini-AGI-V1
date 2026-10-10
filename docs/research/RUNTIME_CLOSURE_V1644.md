# v16.4.4 — Verified Runtime Closure: change and evidence report

Scope: repair the runtime lifecycle defects in
`v16.4.3-hardened-runtime` without creating a second runtime
framework. The learning architecture is untouched; controller
correctness (SEC-005/006) remains the v16.4.5 gate and real-model
learning qualification (SEC-007) the v16.5.0 gate.

Design principles enforced by this release:

* **Authorization is not activation** — a valid grant permits
  preparation, never traffic.
* **Durable intent is not live state** — SQLite records which model
  SHOULD serve; only a verified live handle DOES serve.
* **Deactivation is not unloading** — routing stops and leases drain
  before model resources are released.

## Source-level change report

| Finding | Defect in v16.4.3 | Repair |
|---|---|---|
| SEC-301 | `ServingSupervisor.commit_activation()` published the route BEFORE persisting the signed completion | Two-stage protocol: `AuthorityStore.commit_activation_intent()` atomically CASes the deployment record (generation/transition/phase) before `router.publish_route()`; the signed `routing_observed` event lands after traffic and is never claimed as pre-traffic evidence |
| SEC-302 | Activation failure called `clear_pointer()` — durable state empty while a model served from memory | `reconcile_deployment()` — generation-checked CAS restoring the last committed live predecessor (RESTORED) or recording durable UNAVAILABLE at a new generation |
| SEC-303 | One global `_inflight` counter; `abort()` unloaded before routing stopped; drain timeout returned while requests ran | Per-activation `RouteLeaseState` (accepting/draining/inflight); `stop_accepting → drain_until_idle → cooperative cancel → safe_unload`; timeout retains resources, never unloads; `abort()`/`quarantine_active()` retire the route first |
| SEC-304 | Production startup never supplied the backend manifest or policy epoch | `ProductionRuntimeConfig` + `--production`: signed backend manifest verified, installed module/dependency closure measured, operative policy epoch enforced, fresh revocation evidence, role-checked keys, protected storage — any failure refuses startup |
| SEC-305 | `SupervisorService._audit()` silently dropped audit failures | Durable `admin_audit` table (signed, hash-chained): privileged ops write the decision record before the side effect; audit failure refuses ordinary admin changes; emergency traffic shutdown proceeds but blocks activation until audit reconciles |
| SEC-306 | `PeftServingBackend.infer()` passed `max_new_tokens` straight to `generate()`; no prompt budget, deadline or concurrency bound | `InferenceBudgetPolicyV1` — actual prompt-token count checked, generation clamped, `max_time` deadline + cooperative `stopping_criteria`, router enforces concurrency and per-principal queue limits at lease acquisition |
| SEC-307 | `recover()` reconciled the pointer but never reloaded the model | `RecoveryManager.restore()` — validates durable state, issues a FRESH grant (consumed grants never reused), re-measures staged bytes, isolated load, health probes, commits + publishes through the normal verified path, or durably reports UNAVAILABLE; idempotent; a serving supervisor is never re-restored |

### New/changed durable schema

`deployments` row: `deployment_generation` (monotonic),
`desired_activation_id`, `previous_activation_id`, `transition_id`,
`transition_phase` (COMMITTING/COMMITTED/ROUTED/RESTORED/UNAVAILABLE),
`policy_epoch`, `last_committed_event_digest`.

New journal event kinds: `commit_intent`, `routing_observed` (signed),
`deployment_restored` (signed), `deployment_unavailable` (signed),
`restoration_completed` (signed), `unload_deferred` (same-state
marker). Deployment-level pseudo records use `__deployment__` and are
excluded from per-activation grammar checks alongside `__migration__`.

New `admin_audit` table: sequence-chained signed records
(principal/operation/target/policy digest/decision/before/after);
`verify_admin_chain()` detects tamper.

Migration: opening a v16.4.3 store with the v16.4.4 code seeds the
deployment row from the serving pointer (generation=1,
phase=COMMITTED) and creates `admin_audit`. Event history is
untouched. Covered by `test_v1643_store_migrates_deployment_row`.

## Test evidence (this worktree)

Python: `pytest tests-python` — **794 passed, 1 skipped** (Linux-only
RLIMIT_AS test), Python 3.12.0, macOS arm64, CPU-only. +27 vs the
v16.4.3 recorded baseline (767+1 → 794+1 including the 16 new
v16.4.4 rows; suite also includes the +10 prior rows absorbed in
interim counts).

Native: `cmake -DQW3_ENABLE_CUDA=OFF && ctest` — **28/28 PASS** on the
macOS CPU/stub build (CUDA paths unverified offline, as before).

Release integrity: `scripts/verify_release.py` — **PASS**, 1490 files,
pinned key fingerprint `sha256:0f9d73…b0eb`, manifest
`94dc9a43…`, release `mini-AGI-V1-v16.4.4-Verified-Runtime-Closure`.

### Decisive experiments

| Experiment | Result | Evidence |
|---|---|---|
| 1 — failed activation, pre-routing persistence failure | B processes zero requests; A remains authorized or durable UNAVAILABLE | `test_activation_supervisor.py` (commit-intent failure), `test_v1644_runtime_closure.py` |
| 2 — live switching and rollback | Rollback republishes a live predecessor; durable intent and router agree | `test_v1644_real_model_e2e.py` (real GPT-2 through real `PeftServingBackend`) |
| 3 — in-flight quarantine | New B requests refused; B NOT unloaded while the lease references it; released on lease end | `test_v1644_runtime_closure.py::test_quarantine_does_not_unload_live_inference` |
| 4 — crash and restoration | Service reauthorizes (fresh grant), reloads, health-checks, publishes — or UNAVAILABLE; never a phantom active model | `test_v1644_runtime_closure.py` (missing/tampered artifact refusal, idempotency), `test_v1644_real_model_e2e.py` |

Real-model evidence: the e2e test serves a locally-constructed
single-layer GPT-2 (HF format, `PreTrainedTokenizerFast`, real
`generate()`) through the production `PeftServingBackend` code path —
activate, query, switch, rollback, crash, restore — fully offline.
A large pretrained model is not claimed: this is a real model in the
real serving path, not a mock backend.

## Gate status

| Gate | Status |
|---|---|
| G1 no traffic before durable routing authorization | PASS — intent CAS precedes `publish_route`; failure-path tests |
| G2 rollback preserves generations + authoritative state | PASS — CAS reconcile; durable/live agreement tested |
| G3 no unload with active leases | PASS — per-activation leases; timeout never unloads |
| G4 missing backend identity/policy blocks production | PASS — `--production` startup refusals |
| G5 privileged ops have durable audit evidence | PASS — signed chained `admin_audit`; emergency-stop exception tested |
| G6 token/execution/concurrency/cancellation limits | PASS — budget enforced at router + backend |
| G7 cold restart restores healthy authorized model or stays unavailable | PASS — `RecoveryManager`; missing/tampered artifacts refuse |
| G8 fault-injection/concurrency/real-model tests | PASS (CPU, offline; GPU unverified) |
| G9 release metadata accurate | PASS — `verify_release.py` PASS; stale change manifest rewritten for v16.4.4 |

Residual qualification caveat: all evidence is CPU/macOS/offline. No
GPU, native-QW3, or multi-process-worker claim is made. See
`REMAINING_DEFECTS_V1644.md`.
