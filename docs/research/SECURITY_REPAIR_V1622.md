# v16.2.2 — Security and Correctness Repair (Phase 1)

This release executes Phase 1 of the v17 development plan: the six
security/correctness defects identified in the audit are repaired, each
with a reproducing test that fails on v16.2.1 and passes here. No
scientific claims change; Campaign 2's qualified-negative result and
Campaign 1b's qualification stand unchanged under their original
schemas. Campaign 3 remains drafted and unexecuted.

## Repairs

| ID | File | Defect (v16.2.1) | Repair |
|----|------|------------------|--------|
| FIX-001 | `minagi/v161/evaluators.py`, `minagi/v161/execution_sandbox.py` | `executor_score` ran corpus-supplied python with the full parent environment, no filesystem/network isolation, and no process-tree termination | Declarative evaluator (allowlisted ops, no code execution) preferred for deterministic tasks; `python_assert` checkers run only under an OS-enforced sandbox; fail closed when no backend exists |
| FIX-002 | `minagi/v161/plasticity.py` | `RankAllocator.grow()` clamped to remaining headroom alone, so an exhausted budget drove a task's rank to 0 | `maximum_allowed = min(per_task, max_rank, current + headroom)`; growth is strictly non-decreasing and blocked growth leaves state unchanged |
| FIX-003 | `minagi/v161/generations.py` | `record_promotion` accepted any digest-shaped string as promotion authorization | Append-only `PromotionEvent`s validated against the promotion authority: signed envelope, bound record/campaign/qualification, authorized signer valid at verification time, unexpired, unrevoked |
| FIX-004 | `minagi/v161/stats.py` | false-activation rate divided harmful flips by all pairs | Denominator is previously-correct baseline cases (matching the preregistered bound); conditional regression reported alongside |
| FIX-005 | `minagi/v161/authority.py` | `not_before`/`not_after` stored but never enforced | Validity windows enforced in `is_authorized`/`assert_authorized`/`verifier` and ledger verification, at an explicit verification time; malformed bounds and naive datetimes fail closed |
| FIX-006 | release metadata | attestation `manifest_sha256`/`reissued_utc` stale vs the last reissue; version identities disagreed | Manifest reissued, attestation reconciled, `verify_release.py` now fails on attestation↔manifest↔version drift, regression test added |

## Evaluator sandbox (FIX-001)

Executable evaluation runs through `minagi.v161.execution_sandbox`:

* **macOS** — `sandbox-exec` with an SBPL profile: `(deny default)`;
  reads allowlisted to the interpreter prefix, system libraries, the
  interpreter's resolved dylib closure (via `otool`), explicitly
  mounted read-only reference inputs, and the ephemeral workspace;
  writes allowed only inside the workspace; network denied.
* **Linux** — `bwrap --unshare-all` (user/pid/net/ipc/uts namespaces),
  read-only system binds, workspace-only writes, `--clearenv`,
  `--die-with-parent`.
* **Both** — rlimits (CPU, file size, process count; address space on
  Linux), a minimal environment (the evaluator's `os.environ` is never
  inherited), an ephemeral workspace removed after the run, and
  process-group termination on timeout.
* **No backend → no execution.** `SandboxUnavailable` is raised; there
  is deliberately no unsandboxed fallback. `MINIAGI_SANDBOX_BACKEND`
  may pin a backend (or `none` to force the fail-closed path).

Reproducing evidence (v16.2.1 → v16.2.2): a checker could read a
signing-key file, saw the evaluator's environment, and its grandchild
survived the timeout; all three are blocked now.

## Residual-risk register

| # | Risk | Status / mitigation |
|---|------|---------------------|
| R1 | macOS cannot lower `RLIMIT_AS` below its current (huge) VM reservation — the memory cap is best-effort there | Wall-clock timeout, output caps, and `RLIMIT_FSIZE` still bound damage; memory caps are enforced on Linux. Reported honestly as `limits.memory_limit_enforced=false` in every macOS `SandboxResult` |
| R2 | `sandbox-exec` is deprecated by Apple (functional on Darwin 25.2) | Fail-closed if absent; a validated replacement backend (microVM/container) is required before claiming macOS executable-evaluation isolation for adversarial corpora |
| R3 | `bwrap` requires unprivileged user namespaces; some container/Colab hosts disable them | Executable evaluation fails closed there — campaigns with `python_assert` rows cannot run until bubblewrap or an isolated VM is provided |
| R4 | The declarative `regex_fullmatch` op is not ReDoS-proof | Pattern (≤512 chars) and input (≤4096 chars) caps; prefer non-regex ops; regex evaluation is bounded but not linear-time guaranteed |
| R5 | Sandbox profiles permit file *metadata* (existence/timestamps) on allowlisted prefixes | Content reads remain allowlisted; metadata of paths outside the allowlist is denied |
| R6 | Authority validity windows are checked against the system clock (no attested time source) | Acceptable for a single-host research scaffold; attested time is required before cross-host authority claims |
| R7 | `PromotionEvent`s guarantee generation-chain authorization; runtime admission control is Phase 3 (v16.4) work | Until then, deployment must read the promotion decision (never file presence) — unchanged from v16.2.1 |
| R8 | Kernel-level sandbox escapes are out of scope | The sandbox is defense-in-depth for governed-corpus checkers, not a claim that arbitrary adversarial code is harmless |

## Test inventory

* `test_v1622_evaluator_sandbox.py` — declarative allowlist, secret
  read, authority-state read, outside writes, network, environment
  visibility, orphaned-process survival, read-only reference inputs,
  ephemeral workspace cleanup, fail-closed paths (Linux-only memory
  test skipped on macOS).
* `test_plasticity_generations.py` — rank-budget exhaustion, headroom
  arithmetic, randomized growth invariants; bare-digest promotion
  rejection, unregistered/wrong-role signers, tampered values, wrong
  record/campaign binding, expiry, revocation, signature re-verification.
* `test_v1622_authority_time.py` — expired/not-yet-valid/revoked keys,
  window boundaries, malformed bounds, naive datetimes, verifier
  exclusion, ledger time-awareness.
* `test_v166_scientific_gates.py` — corrected false-activation
  denominator, conditional regression, zero-denominator behavior,
  qualifier surfacing.
* `test_v1622_release_metadata.py` — attestation ↔ manifest ↔ version
  identity reconciliation, signature verification.
* `test_v1622_verify_release_script.py` — adversarial exercise of the
  release verifier against a signed fixture: tampered/extra/missing
  files, corrupted manifest digest entries, unsupported manifest
  schema, forged signatures, stale attestation digests, stale reissue
  history, missing attestation, and version-identity disagreement all
  fail with their documented exit codes.

Every defect-reproducing test fails on the v16.2.1 code and passes on
this release (verified by reverting each repaired module in turn).
