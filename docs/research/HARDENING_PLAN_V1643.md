# v16.4.3 — Full Repair and Security Closure Plan

Baseline: v16.4.2 Authority and Activation Closure.
Goal: make the existing architecture enforce the guarantees it claims —
not more intelligence, not another orchestration framework. Retain the
Ed25519 authority, signed revocation snapshots, measured artifacts,
admission grants, supervisor, and activation journal; close the places
where security decisions still trust process-local state, caller inputs,
or unverified historical records.

## Confirmed findings

| Pri | Finding | Repair |
|-----|---------|--------|
| P0 | SEC-201 | Durable single-use admission grants (in-memory set) |
| P0 | SEC-207 | Atomic grant reservation under concurrency |
| P0 | SEC-203 | Role-based authorization for supervisor operations |
| P0 | SEC-204 | Eliminate caller-controlled storage paths |
| P0 | SEC-202 | Reconstruct actual serving state after restarts |
| P1 | SEC-205 | Verify journal integrity and state transitions |
| P1 | SEC-206 | Bind authorization to backend implementation and policy epoch |
| P1 | OPS-001 | Bound socket requests, connections, and execution time |
| P1 | OPS-002 | Repair cross-UID communication and service deployment |
| P1 | OPS-003 | Connect supervised activation to real inference traffic |

Release policy: all P0 closed before "security-hardened"; all P1 closed
before "production-serving qualification". Learning-controller changes
(WP9) follow in v16.4.4; real-model science in v16.5.

## Work packages

* **WP1 — durable atomic grant ledger.** `runtime/authority_store.py`:
  SQLite (WAL, synchronous=FULL), `admission_grants` (unique grant_id,
  nonce, digest, activation_id) + `runtime_events` (hash-chained event
  log). `BEGIN IMMEDIATE` reservation transaction; a consumed grant
  stays consumed across failures and restarts; idempotent retry returns
  the recorded outcome of the same request identity. One transactional
  authority — the JSONL journal becomes a verified projection, not a
  second authority.
* **WP2 — role-based authorization.** `runtime/access_policy.py`:
  `PrincipalContext` derived from the authenticated peer uid (never a
  client-supplied `role` field); per-operation permission table;
  separate `research.sock` / `operator.sock` endpoints; signed audit
  events for administrative decisions.
* **WP3 — no caller-controlled writes.** Server-generated opaque
  activation ids (`secrets.token_hex(16)`); campaign/seed are metadata
  only; protected storage layout (`snapshots/ receipts/ state/
  revocations/ quarantine/`); service-owned receipt destinations;
  symlink/non-regular rejection; containment under the storage root.
* **WP4 — verified journal.** Journal V2 records (sequence, hash chain,
  state-transition grammar, role-bound signatures); validated reading
  interface; one-time V1 parser + signed migration checkpoint; no
  fabricated completions.
* **WP5 — crash-safe restart.** Serving-state model (`UNAVAILABLE`,
  `RECOVERY_REQUIRED`, `PREPARING`, `READY`, `SERVING`, `QUARANTINED`);
  boot never restores SERVING from a pointer alone; restoration requires
  fresh admission, byte re-verification, and health probes; crash-
  injection matrix at every lifecycle boundary.
* **WP6 — backend identity + policy epoch.** `RuntimeBackendManifestV1`
  (implementation digest, dependency-lock digest, supported formats,
  policy epoch, signature); supervisor verifies the installed backend
  matches the authorized implementation and the grant's policy epoch is
  not superseded.
* **WP7 — bounded service I/O.** Size-bounded reads, socket deadlines,
  fixed worker pool with bounded queue, per-principal quotas, auth-
  failure rate limiting, graceful shutdown; protected socket directory
  with explicit ownership for cross-UID access.
* **WP8 — real inference routing.** `ServingRouter` (lock/lease
  versioned routing table) exposing only active+healthy+committed
  handles; rollback rechecks authorization, revocation, and predecessor
  health; durable rollback intent + completion; interrupted rollback is
  idempotent.
* **WP9 — learning controller (v16.4.4, deferred).** `NO_CHANGE`
  abstention, verified attempt records, budgets, exploration/promotion
  separation, determinism.
* **WP10 — qualification + release engineering.** Reconcile packaged
  test counts; regression suite layout under `tests-python/security/`,
  `runtime/`, `integration/`; reseal + verify the release ZIP.

## Sequence (§16)

1. Single transactional authority store + operation-permission model.
2. Remove client-controlled storage paths.
3. Recovery: historical records can never masquerade as live state.
4. Bind verified journal, backend identity, policy, routing lifecycle.
5. Adversarial, cross-process, crash-injection, real-model tests.

Package name: `mini-AGI-V1-v16.4.3-Authority-Recovery-Hardening.zip`.
