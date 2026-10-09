# v16.4.3 — Hardened Runtime: Authority, Recovery, and Serving Closure

UPGRADE_PLAN_V17 milestone P0/P1, gate: *durable single-use grants,
strict operation authorization, safe paths, verified recovery, journal
integrity, backend identity, and bounded service I/O.*

This document describes the v16.4.3 repairs and the executable evidence
behind them. Nothing here changes a scientific claim — Campaign
qualification status is unchanged. Learning-controller repairs
(SEC-005/SEC-006) are the v16.4.4 gate; real-model qualification
(SEC-007) is the v16.5.0 gate.

## What was wrong

Ten defects were confirmed verbatim in v16.4.2:

| Finding | Confirmed defect |
|---|---|
| SEC-201 | `ServingSupervisor` kept consumed grant ids in `self._consumed_grants: set[str]` — process-local, lost on restart |
| SEC-207 | Grant check-and-insert was two steps under a Python set — two threads or processes could both authorize one grant |
| SEC-203 | `SupervisorService` authenticated the peer uid, then dispatched `launch`/`quarantine`/`rollback`/`recover` with one allowlist — an authorized research client held operator authority |
| SEC-204 | `TrustedRuntimeLauncher` derived `activation_id` from caller `campaign_id`/`seed` and appended it to `snapshot_root`; the service accepted a caller-supplied `receipt_path` — clients steered privileged writes |
| SEC-202 | Recovery restored `_active_id` from `active.json` with no loaded model behind it — a historical pointer masqueraded as live serving state |
| SEC-205 | `DurableJournal.records()` parsed persisted JSON without verifying digests, signatures, or transition legality — tampered history was trusted input |
| SEC-206 | `backend_binary_digest`/`policy_epoch` had permissive defaults and were never supplied or enforced — a grant for one implementation could activate another |
| OPS-001 | The service read with unbounded `readline()` then checked size, and spawned an unbounded thread per connection |
| OPS-002 | Socket mode `0600` contradicted documented cross-uid research access; no socket-directory ownership model existed |
| OPS-003 | Activation control never reached inference: no router, no backend inference method, traffic could not be shown to obey admission |

## What v16.4.3 changes

### One transactional authority — `runtime/authority_store.py`

A single SQLite store (WAL, `synchronous=FULL`, `BEGIN IMMEDIATE`)
owns grant reservations, the hash-chained signed runtime-event log,
idempotent request outcomes, audit records, and the active-pointer
projection. There is no second authority for the same state — the
event log and the pointer live in the same database and are committed
in the same transaction.

`admission_grants` enforces `UNIQUE` on `grant_id`, `grant_nonce`,
`grant_digest`, and `activation_id`. Reservation is one atomic
transaction: signature, audience, validity window, backend binding,
policy epoch, and revocation epoch are verified first, then the
reservation row and the `AUTHORIZED` event commit together — or not
at all. A consumed grant stays consumed forever, including across
failed activations, process restarts, and crashes after the commit.
Retries of the same logical request return the recorded outcome from
`request_outcomes` rather than re-executing (idempotent launch).

If the database cannot be opened or its integrity check fails, every
authorization path raises — the store fails closed.

Evidence: `tests-python/security/test_grant_ledger.py` (replay,
restart, two-thread and two-process races, nonce collision,
crash-after-reservation, corrupt/unavailable DB, idempotent retry).

### Per-operation authorization — `runtime/access_policy.py`

`PrincipalContext` carries an `operator_id` and `role` derived from
authenticated OS credentials (`peer_uid` → uid → principal map), never
from client-supplied JSON. `OPERATION_ROLES` maps each operation to the
roles that may perform it; every dispatch calls
`require_permission()` before any side effect and writes a signed
audit row for both allowed and refused operations.

| Operation | research | operator | runtime | admin |
|---|---|---|---|---|
| `ping`, `status` | ✓ | ✓ | ✓ | ✓ |
| `launch` (proposal) | ✓ | ✓ | ✓ | ✓ |
| `recover`, `rollback` | — | ✓ | ✓ | — |
| `quarantine` | — | ✓ | — | ✓ |
| `infer` | ✓ | ✓ | ✓ | ✓ |

The service exposes `research.sock` (status/proposal/infer only) and
`operator.sock` (lifecycle). Endpoint whitelists are enforced in the
dispatcher *independent* of filesystem permissions — socket access is
not authorization. The development insecure mode now requires an
explicit `dev_role` and cannot authorize `admin`.

Evidence: `tests-python/runtime/test_supervised_service.py` (every
privileged op attempted as research/operator/runtime/revoked/unknown),
`test_supervised_service.py::test_*`.

### Filesystem authority — `secure_dir` / `contained_child`

Activation ids are `secrets.token_hex(16)` generated inside the
service; `campaign_id`/`seed` are metadata, never path components.
Receipts are written under a service-owned `receipts/` root named by
the activation id — clients receive an artifact id, not a path.

`contained_child()` refuses: symlinked roots, non-opaque names,
symlink components anywhere between root and candidate, resolved
paths escaping the resolved root, and nonexistent ancestors under
attacker control. `secure_dir()` verifies ownership/mode and refuses
symlinked or wrong-owned directories.

Evidence: `tests-python/security/test_path_containment.py` (traversal,
absolute paths, separators, unicode, symlinked parents, post-validation
races, hostile `campaign_id` values reaching `launch`).

### Verified journal — `runtime/journal_v2.py`

Every runtime event is schema-checked, canonically digested,
hash-chained (`previous_event_digest` → previous `event_digest`),
signed by the role `EVENT_ROLES` requires for its event type, and
validated against the activation transition grammar
(`'' → REQUESTED → AUTHORIZED → STAGED → PREPARED → READY →
COMMITTED → ACTIVE`, terminals `ABORTED`/`QUARANTINED`, supervised
`ACTIVE → ACTIVE` rollback/restore arcs). `verify_event_log()` also
checks grant/activation identity consistency, conflicting
completion histories, and the signed-checkpoint anchor so tail
truncation cannot pass silently.

`migrate_v1_journal()` replays a legacy `DurableJournal` file,
verifies each V1 record against its own embedded signature/digest,
re-issues each event under the `migration` authority with the original
signature preserved inside `detail`, and seals the output with a
signed `migration_checkpoint`. Historical records are validated
evidence, never live state. `scripts/migrate_journal_v1_to_v2.py` is
the operator entry point.

Evidence: `tests-python/runtime/test_journal_integrity.py` (modified
record, truncated tail, spliced middle, unsigned completion,
wrong-role signature, illegal transition, forged `from_state`,
migration round-trip and tampering).

### Crash-safe recovery — `ServingSupervisor.recover()`

Serving state is explicit: `UNAVAILABLE`, `RECOVERY_REQUIRED`,
`PREPARING`, `READY`, `SERVING`, `QUARANTINED`. Every cold start is
`UNAVAILABLE`. `recover()` stops routing, verifies the event log,
finds the last committed candidate, re-checks qualification and
revocation freshness, re-measures artifacts, obtains fresh admission,
loads into a non-serving instance, probes health, commits a new
`RECOVERED → ACTIVE` transition, and only then enables routing. A
durable `ACTIVE` record is evidence for re-qualification, not a live
handle. Interrupted activations, rollback intents, and
quarantine intents reconcile deterministically; rollback only targets
a predecessor that is verified live and re-authorized at the current
revocation epoch.

Evidence: `tests-python/runtime/test_activation_supervisor.py` crash
matrix (termination after grant reservation, staging, prepare, ready,
commit intent, pointer move, receipt, and during rollback) and
`test_revocation_during_serving` / rollback-authority tests.

### Backend identity — `runtime/backend_manifest.py`

`RuntimeBackendManifestV1` binds `backend_id`,
`implementation_digest`, `dependency_lock_digest`,
`inference_configuration_digest`, `qualification_digest`, and
`policy_epoch` under a signature. The supervisor verifies the
installed backend against the manifest — a grant naming `hf-peft`
cannot activate a different implementation claiming the name, and a
grant issued under a superseded policy epoch is refused at admission
and again at rollback.

Evidence: `tests-python/security/test_backend_identity.py` (wrong
digest, changed dependency closure, stale policy epoch, cross-backend
substitution).

### Bounded service I/O — `runtime/service.py`

Requests are read with `readline(MAX_REQUEST_BYTES + 1)` under socket
deadlines — oversized input is refused before parse. A fixed worker
pool with a bounded queue replaces thread-per-connection; accept is
bounded by `max_connections`; per-principal quotas and
authorization-failure rate limiting apply; shutdown stops admitting
new activations. Bare connection resets are handled on the client.

The socket directory is service-owned (`0700`); per-endpoint sockets
are created `0660` under an explicit group model. Development insecure
mode remains available only with an explicit flag and `dev_role`.

Evidence: `tests-python/runtime/test_supervised_service.py` (oversized
request refused pre-parse, connection bound, deadline, refused-auth
rate limit, worker-pool exhaustion, cross-uid allow/deny).

### Serving router — `runtime/serving_router.py`

`ServingRouter` exposes `route`/`activate`/`deactivate`/`drain`/
`status` over a versioned routing table under one lock. `route()`
dispatches only to the active handle's `infer()`/`generate()` and
refuses when nothing committed+healthy is serving; raw paths and
client-supplied objects can never become routes. Activation,
quarantine, rollback, and draining serialize; in-flight requests
complete on the version they started. `PeftServingBackend.infer()`
exists so routed calls reach real inference.

Evidence: `tests-python/runtime/test_serving_router.py`,
`tests-python/integration/test_full_admission_chain.py` (admit A →
route → prepare B without exposure → activate B → traffic moves →
failure → guarded rollback to A → unauthorized C refused → restart
recovery).

## Acceptance-gate status

| Gate | Status | Evidence |
|---|---|---|
| SEC-201 durable single-use grants | CLOSED | `test_grant_ledger.py` |
| SEC-207 atomic reservation under concurrency | CLOSED | `test_grant_ledger.py` thread/process races |
| SEC-203 role-based operation authorization | CLOSED | `test_supervised_service.py` op matrix |
| SEC-204 caller-controlled paths eliminated | CLOSED | `test_path_containment.py`, launcher rewrite |
| SEC-202 actual serving state reconstructed | CLOSED | `test_activation_supervisor.py` crash matrix |
| SEC-205 journal integrity + grammar + migration | CLOSED | `test_journal_integrity.py`, `migrate_journal_v1_to_v2.py` |
| SEC-206 backend implementation + policy epoch | CLOSED | `test_backend_identity.py` |
| OPS-001 bounded requests/connections/execution | CLOSED | `test_supervised_service.py` |
| OPS-002 cross-uid repair + deployment model | CLOSED | socket-directory ownership model; dual endpoints; `test_supervised_service.py` |
| OPS-003 supervised activation → real inference | CLOSED | `test_serving_router.py`, `test_full_admission_chain.py` |

Suite: **767 passed, 1 skipped** (Linux-only `RLIMIT_AS` on the macOS
host), ruff/flake8/pylint zero findings, CTest 28/28.

## Deliberately not in this release

- **SEC-005 / SEC-006** (attempt-record verification, negative-utility
  abstention, budgets, deterministic selection) — v16.4.4 gate.
- **SEC-007** (real-model training + independently evaluated
  transfer/retention/serving) — v16.5.0 gate.
- **Platform peer-credential coverage on macOS** — the service fails
  closed there; a platform transport remains future work.
