# v16.4.2 — Authority and Activation Closure

UPGRADE_PLAN_V17 milestone P0, gate: *all supported production-serving
paths enforce independent admission, and fault injection demonstrates
that unauthorized or incompletely committed candidates cannot receive
inference traffic.*

This document describes the v16.4.2 changes and the executable evidence
behind them. Nothing here changes a scientific claim — Campaign
qualification status is unchanged.

## What was wrong

Three weaknesses were confirmed verbatim in v16.4.1:

1. `TrustedRuntimeLauncher` ran `backend.load(snapshot)` **before**
   activation evidence was durable. A persistence failure afterwards
   raised an error but had no guaranteed unload — a model could be left
   resident and routable without a receipt.
2. `RuntimeAdmissionController` computed
   `covered = qual.get("runtime_backends") or list(QUALIFIED_BACKENDS)` —
   a qualification that omitted backend coverage silently covered every
   supported backend.
3. Revocation evidence was a bare unsigned digest list with a
   timestamp: unsigned, non-monotonic, and unable to reject a
   future-dated list.

## What v16.4.2 changes

### Measurement is not authorization

`ApprovedSnapshot` is now `MeasuredSnapshot`
(`v161/immutable_snapshot.py`): it proves artifact content and file
structure only. Deployment authority is a separate signed object,
`AdmissionGrantV1` (`security/admission_grants.py`), issued by the new
`admission` authority role and bound to the promotion-decision digest,
qualification digest, runtime-manifest digest, measured artifact-root
digest, backend id, policy epoch, revocation epoch, validity window,
nonce, and the runtime identity it is issued *to*. Grants are
short-lived, single-use (the supervisor journals consumed grant ids),
and verified independently by the supervisor — a caller's claim of
authorization is not evidence.

### Transactional activation

`runtime/supervisor.py` owns every production model load through the
state machine

    REQUESTED → AUTHORIZED → STAGED → PREPARED → READY → COMMITTED → ACTIVE

with terminal ABORTED/QUARANTINED states. `runtime/durable_journal.py`
is the crash-consistency substrate: every transition is fsynced before
the effect it describes, the active-version pointer is atomically
replaced (tmp + fsync + rename), and the activation-completion record
is signed by the runtime role. `activation_state.py` makes the
transition legality a protocol, not a convention.

Failure semantics:

- a failed load aborts the candidate — nothing half-loaded proceeds;
- a failed health probe unloads the candidate and the previous healthy
  model keeps serving;
- a pointer-write failure aborts without touching routing;
- a completion-record failure after the swap rolls the pointer back to
  the previous *live* handle — a pointer never routes to an unloaded
  model;
- `recover_from_journal()` reconciles on restart: committed-but-unrouted
  candidates abort, committed-and-routed candidates get a reconciled
  signed completion, pre-commit stalls abort, and a pointer naming an
  activation with no journal history is cleared rather than served;
- `rollback()` restores the retained committed predecessor, which stays
  resident as the protected deployment target. There is no path that
  restores traffic to a dead runtime — a missing live handle is refused
  and must re-admit through admission.

### The privilege boundary

`runtime/service.py` + `security/trusted_authority_client.py` wrap the
supervisor and launcher in a dedicated service process speaking
newline-delimited JSON over an authenticated Unix socket. Peer
credentials are checked via `SO_PEERCRED`/`getpeereid` against an
explicit allowed-uid policy; platforms that cannot prove a uid fail
closed. Clients submit *documents* and name *backends* — they never
supply loader objects, signing keys, or writable paths into the
protected store. `scripts/trusted_launch.py` drives the supervised path
directly; the research-side `admit_runtime.py` still emits only the
non-production receipt.

### Explicit backend coverage and signed revocations

`runtime_admission.py` no longer defaults missing `runtime_backends`:
a signed qualification that omits backend coverage is refused, and
hf-peft coverage never transitively qualifies the native backends
(§3.3). `security/signed_revocations.py` adds `RevocationSnapshotV2`:
signed by the dedicated `revocation` role, epoch-monotonic,
freshness-and-future bounded, stored atomically by `RevocationStore`,
which refuses epoch regression, epoch replay, and — on read — never
falls through to an older snapshot when the newest is invalid. Missing
revocation evidence fails closed for production activation.

## Executable evidence

- `tests-python/security/test_signed_revocations.py` — unsigned lists,
  wrong-role signatures, forged signatures, digest mismatch, 2100
  future-dating, staleness, expired `valid_until`, epoch replay,
  store regression/write-once/refuse-when-empty, newest-invalid never
  silently skipped, atomic naming.
- `tests-python/security/test_admission_grants.py` — unsigned grants,
  wrong-role issuers (plan/promotion/runtime/qualification), forged
  signatures, wrong audience/backend/manifest/artifact-root bindings,
  stale revocation epoch, expiry, future-dating, envelope mismatch.
- `tests-python/runtime/test_activation_supervisor.py` — the full
  lifecycle, write-once activation ids, arbitrary-object refusal, grant
  binding mismatches, replay refusal, load/probe/pointer/completion
  failure injection, quarantine+rollback, live-handle enforcement, and
  all four crash-recovery dispositions.
- `tests-python/runtime/test_supervised_service.py` — peer-credential
  boundary, unknown-op and unregistered-backend refusal, recovery op.
- `tests-python/v161/test_v1642_activation_closure.py` — the spec's
  13-row qualification table end to end through the launcher.

## Still open

SEC-005/SEC-006 (controller correctness — evidence-backed prerequisites,
NO_CHANGE, budgets) are the v16.4.3 gate. SEC-007 (real-model Campaign
3 qualification) is the v16.5.0 gate. OPS-003 (macOS lacks a CPython
peer-credential mechanism; the service fails closed) awaits a platform
transport.
