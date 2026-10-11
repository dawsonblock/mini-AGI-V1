# Mini-AGI v16.4.6 — Worker Isolation Specification

Status: implemented and tested (946 passed / 4 skipped on macOS dev;
Linux/Colab qualification recorded separately).

## Trust boundary

```
Trusted supervisor ── runtime authority, authority DB, admission, signing
        │
        │  framed JSON v2 IPC (bounded, validated, no pickle)
        ▼
Untrusted model worker ── separate OS identity (production),
   private scratch (worker-owned), approved model artifacts only
```

The boundary is enforced by the operating system — environment
filtering alone is NOT filesystem isolation and is not treated as
such. A worker running under the supervisor's uid can read every file
the supervisor can; production therefore REQUIRES a distinct worker
identity and refuses to start without one.

## Policy model — `WorkerIsolationPolicy`

Backward-compatible fields (v16.4.5): `env_passthrough`, `env`,
`cpu_seconds`, `memory_bytes`, `max_processes`, `max_open_files`,
`max_file_bytes`, `demote_to`, `sandbox_profile`.

New enforcement requirements (v16.4.6) — each flag means "refuse to
operate if this host cannot enforce it":

| Field | Requirement |
|---|---|
| `require_separate_identity` | `demote_to` set, worker uid ≠ supervisor uid, host has setuid privilege |
| `require_filesystem_confinement` | implies separate identity — a same-uid confinement claim is rejected as illusory |
| `require_process_containment` | process-group/session ownership + group signaling available |
| `require_network_isolation` | a working network-namespace mechanism must exist (Linux); unavailable → refuse |

`production_policy(worker_uid, worker_gid, …)` builds the full
profile. `validate_isolation(policy)` checks the host's real
capability set (`supported()`) and returns the mechanism report that
the startup line records — or raises `IsolationError`.

### Resource-limit semantics (corrected)

`RLIMIT_NPROC` bounds the entire UID, not a process tree. Setting it
on a same-identity worker would starve the supervisor's own forks
(observed: fork → `EAGAIN`). The default policy therefore leaves
`max_processes` unset; the production profile sets it (32) because a
demoted worker has its own uid.

## Identity separation mechanics

- `demote_to=(uid,gid)` applied in `worker_preexec` between fork and
  exec (`setgroups([])`, `setgid`, `setuid`, verify, `_exit(70)` on
  failure — demotion failure never produces a same-uid worker).
- Worker scratch: `make_worker_scratch` creates the directory under
  the supervisor identity, then `chown`s it to the worker uid with the
  supervisor's gid, mode `0750`. The demoted worker can enter and
  write; the supervisor retains read/traverse for result collection;
  everyone else is denied. `tmp_is_private(path, worker_uid=…)`
  verifies this contract.
- Protected assets (signing keys, authority DB, revocation store,
  production sockets) remain supervisor-owned with restrictive modes —
  the worker's uid simply cannot open them. Verified from inside a
  real demoted worker (`AccessProbeBackend`) on Linux.

## Process containment (WP2)

- Every worker gets a random `MINAGI_WORKER_UNIT` token in its initial
  environment — inherited by all descendants, unguessable, and
  un-scrubbable via `/proc/<pid>/environ`.
- `terminate()`/`_kill_and_reap()` now run the
  `WorkerProcessController` protocol: snapshot the ppid-descendant set
  while the leader lives → SIGTERM the worker's own process group
  (`os.killpg`, only after verifying `getpgid(pid)==pid` and the group
  is not ours — no unchecked negative PIDs) → bounded grace → SIGKILL
  the group → individually SIGKILL any survivor found by tracked-pid
  liveness or the env-marker scan → `verify_terminated` must return
  the empty set, else the death record names the unconfirmed pids.
- `start_new_session=True` (group ownership) was already in place —
  v16.4.6 is what actually uses it.

Known limits, documented not hidden: a child that `exec`s with a
scrubbed environment escapes the marker scan; platforms without
`/proc` (macOS) cannot scan environ and rely on tracked ancestry +
group kill; Linux cgroups remain the stronger mechanism and are the
recommended production substrate.

## IPC exception safety (WP3)

The worker reports errors as a bounded `error_code` plus inert text
(`error_module`/`error_type`/`traceback`, truncated). Dynamic
exception-class resolution (`_remote_exception_class`) is REMOVED —
worker-controlled metadata can no longer instantiate `SystemExit`,
`KeyboardInterrupt`, `GeneratorExit`, or any `BaseException` inside
the supervisor. Codes: `WORKER_LOAD_FAILED`, `WORKER_INFERENCE_FAILED`,
`WORKER_CANCELLED`, `WORKER_BUDGET_EXCEEDED`,
`WORKER_PROTOCOL_VIOLATION`, `WORKER_UNAVAILABLE`,
`WORKER_INTERNAL_ERROR`. The only code that maps to a local class is
`WORKER_BUDGET_EXCEEDED → BudgetExceeded` (ordinary `Exception`).

## Overflow integrity (WP4)

`read_overflow` is descriptor-based end to end:

1. The overflow directory's fd is pinned open at spawn
   (`O_RDONLY|O_DIRECTORY|O_NOFOLLOW`) — renaming the directory or
   swapping in a symlink at its path cannot redirect later reads.
2. The leaf opens relative to that fd with `O_NOFOLLOW`.
3. `fstat` on the SAME descriptor: regular file, size == announced,
   ≤ `MAX_OVERFLOW_BYTES`.
4. Bounded read from that descriptor → SHA-256 re-measure → strict
   JSON parse. The bytes checked are the bytes parsed.

The ref's `name` was already validated as a bare file name by
`OverflowRef.from_doc` (no separators, no dotfiles).

## Production wiring (WP5)

`service.py` CLI: `--worker-uid`, `--worker-gid`, `--worker-user`,
`--worker-net-isolation`. Under `--production` + process isolation,
`resolve_worker_policy()` requires a distinct identity, validates the
profile against the host, and supplies `backend_id` from the VERIFIED
signed backend manifest — the parent interpreter never imports the
backend module to learn its identity. The launch line records the
mechanisms actually enabled. Development mode is unchanged and
explicitly labeled (`development same-identity`).

## Platform qualification status

| Platform | Separate identity | Process containment | Overflow fd | Network ns |
|---|---|---|---|---|
| Linux (Colab root) | enforced — demote to uid 65534 | group + marker scan (/proc) | yes | available via `unshare` |
| macOS dev | unavailable (no setuid privilege) → production REFUSES | group + tracked ancestry (no environ scan) | yes | unavailable → policy refuses |

Colab is an experimental platform, not a production-isolation
authority — qualification there proves the mechanisms work, not that
the host is trusted.
