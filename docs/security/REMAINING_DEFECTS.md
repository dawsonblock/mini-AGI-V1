# Remaining Defects / Open Items — v16.4.10

## v16.4.10 audit defects — FIXED (SEC-901..906, VAL-901/902)

- SEC-901 strict cgroup.events parsing — UNKNOWN is never EMPTY; no
  cgroup.procs fallback
- SEC-902 shared ContainmentState — populated 1 cannot be overridden
  by inactive unit/exited client; InvocationID-bound teardown
- SEC-903 broker-generated w-<hex32> ids; caller ids refused;
  root-constrained descriptor-relative workspaces
- SEC-904 euid==0 shortcut removed; capability preflight vs
  enforcement qualification separated
- SEC-905 AuthorizedLaunchRecord: signed, expiry+epoch-bound,
  resolver-backed, single-use nonce, exact-digest bindings
- SEC-906 mandatory real preflight probe at production startup with a
  persisted host-bound qualification record
- VAL-901 true mid-generation cancellation; VAL-902 signed
  A->B/rollback/cold-recovery/revocation inside the jail

## v16.4.9 audit defects — FIXED (SEC-801..805)

The external audit of v16.4.8 reproduced five defects; all are
repaired and re-qualified on this branch (see ISSUE_REGISTER_V1648.md
for the fix table):

- SEC-801 cgroup occupancy read of a nonexistent file + `None`
  certified as empty
- SEC-802 ancestor mounts exposing denied descendants
- SEC-803 launch accepted without a verified cgroup
- SEC-804 preflight graded only a subset of measured properties
- SEC-805 backend-manifest/budget digests unvalidated; nonce
  reservation not atomic

## Open (blocking production claims)

- **G10 real-model lifecycle** — DONE on the qualified container:
  real HF/PEFT load, inference, cooperative cancel, fresh-jail model
  switch, and verified-empty cgroup teardown all PASS inside NsJail
  as uid 65534 (see REAL_MODEL_LIFECYCLE.md). GPU scale-out and
  production-scale models remain to qualify on target hardware.
- **Non-root `MINAGI_WORKER_UID` qualification** — the containerized
  run exercised root→65534 mapping; a rootless `systemd-run --user`
  deployment is a separate qualification.
- **IPC transport** — production channel is `systemd-run --pipe`
  carrying the same strict framed-JSON protocol (bounded, typed,
  no pickle). The plan's optional dedicated Unix-socket broker
  channel is NOT implemented; --pipe gives a per-unit, broker-mediated
  channel without adding a socket surface. Recorded decision, not an
  oversight. If a Unix socket is later required it must keep the same
  parser/bounds and add peer-credential checks.

## Known limits (accepted, documented)

- Development executor remains the weakest mechanism set (env marker,
  kqueue, pgid) — trusted local testing only, refused under
  `--production`.
- Rootless `systemd-run --user` path is implemented but unqualified;
  qualify whichever mode will run in production.
- NsJail seccomp is the built-in `default`; a deployment-specific
  kafel policy is future work.
- A worker that compromises the *kernel* (not the sandbox config) is
  outside this threat model; use VM isolation for arbitrary code.
