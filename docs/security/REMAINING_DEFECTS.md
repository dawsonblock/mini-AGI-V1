# Remaining Defects / Open Items — v16.4.9

## Audit-reproduced defects — FIXED (SEC-801..805)

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

- **G10 real-model lifecycle** — HF/PEFT load, inference, cancel,
  rollback, and revocation INSIDE the NsJail boundary still need a
  real-model run on a qualified host.
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
