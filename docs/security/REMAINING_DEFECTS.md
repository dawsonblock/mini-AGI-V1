# Remaining Defects / Open Items — v16.4.8

## Open (blocking qualification claims)

- **No Linux run yet** — G1–G4 and G10 are source-complete but
  unqualified until `run_nsjail_qualification.sh` executes on the
  target host (needs an SSH destination or equivalent).
- **SEC-707 release metadata** — manifest/signature/SBOM/source
  inventory must be regenerated from the final tree at freeze.
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
