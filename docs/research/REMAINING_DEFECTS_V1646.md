# Mini-AGI v16.4.6 — Remaining Defects / Open Items

Closed this release: SEC-501, SEC-502, SEC-503, SEC-504, VAL-501
(production startup wiring). See `V16_4_6_REPAIR_REPORT.md` and
`WORKER_ISOLATION_SPEC_V1646.md`.

## Open — platform qualification

- **Worker privilege isolation on Linux** — the demotion path
  requires privilege (root or CAP_SETUID); on macOS dev hosts the
  tests skip and production correctly refuses. Qualified on the
  Colab Linux root profile; a dedicated Linux VM with a service
  account remains the recommended production substrate.
- **macOS containment gaps** — no `/proc` environ scan, so a
  detached descendant that exec's with a scrubbed environment is
  outside the marker set; tracked ancestry + group kill still cover
  the normal tree. sandbox-exec remains opt-in/deprecated upstream.
- **Network isolation** — wired as a required-when-requested control
  (`--worker-net-isolation`); enforcement is a Linux network
  namespace. Not enabled by default because enforcement is platform-
  dependent; requiring it on a host without the mechanism refuses
  startup rather than pretending.
- **cgroups** — the stronger descendant-containment mechanism
  (catches post-parent-death escapees). Recommended for Linux
  production; not yet wired.

## Open — carried forward

- **G7 real-model qualification (v16.4.7)** — real PEFT load → infer
  → kill → quarantine e2e continues to pass; the full
  activate/infer/cancel/terminate/rollback/restart/revoke/corrupt
  matrix on GPU under the demoted worker identity is the next gate.
- **G9 scientific credibility** — no learning-improvement claim; the
  Campaign 3A result stands as historical evidence only.
- **Windows** — restricted token + Job Object + ACL design is
  specified in the plan; not implemented (POSIX substrate only).
