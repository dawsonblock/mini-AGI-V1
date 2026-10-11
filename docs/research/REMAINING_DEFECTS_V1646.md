# Mini-AGI v16.4.6 — Remaining Defects / Open Items

Closed this release: SEC-501, SEC-502, SEC-503, SEC-504, VAL-501
(production startup wiring). See `V16_4_6_REPAIR_REPORT.md` and
`WORKER_ISOLATION_SPEC_V1646.md`.

Follow-on hardening landed on this branch after the initial push:

- **kqueue `NOTE_TRACK` descendant tracking** — `KqueueDescendantTracker`
  follows forks event-driven where the kernel honours it. Real finding:
  on Darwin 25.x/arm64 the kernel answers `NOTE_TRACK` registration
  with `EOPNOTSUPP` — probed live by `kqueue_tracking_supported()`,
  which now tests a real registration instead of trusting constants.
- **seatbelt reference profile** — `configs/minagi-worker-seatbelt.sb`
  is the macOS descendant-containment mechanism: `deny process-fork`
  blocks every spawn primitive (fork/vfork/posix_spawn all route
  through it), plus `deny network*` and scratch-confined writes.
  Qualified live on this host: python runs, fork/subprocess/socket/
  out-of-scratch writes all refused. `validate_isolation` accepts it
  as the `process_containment` mechanism; `wrap_argv` renders
  `@SCRATCH@` into a supervisor-owned temp profile per worker.
- **Release gates rewritten for this project** — the old
  `host_gate.sh` / `gpu_gate.sh` / `source_audit.py` /
  `verify_manifest.py` were QW3-donor scripts that could never pass
  (qw3 CLI contract, `continual/` tree, RC9/RC10 files). Now:
  `host_gate.sh` runs source audit → native cmake+ctest → full
  pytest → manifest verify against a `git archive` extraction;
  `gpu_gate.sh` runs the real-model e2e and fails on skips;
  `source_audit.py` checks Mini-AGI release identity (VERSION↔
  pyproject↔`__init__`↔SBOM coherence, private-key material, dev
  paths in governed source, forbidden weight artifacts);
  `verify_manifest.py` delegates to `scripts/verify_release.py`;
  the dead HTTP probe `selected_replay_gate.py` was removed — the
  Mini-AGI service speaks on unix sockets, nothing could answer it.

## Open — platform qualification

- **Worker privilege isolation on Linux** — the demotion path
  requires privilege (root or CAP_SETUID); on macOS dev hosts the
  tests skip and production correctly refuses. Qualified on the
  Colab Linux root profile; a dedicated Linux VM with a service
  account remains the recommended production substrate.
- **macOS containment posture** — kernel refuses `NOTE_TRACK`
  (measured, Darwin 25.2/arm64), so marker/environ scanning is
  unavailable and ancestry covers only the normal tree. With the
  shipped seatbelt profile the escape class is impossible outright
  (no descendants can be created); without it,
  `require_process_containment` refuses startup. sandbox-exec is
  deprecated upstream — operators must qualify their own profile.
- **Network isolation** — wired as a required-when-requested control
  (`--worker-net-isolation`); enforcement is a Linux network
  namespace, or the seatbelt profile's `deny network*` on macOS.
- **cgroups** — the stronger descendant-containment mechanism
  (catches post-parent-death escapees). Recommended for Linux
  production; not yet wired.

## Open — donor-tree hygiene (documented, not blocking)

- `scripts/kvmem_eval/` and `references/` are donor research archives
  shipped for completeness. They contain developer-specific absolute
  paths (`/home/chaidi/...`) — the source audit scopes its dev-path
  check to governed source (minagi runtime, release tooling, configs)
  and treats these as out-of-contract legacy content.

## Open — carried forward

- **G7 real-model qualification (v16.4.7)** — real PEFT load → infer
  → kill → quarantine e2e continues to pass; the full
  activate/infer/cancel/terminate/rollback/restart/revoke/corrupt
  matrix on GPU under the demoted worker identity is the next gate.
- **G9 scientific credibility** — no learning-improvement claim; the
  Campaign 3A result stands as historical evidence only.
- **Windows** — restricted token + Job Object + ACL design is
  specified in the plan; not implemented (POSIX substrate only).
