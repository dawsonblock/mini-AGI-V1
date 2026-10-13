# Mini-AGI Worker Threat Model (v16.4.8)

## Trust boundaries

```
Trusted supervisor            restricted launch broker         jailed worker
admission · signatures        LaunchWorkerRequest (digests+    Qwen/HF/PEFT
revocations · SQLite          nonce only — no argv/env/shell)  model code
        │                             │                              │
        └─ authorizes ────► systemd transient unit ──► NsJail ───────┘
                          cgroup v2 ownership          ns/fs/net/syscall
```

- The **supervisor** decides *whether* a model may execute; it never
  hands the worker authority material.
- The **broker** decides *how* it executes; it accepts only digests
  of already-approved material plus a replay nonce. It cannot sign,
  cannot take argv/env/mounts from the caller, and runs only the
  pinned NsJail binary.
- **systemd/cgroup v2** owns the process tree — membership is kernel
  state; `cgroup.populated==0` is the termination proof.
- **NsJail** is the syscall-facing boundary — mount/pid/net/ipc/uts/
  user namespaces, read-only binds, no_new_privs, rlimits, seccomp.
- The **worker** is untrusted model code. It may attempt filesystem
  reads, network egress, fork bombs, setsid escapes, SIGTERM
  ignoring, env scrubbing, and parent suicide. None may weaken its
  containment.

## What the worker can do (by design)

Read the approved runtime closure, read admitted artifact dirs
(identical paths, read-only), write inside `/work` (broker-owned
scratch), speak the strict framed-JSON protocol on its unit channel,
spawn children bounded by `TasksMax`/RLIMIT_NPROC.

## What the worker cannot do

See supervisor authority paths (`denied_paths` mount screening +
fresh mount namespace), reach the network (`clone_newnet` +
`iface_no_lo`), see host processes (pid ns), hold capabilities
(`keep_caps: false` + `NoNewPrivileges`), escape its cgroup
(kernel membership is not escapable by setsid/reparent/env-scrub),
inherit secrets (env allowlist + credential-name refusal).

## Residual risks (documented, not waived)

- NsJail is not a VM. A kernel or GPU-driver vulnerability inside
  the syscall surface is out of scope for this boundary; arbitrary
  AI-generated executable code needs stronger (VM) isolation.
- seccomp is the NsJail `default` policy in
  `linux-production-offline-v1`; a tighter kafel policy is a
  deployment knob, not yet qualified.
- Rootless (`systemd-run --user`) mapping differs from rootful;
  qualify the same mode that will run in production.
