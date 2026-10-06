# NEMO / QW3 KVMem adapter (RC9.2)

NEMO owns durable workspace history, authority, admission and effects. QW3 owns
model inference plus derived model-memory state.

RC9.2 preserves the detachable-session boundary and adds explicit physical executor ownership:

- canonical session state is detachable host-side derived memory;
- executor slots are ephemeral compute affinity;
- snapshots are restart caches, not authoritative history;
- recalled memory or scheduler state can never grant an effect.

`agent-safe` / `selected-replay` still uses full-transcript one-shot `infer()`.
`agent-fast` / `query-replay` uses explicit `sessionStep(start|append|finish)`;
`start` and `append` require `maxTokens=0`.

Each persistent session is bound to a NEMO `workspaceId`. The session receipt
now exposes `executor_slot`. `inspectScheduler()` reads
`GET /v1/kvmem/scheduler` so NEMO can record warm hits/cold mounts and detect
backpressure rather than infer them from latency.

RC9.2's scheduler and physical-runtime APIs are N-slot structured, but the native runtime certifies one
physical executor slot. `preflight()` rejects a runtime whose configured slot
count exceeds its certified slot count.

`snapshotWorkspace()` / `restoreWorkspace()` remain derived-cache operations.
If a snapshot is missing, corrupt or incompatible, NEMO must reconstruct from
its authoritative WorkspaceLedger.

## RC9.2 executor ownership

The adapter now exposes `inspectExecutors()` and preflight verifies `executor-slot-runtime-v2`. Scheduler affinity and physical runtime ownership are inspected separately; neither is an authority source.
