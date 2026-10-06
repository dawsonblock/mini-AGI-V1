# RC9.2 logical KVMem session state

RC5 introduced the separation between two concepts that were coupled in RC4; RC6 preserves and hardens it:

- **detachable canonical session state**: host-side derived model memory that can
  be copied, snapshotted and cold-rehydrated;
- **executor affinity**: an ephemeral mount of that session on one mutable Qwen
  executor lineage.

`KvMemSessionState` contains canonical token history, embedding identity/rows,
M-RoPE coordinates and semantic pins. It intentionally excludes CUDA pointers,
KV-page/tier ownership, DeltaNet device state, MTP state and file descriptors.
It is therefore safe to detach from an executor, but remains a derived cache;
NEMO's WorkspaceLedger is still authoritative.

`KvMemSessionManager::Record` adds identity, workspace binding, version,
timestamps, cold-rehydrate count and an optional `executor_slot` mount to the
canonical state.

## Mount lifecycle

For a persistent session request:

1. the registry resolves the canonical session record;
2. `KvMemExecutorScheduler` acquires a slot using warm affinity, otherwise an
   empty/LRU-idle slot;
3. any evicted logical session is marked cold;
4. if the target was not already warm on that exact slot, QW3 rehydrates its
   canonical state through pressure prefill;
5. generation/append runs;
6. canonical state is committed and the scheduler lease is released while the
   successful session remains mounted for reuse.

If native execution fails, the lease is released as failed, its warm affinity is
discarded, and the physical slot is marked dirty. A later session request may
reuse that slot only through a forced cold-reset lease and successful canonical
rehydration.

## Current physical limit

The scheduler implementation is N-slot capable, and the session registry can
represent several mounted sessions on distinct slots. The current QW3 native
backend, however, still owns one actual shared mutable `QwenExecutor` lineage.
RC9.2 therefore certifies and accepts only:

```text
--kvmem-executor-slots 1
```

Values above one fail at engine load. This is deliberate: logical scheduler
capacity must not be confused with real GPU concurrency.

## Admission

The registry remains bounded by:

- `--kvmem-session-max`;
- `--kvmem-session-host-token-limit`;
- `--kvmem-session-host-byte-limit`.

Byte admission includes canonical token IDs, M-RoPE rows, semantic pins and
embedding payload estimates.

## Inspection

- `GET /v1/kvmem/sessions`
- `GET /v1/kvmem/sessions/<id>`
- `GET /v1/kvmem/scheduler`

Session receipts expose `executor_slot` in addition to status/version/tokens,
host bytes and cold-rehydrate count. Scheduler telemetry exposes warm hits,
cold mounts, current slot occupants, lease generations and backpressure
rejections.

None of this metadata grants authority. NEMO/Correct-Once must continue to use
current policy/grants for effects.


## Active executor resource admission

RC6 separates registry storage limits from active executor resource limits. See
`kvmem_resource_admission.md` for the VRAM/host/NVMe fail-fast envelope used
before native inference begins.
