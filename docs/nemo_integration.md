# External NEMO integration boundary

QW3/KVMem should be deployed as a model-memory service beside NEMO, not linked
into NEMO's admission or authority kernel.

```text
Agent controller
  |-- durable WorkspaceLedger
  |-- QW3/KVMem HTTP model runtime
  `-- NEMO -> Correct-Once -> effect providers
```

## Startup contract

RC4 exposes:

- `GET /healthz` — process liveness;
- `GET /readyz` — model/runtime readiness plus KVMem status;
- `GET /v1/kvmem/status` — memory-plane configuration, coherence contract, and
  logical-session capacity/status;
- `GET /v1/kvmem/sessions` and `GET/DELETE /v1/kvmem/sessions/<id>` — derived
  logical session inspection/eviction;
- `GET/POST /v1/kvmem/sessions/<id>/snapshot` and
  `POST /v1/kvmem/sessions/<id>/restore` — versioned restart-cache operations
  when `--kvmem-session-snapshot-dir` is configured.

For a correctness-oriented one-shot agent integration, launch QW3 with
`--kvmem-profile agent-safe`. The profile requires Mean-K, strict scorer
behavior, immutable source K, message semantic expansion, query replay, and
`selected-replay` state coherence.

For lower latency and incremental sessions, `agent-fast` uses query replay but
explicitly reports that KVMI-012 still applies above budget.

## Request receipts

Terminal Chat Completion responses contain a `kvmem` extension. NEMO should log
that receipt beside its model/admission/effect receipts. It may use the receipt
to reject a model result that violated a configured memory contract, but the
receipt is not an authorization object.

`selected-replay` is currently one-shot only. A NEMO workspace using that
profile sends its authoritative transcript again (or reconstructs it from the
WorkspaceLedger); QW3 performs sparse retrieval and then densely rebuilds the
selected source context for the answer. Do not manufacture a persistent QW3
session ID for that profile.

For `agent-fast`, NEMO uses explicit `start|append|finish` operations and may
maintain an explicit mapping:

```text
NEMO workspaceId
    -> QW3 session id
    -> model/runtime fingerprint
    -> WorkspaceLedger sequence
```

The mapping is disposable. The WorkspaceLedger remains authoritative. RC4 binds
the optional `kvmem_workspace_id` to the session and returns a `kvmem_session`
receipt. A cold rehydrate is observable through `cold_rehydrates`; it is not a
durability or authorization event.

Only one executor lineage is hot. Other logical sessions are reconstructed from
canonical host tokens when selected again. NEMO must not use
`kvmem_prefill_window=keep_selected` across a cold switch; force a new selection
or use pressure mode.

## Authority boundary

Retrieved KVMem content is model evidence only. It must never mint, extend, or
restore an authority grant. Current authorization is evaluated independently by
NEMO/Correct-Once from current principals, policy, capability identity, and
runtime state.

The normal GPU/RAM/NVMe KVMem tier is not durable workspace truth. Persist raw
user/tool/effect events outside QW3 so KVMem can be rebuilt after cache loss,
process restart, model upgrade, or incompatible snapshot.


## RC4 restart flow

For `agent-fast` persistent sessions, NEMO may request a QW3 snapshot after a
committed workspace checkpoint. After a QW3 restart, NEMO explicitly restores
the snapshot, verifies the workspace binding/session receipt, and continues. A
restored record is cold and must rehydrate before use. If restore fails because
of corruption or a runtime fingerprint mismatch, discard the cache and rebuild
from the authoritative WorkspaceLedger. Snapshot success never confers or
restores action authority.

## RC9.2 physical executor inspection

NEMO should inspect `/v1/kvmem/executors` during runtime diagnostics. The scheduler reports logical affinity; the executor-pool endpoint reports concrete mutable runtime ownership. A scheduler warm hit is not sufficient evidence of warm physical state unless the physical runtime reports the same mounted session and is not dirty.


## RC9 physical ownership

The native QW3 executor reported through `/v1/kvmem/executors` is now owned by
`KvMemPhysicalExecutorPool`. NEMO should continue treating executor slots as ephemeral compute
placement only. Workspace authority and durable history remain outside QW3. The current native
runtime certifies one CUDA slot; NEMO preflight must continue rejecting configurations where
configured slots exceed certified slots.
