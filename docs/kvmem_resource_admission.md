# RC9.2 KVMem active resource admission

Logical-session storage and active executor memory are different resources.
`KvMemSessionManager` bounds detachable canonical host state across all sessions;
RC6 adds `KvMemResourceAdmission` for the memory envelope consumed while an
executor lineage is active.

The admission dimensions are:

- VRAM bytes;
- host-tier bytes;
- NVMe-tier bytes;
- maximum in-flight executor reservations.

A zero explicit byte cap means the CUDA load path derives the current single-slot
capacity from `QwenExecutor::kvmem_tier_usage()`. Explicit caps are fail-closed:
if a configured cap is smaller than the footprint of one certified executor slot,
model load fails instead of allowing a request to discover the mismatch after it
starts mutating state.

The current native backend certifies one physical executor slot, so RC9.2 resource
admission is primarily a correctness and observability boundary today. It is
already N-slot compatible: once independent executor instances exist, the same
manager can prevent aggregate slot footprints from oversubscribing a process or
worker allocation.

`GET /v1/kvmem/resources` returns capacities, current use, high-water marks,
admission count and rejection count. A resource rejection is distinct from a
scheduler rejection: scheduler rejection means no executor lineage is available;
resource rejection means an executor could be scheduled but its declared active
memory footprint does not fit the configured envelope.

Resource envelopes are not authority or durable state. NEMO remains responsible
for request admission/authority, while its WorkspaceLedger remains the durable
source of truth for reconstructing KVMem state.
