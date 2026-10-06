# RC5 KVMem session snapshots

RC4/RC5 provide restartable **derived** snapshots for canonical logical-session state.
They are intended to reduce replay work after a QW3 process restart. They are
not an authority database and they do not replace an external durable
WorkspaceLedger.

## Enable

```bash
qw3 serve ... \
  --kvmem-profile agent-fast \
  --kvmem-session-snapshot-dir /var/lib/qw3/session-snapshots
```

Optional host admission can additionally bound the aggregate logical-session
payload:

```bash
--kvmem-session-host-token-limit 8388608 \
--kvmem-session-host-byte-limit 8589934592
```

The byte charge includes canonical token IDs, M-RoPE positions, semantic pin
metadata and embedding payload estimates. It is a control-plane admission
estimate, not allocator telemetry for VRAM/RAM/NVMe tiers.

## API

```text
POST /v1/kvmem/sessions/<id>/snapshot
GET  /v1/kvmem/sessions/<id>/snapshot
POST /v1/kvmem/sessions/<id>/restore
```

A restore imports the canonical record as **cold**. The next append/finish uses
the normal cold-rehydrate path and reconstructs executor state from canonical
tokens before applying the new fragment.

## Integrity and compatibility

Snapshot files contain:

- format magic/version;
- session/workspace identity;
- session version and creation timestamp;
- canonical token IDs;
- serializable CPU embedding overrides;
- M-RoPE positions;
- semantic pinned spans;
- input embedding fingerprint;
- runtime/config compatibility fingerprint;
- payload checksum.

Load fails before registry import on malformed/truncated data, checksum failure,
ID mismatch or runtime-fingerprint mismatch.

The runtime fingerprint is deliberately a compatibility guard rather than a
cryptographic model identity. A production supervisor should separately pin the
exact model artifact hash and QW3 source/build manifest.

## Multimodal limitation

Opaque device-backed `InputEmbeddingStorage` cannot be serialized safely. If a
logical session still depends on device-only visual embedding storage, snapshot
creation fails closed. Rebuild that session from the authoritative workspace
history and original image inputs.

## Authority invariant

```text
WorkspaceLedger = authoritative trajectory truth
Snapshot        = rebuildable model-memory cache
KVMem tiers      = rebuildable execution cache
```

Restored text or session metadata must never recreate an expired/revoked action
grant. Authorization remains a current NEMO/Correct-Once decision.
