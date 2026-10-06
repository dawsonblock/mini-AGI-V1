# RC11 Architecture

## Validity domains

RC11 deliberately separates four kinds of state:

1. **Execution memory** reconstructs finite neural context. It may contain attention KV, recurrent transition summaries, convolution boundary state and seam metadata.
2. **Persistent memory** stores durable episodic/semantic information and retrieval metadata.
3. **Trainable adapters** change behavior without mutating the frozen base model.
4. **Promotion authority** decides which learned or derived artifacts may enter production.

A failure in one domain must not silently grant authority in another.

## Data flow

```text
request / environment
        |
        v
selection + context planner
        |
        v
immutable source plane
(content digest = tokens + tokenizer + normalization)
        |
   +----+--------------------+
   |                         |
   v                         v
exact replay            derived execution artifacts
                         KV / GDN / conv / seam
                              |
                       qualification evidence
                              |
                 +------------+------------+
                 |                         |
                 v                         v
              exact                     HYPIC /
              oracle                    candidate fast path
                 |                         |
                 +------------+------------+
                              |
                         model execution
                              |
                  experience / persistent memory
                              |
                       candidate training
                              |
                 evaluate -> qualify -> sign
                              |
                            promote
```

## Source authority

Random segment IDs are aliases, not identities. The authoritative identity is `SourceSegment.content_digest`, which includes token IDs, tokenizer digest and normalization digest. Segment IDs are append-only.

Every production execution artifact should bind this source digest. RC11's artifact store can require the digest on lookup and rejects replacement of a bound alias with a different source digest.

## Hybrid execution-memory policy

For arbitrary selected segments, the preferred candidate path is fixed-seam HYPIC-style reconstruction when—and only when—the backend advertises all required capabilities and the artifacts are qualified. Otherwise RC11 falls back to exact selected replay.

A single-state-initialization / LinearKV-style path remains experimental and must be selected by measured qualification, not assumption.

## RoPE

RoPE geometry is explicit through `RopeSpec`:

- full head dimension;
- rotary sub-dimension;
- split-half or interleaved pairing;
- theta/scaling identity.

This avoids treating a self-consistent reference transform as proof of compatibility with the actual model implementation.

## Storage

Dense affine `(T,Z)` summaries are kept for correctness and oracle work. `recurrent.storage_policy` estimates their footprint and can fail closed above a configured budget. A structured/factored representation should replace dense storage before large-scale production deployment.

## Production authority state

The production transition ledger is the authoritative deployment history. `meta/production.json` is a derived snapshot used for fast reads, not the root of authority.

Each RC11.4 transition-v2 record is hash-linked to the previous record and binds the registry instance, monotonic generation, semantic predecessor candidate, current candidate artifacts, cache namespace, qualification bundle, and exact production-binding receipt. In signed mode the complete transition body is Ed25519-authorized.

This ordering intentionally permits crash recovery:

1. prepare/verify immutable promoted artifacts;
2. construct and sign the production binding;
3. durably append the transition record;
4. mirror the versioned binding receipt;
5. atomically update the derived production snapshot.

A crash after step 3 can be recovered from the verified ledger head. A crash before step 3 has not changed production authority.

Whole-registry rollback is outside what a local append-only file can prove. RC11.4 therefore supports an external anchor containing the registry ID, generation and transition-tail digest. Operators that need anti-rollback guarantees must persist that anchor independently or replace it with a hardware/remote monotonic witness.
