# RC11 Security and Promotion Authority

## Threats explicitly addressed

RC11 assumes local files can be modified accidentally or by an untrusted process between training, qualification, promotion and rollback. Path identity is never sufficient.

Protected transitions are content-bound:

`candidate bytes -> candidate manifest -> qualification evidence -> staged cache namespace -> production-binding receipt -> signed transition ledger -> derived production snapshot`.

The registry re-hashes candidate bytes before qualification and promotion, confines candidate/artifact paths to registry roots, verifies the promoted copy, and re-verifies the signed chain before a signed rollback. RC11.4 also makes qualification records immutable after creation and freezes a candidate's cache namespace after first promotion.

## Ed25519 separation

`Ed25519ReceiptSigner` owns the private key. `Ed25519ReceiptVerifier` contains only the public key. A serving/runtime process should normally receive only the verifier.

Signed mode is enabled with:

```python
registry = AdapterRegistry(root, require_signed_promotions=True)
```

The authority process signs qualification and production binding:

```python
registry.write_qualification(record, signer=authority_signer)
registry.stage_cache_namespace(candidate_id, namespace_digest)
registry.promote(candidate_id, verifier=trusted_verifier, signer=authority_signer)
```

The second signature is important: a valid qualification does not authorize an arbitrary cache namespace. The production-binding receipt binds the exact candidate digest, manifest digest, qualification payload, optional qualification-bundle digest and staged namespace. New transition-v2 records embed that exact binding receipt and sign the complete state transition.

## Rollback and active-state verification

In signed mode:

```python
registry.verify_current(verifier=trusted_verifier)
registry.rollback(verifier=trusted_verifier, signer=authority_signer)
```

re-check the promoted bytes, manifest, qualification receipt, namespace, production-binding receipt and signed transition head. Rollback is itself a newly authorized transition. A verifier-only serving process cannot mutate production state.

`production.json` is a derived snapshot. The append-only transition ledger is the recoverable authority, and `recover_current()` can rebuild a stale or missing snapshot after verifying the ledger tail and promoted artifacts.

Software-only receipts do not provide a hardware monotonic counter. An attacker able to atomically restore an entire previously valid filesystem snapshot can still replay a whole old registry. RC11.4 therefore supports an **external transition anchor** (registry ID + generation + tail digest). Store the anchor outside the registry and provide it to verification/recovery to enforce a monotonic floor.

## Path confinement

Candidate identifiers are restricted to safe path components. Manifest `artifact_relpath` values must remain inside the candidate directory and may not traverse symlinks. These checks prevent registry metadata tampering from becoming arbitrary filesystem reads/writes.

## Signed JSON rules

Signed receipt JSON is deterministic and rejects non-finite numbers (`NaN`, `Infinity`, `-Infinity`). Ed25519 signatures use strict Base64 decoding and must be exactly 64 bytes.

## Execution-artifact receipts

`artifact_receipts.py` defines a signed receipt body capable of binding:

- source content digest;
- execution identity digest;
- base-model weights digest;
- tokenizer digest;
- capture manifest;
- artifact digest;
- algorithm/seam width;
- numerical tolerance;
- backend identity;
- hardware fingerprint;
- kernel build digest;
- oracle result digest;
- policy generation.

An optimization should not become authoritative merely because local metadata says `oracle_qualified=True`. Production wiring should verify the signature and compare all bound fields against the live artifact/runtime identity.

## Private-key handling

No private promotion key is included in this release. Generate one locally with `scripts/rc11/generate_promotion_key.py`. Restrict the private key and distribute only the public verifier to serving processes.


## RC11.4 qualification evidence policy

Qualification bundle schema v2 binds model weights, tokenizer, execution identity, runtime/release build, hardware/toolchain fingerprint, capture manifest, compiled kernel, oracle observations and thresholds. `AdapterRegistry(..., require_qualification_bundle=True)` makes a passing bundle mandatory before a qualification record can authorize promotion.

The deterministic evidence ZIP scripts package the exact qualified kernel and canonical evidence inputs so the bundle can be audited later without relying on mutable working directories.
