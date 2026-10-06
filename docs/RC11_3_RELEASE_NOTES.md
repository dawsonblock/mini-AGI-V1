# RC11.3 — Authority-chain and Mac qualification hardening

RC11.3 closes a remaining authority gap in RC11.2: rollback is now a new state transition that must be explicitly authorized rather than a pointer swap to an older, previously signed candidate.

## Production state transition chain

Every promotion and rollback appends a transition record containing:

- monotonically increasing generation;
- unique event ID;
- previous transition digest;
- action (`promote` or `rollback`);
- current and previous candidate IDs;
- adapter and candidate-manifest digests;
- cache-namespace digest;
- production-binding receipt digest;
- event timestamp;
- optional Ed25519 authority receipt.

In signed mode, both promotion and rollback require current signing authority. `verify_current()` validates the complete transition chain and requires the production state to match the ledger head. Replaying an older `production.json` while retaining a newer ledger therefore fails closed.

This is not a hardware monotonic counter. An attacker capable of atomically rolling back the entire registry, ledger, and trusted key state still requires an external anchor (TPM/Secure Enclave/remote transparency log) for cryptographic rollback detection across total-storage snapshots. RC11.3 does not claim otherwise.

## Deterministic Mac qualification bundle

`scripts/rc11/build_qualification_bundle.py` creates one canonical evidence object from:

- `macos_qualification.py` hardware/toolchain fingerprint;
- live-model capture manifest;
- exact-vs-candidate oracle observations;
- model weights digest;
- tokenizer digest;
- compiled Metal/kernel binary digest;
- qualification thresholds and summary.

RC11.4 extended the current builder with required execution-identity and runtime-build bindings. The historical RC11.3 schema-v1 format remains readable, but the current CLI emits schema v2.

The builder requires a real Apple-Silicon probe with Metal and PyTorch MPS available and fails closed when the measured qualification thresholds are not met.

Example target-Mac flow:

```bash
python3 scripts/rc11/macos_qualification.py --out /tmp/mac-probe.json
python3 scripts/rc11/build_qualification_bundle.py \
  --probe /tmp/mac-probe.json \
  --capture-manifest /tmp/capture-manifest.json \
  --observations /tmp/oracle-observations.jsonl \
  --model-weights-digest sha256:<64-hex> \
  --tokenizer-digest sha256:<64-hex> \
  --execution-identity-digest sha256:<64-hex> \
  --runtime-build-digest sha256:<64-hex> \
  --kernel /path/to/qualified-metal-library \
  --out /tmp/qualification-bundle.json
```

Exact selected replay remains the production authority until that bundle passes on the actual target Mac and the accelerated HYPIC/Metal backend is wired to consume a valid signed artifact receipt referencing the qualified evidence.
