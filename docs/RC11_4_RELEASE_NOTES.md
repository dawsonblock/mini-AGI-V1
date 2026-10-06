# RC11.4 — Durable authority and execution-bound qualification

RC11.4 turns the production transition ledger into the recoverable authority for deployment state and closes several crash-consistency and evidence-portability gaps left in RC11.3.

## Durable transition authority

New production transitions use `rc11.production-transition.v2`. In addition to the hash link and monotonic generation, each record binds:

- stable registry instance ID;
- unique event ID;
- action (`promote` or `rollback`);
- semantic previous-candidate continuity;
- current adapter and candidate-manifest digests;
- cache namespace digest;
- qualification-bundle digest when present;
- exact embedded production-binding receipt and its digest;
- event timestamp;
- optional Ed25519 transition signature.

The embedded production-binding receipt is important. A mutable per-candidate receipt file is no longer needed to reconstruct the current authority state after a crash. Receipt files under `promoted/<candidate>/promotion_receipts/` are durable mirrors indexed by event ID.

## Crash recovery

Registry JSON writes now fsync the temporary file, atomically replace the destination, and fsync the parent directory. Candidate publication and new promoted-candidate publication use pending directories followed by durable renames.

If a crash occurs after the transition ledger append but before `meta/production.json` is updated, recover with:

```bash
python3 scripts/rc11/recover_registry.py \
  --registry data/registry \
  --public-key data/authority/promotion_ed25519.public \
  --anchor /trusted/location/rc11-anchor.json
```

`recover_current()` re-verifies the complete ledger, promoted bytes, manifest, qualification evidence, namespace and embedded binding receipt before rebuilding the derived snapshot.

## External anti-rollback anchor

Local signatures cannot detect replay of an entire older but internally valid filesystem snapshot. RC11.4 exposes a compact external anchor containing:

- registry ID;
- generation;
- transition-record digest;
- event ID.

Export it after an authorized state transition and store it outside the registry:

```bash
python3 scripts/rc11/export_authority_anchor.py \
  --registry data/registry \
  --public-key data/authority/promotion_ed25519.public \
  --out /trusted/location/rc11-anchor.json
```

`verify_authority_chain.py --anchor ...` and `recover_registry.py --anchor ...` enforce both the monotonic generation floor and exact expected ledger head.

## Qualification bundle schema v2

New Apple-Silicon qualification bundles additionally bind:

- exact `ExecutionIdentity` digest;
- exact runtime/release-tree digest;
- a canonical qualification-context digest.

This prevents a passing hardware/oracle result from being transplanted onto another execution implementation or runtime build. RC11.3 schema-v1 bundles remain readable for migration, but new bundles are generated as schema v2.

A registry may enable `require_qualification_bundle=True`. In that mode, promotion qualification must include a passing bundle, and candidate model/tokenizer/execution identity can be cross-checked against it.

## Deterministic evidence ZIP

The target-Mac qualification can be exported as a self-contained deterministic evidence package:

```bash
python3 scripts/rc11/build_qualification_evidence.py \
  --bundle /tmp/qualification-bundle.json \
  --probe /tmp/mac-probe.json \
  --capture-manifest /tmp/capture-manifest.json \
  --observations /tmp/oracle-observations.jsonl \
  --kernel /path/to/qualified-metal-library \
  --out /tmp/rc11-qualification-evidence.zip

python3 scripts/rc11/verify_qualification_evidence.py /tmp/rc11-qualification-evidence.zip
```

The evidence ZIP contains the exact kernel binary and canonical probe/capture/oracle inputs, with an internal manifest and deterministic timestamps.

## Still not claimed

RC11.4 still does not claim that the real Qwen hybrid HYPIC/Metal path is production-qualified on Apple Silicon. Exact selected replay remains authoritative until the real-weight differential gates pass on the target hardware.
