# mini-AGI v15 Integration Release

This release intentionally does **not** add a new promotion authority or claim AGI/RSI.
It closes the previously missing native/governance serving seam.

## New invariant

A generation request is eligible to run only when the QW3 process reports exactly the
StateEpoch digest, served-artifact manifest digest, and aggregate artifact root that the
caller holds under its request lease.

QW3 exposes `/v1/runtime/state` and, when launched with the three governance flags below,
rejects `/v1/chat/completions`, `/v1/completions`, and `/v1/messages` unless request headers
match the loaded epoch and manifest digest.

```text
qw3 serve --model MODEL \
  --state-epoch-id <sha256> \
  --state-manifest-digest <sha256> \
  --state-artifact-root <sha256>
```

The Python `minagi.integration.GovernedServingContract` independently fetches native loaded
state, compares it to the lease-bound `ServedArtifactManifest`, and only then emits request
headers. This creates two checks: the caller verifies the native process, and the native
process verifies the caller's epoch/manifest identity.

## Artifact manifest

The serving manifest binds foundation, tokenizer, KV archive, adapter set, retrieval policy,
skill policy, and runtime binary digests. Changing any one changes the aggregate artifact
root and manifest digest.

## Explicit non-claims

This does not yet prove useful continual learning, adapter training, long-horizon stability,
or external reproduction. It is an integration/serving-closure release: the governance
StateEpoch can now be mechanically tied to what the inference process says it loaded.
