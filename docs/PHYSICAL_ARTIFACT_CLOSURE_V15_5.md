# v15.5 Physical Artifact Closure

v15.5 closes the remaining declared-root gap in governed QW3 serving. A nonzero tokenizer, adapter-set, retrieval-policy, or skill-policy identity must now correspond to physical bytes that QW3 can independently measure at startup. The aggregate served-artifact root is recomputed from those measured identities together with the already measured model, runtime executable, KVMem archive, and native-adapter identity.

## Security invariant

A governed server may become ready only when:

`authorized component roots == roots independently measured from the deployed bytes`

and the canonical aggregate root recomputed by QW3 equals the `ServedArtifactManifest.artifact_root`.

For optional adapter-set/retrieval/skill artifacts, 64 zeroes means the artifact is absent. Supplying bytes for a zero-root artifact is rejected. A nonzero root without an artifact path is also rejected.

## Tokenizer identity

Tokenizer identity is derived from the tokenizer inputs QW3 actually uses:

- GGUF: the tokenizer is embedded in the GGUF, so v15.5 conservatively uses the whole GGUF SHA-256 as the tokenizer identity.
- HF model directory: QW3 hashes `tokenizer.json`, `config.json`, and `tokenizer_config.json` when present; if the latter is absent and `generation_config.json` exists, that file is included instead.

Symlink substitution is rejected on both the Python and native paths.

## Materialized canonical governance objects

`materialize_canonical_artifact()` writes the canonical bytes already used by the governance digest implementation. Therefore the file SHA-256 is exactly the hex portion of the governance `sha256:<hex>` digest. This is suitable for `SkillPolicyBundle`, `AdapterSetBundle`, and other canonical governance objects that need a physical serving representation.

`build_physically_bound_served_manifest()` derives all served roots from deployed bytes rather than requiring operators to copy hashes manually.

## Native launch additions

When the corresponding root is nonzero, governed QW3 requires:

- `--state-adapter-set-artifact PATH`
- `--state-retrieval-policy-artifact PATH`
- `--state-skill-policy-artifact PATH`

The runtime state endpoint reports measured tokenizer, adapter-set, retrieval-policy, and skill-policy roots. Generation requests carry these component identities in addition to epoch, manifest, aggregate artifact root, and adapter identities.

## Validation

For the release source tree:

- 285/285 active Python tests passed with third-party pytest plugin autoload disabled.
- 28/28 portable native C++ tests passed.
- Host-native QW3 compiled with CUDA disabled.
- Python/C++ parity is covered for tokenizer identity, canonical runtime closure digest, model-directory identity, and served-artifact root.

## Nonclaims

v15.5 does not claim CUDA qualification, real-weight unseen-task improvement, all-linear native LoRA, native execution of retrieval/SkillIR policy bytes inside QW3, 10k–100k episode stability, or independent reproduction. Retrieval and skill artifacts are physically bound to the served state; their behavioral execution remains above the native token engine.
