# NativeAdapter1 — v15.3

## Objective

Close the gap between a qualified adapter artifact and the QW3 native inference calculation without weakening the existing authority boundary.

## Bound flow

1. v15.2 produces a PASS-qualified `QualifiedAdapterSetCandidate`.
2. `NativeQW3AdapterCompiler` re-hashes the qualified source artifact bytes.
3. The compiler parses safetensors without executing artifact code.
4. NativeAdapter1 requires all LoRA A/B pairs to target the LM-head/output projection. Any unsupported target rejects the compilation.
5. A/B matrices are normalized to deterministic row-major FP32 payloads and individually SHA-256 hashed.
6. `manifest.json` binds the adapter-set root, raw model SHA-256, foundation digest, source adapter/set digests, qualified-candidate digest, tensor shapes, scale, filenames, byte lengths, and hashes.
7. The SHA-256 of the native manifest becomes `native_adapter_bundle_root`.
8. `ServedArtifactManifest` closes over both `adapter_set_root` and `native_adapter_bundle_root`.
9. QW3 re-verifies model identity, manifest identity, payload digests, tensor dimensions, and safe paths before upload.
10. The native executor applies `logits += scale * B * (A * normalized_hidden)`.
11. `/v1/runtime/state` reports the loaded native bundle identity. Governed requests must carry the matching identity.

## Fail-closed properties

- A qualified source file changed after qualification is rejected.
- A native A/B file changed after bundle compilation is rejected.
- A source adapter containing unsupported non-output LoRA tensors is rejected.
- Model SHA mismatch is rejected.
- Adapter-set root mismatch is rejected.
- Native bundle-root mismatch is rejected.
- A newly qualified adapter set does not inherit the previous native bundle root.
- Governed generation rejects stale or missing native bundle identity.

## Scope and nonclaims

NativeAdapter1 is an LM-head LoRA implementation, not full all-linear LoRA. The custom CUDA dense-FP32 low-rank primitive is present in source, but this build environment has no CUDA compiler or GPU, so CUDA compilation, numerical parity, performance, and real-weight inference are not claimed. Metal native LoRA is also not implemented. The next milestone should qualify the CUDA path on real QW3 weights and then extend the same content-bound mechanism to internal linear projections.
