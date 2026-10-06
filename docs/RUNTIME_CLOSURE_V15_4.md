# RuntimeClosure1 — v15.4

## Purpose

v15.3 bound requests to a StateEpoch and a native adapter bundle, but the native server still trusted several state identities passed by the launcher. RuntimeClosure1 turns physically observable deployment state into measured evidence before serving begins.

## Native startup invariant

For governed serving, QW3 now requires all component roots needed to reproduce the served-artifact root. Before constructing the HTTP serving surface it:

1. hashes the loaded model file, or deterministically hashes every regular file in a model directory;
2. resolves and hashes the **currently executing QW3 binary**;
3. hashes the attached sealed KVMem `manifest.json`, or derives the zero root when no archive is attached;
4. combines those measured values with the authorized tokenizer, adapter-set, retrieval-policy, skill-policy and optional native-adapter roots;
5. recomputes the canonical `mini-agi-v15-served-artifact-root-v1` SHA-256;
6. refuses startup if the recomputed root differs from the authorized `state_artifact_root`.

The runtime then reports a `mini-agi-v15.4-runtime-closure-v1` digest on `/v1/runtime/state` together with the measured roots.

## Python deployment helper

`MeasuredRuntimeClosure.from_files()` hashes the intended deployment inputs before launch and checks them against `ServedArtifactManifest`. `MeasuredGovernedServingContract` requires the native server to report the same measurements and closure digest before request headers are emitted.

## Directory identity

Model directories use a deterministic tree digest over canonically sorted relative paths, file sizes and per-file SHA-256 values. Symlinks are rejected. Python and C++ tests share a fixed digest oracle for this format.

## Nonclaims

RuntimeClosure1 does not independently materialize tokenizer, retrieval-policy or skill-policy bytes inside QW3; those identities are currently aggregate-bound inputs. It does not add all-linear LoRA, prove real-weight task improvement, qualify CUDA, establish long-horizon continual-learning stability, or provide external reproduction.
