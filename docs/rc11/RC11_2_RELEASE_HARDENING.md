# RC11.2 release hardening

RC11.2 does not expand the claim boundary for native Metal HYPIC. Exact selected-token replay remains authoritative until differential hardware qualification passes on the target Apple Silicon machine.

New release controls:

- manifest v3 binds file bytes, sizes, SHA-256 digests, and POSIX mode bits;
- symlinks are forbidden in signed/released source trees;
- deterministic ZIP construction fixes archive timestamps and preserves file modes;
- `verify_authority_chain.py` checks the complete signed current-production adapter chain from a public Ed25519 key;
- `macos_qualification.py` records a canonical Apple-Silicon hardware/toolchain fingerprint for binding into later execution-artifact qualification receipts.

The hardware fingerprint is provenance, not correctness evidence. A production HYPIC receipt still requires real model/tensor differential evidence and the existing oracle gates.
