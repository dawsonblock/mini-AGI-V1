# Unresolved production qualification limits

Production acceleration is not qualified. No uploaded real-weight Qwen tensors, hidden target datasets, independent target-host qualification signatures or production deployment keys were available. No passing target certificate was generated. The canonical `GovernedSystem` therefore performs exact selected replay even when FAST reconstruction is requested.

Required evidence remains: real Qwen tensors → native capture → correct causal Conv1D boundary → GDN reconstruction → attention relocation → live Metal model execution → exact-replay differential across the preregistered suite → independently signed execution qualification → matching release/host/backend/deployment binding. Portable native tests, oracle tests, model metadata and Mac preflight success do not replace this chain.

The full 25-test native build used CUDA OFF and Metal GDN OFF. A separate Metal-enabled build ran the standalone GDN kernel/summary/chain parity test on the actual Apple GPU, with assertions enabled. That test uses synthetic tensors, not a full real-weight Qwen checkpoint. CUDA and full real-checkpoint model execution were not exercised. The local preflight report records actual host/toolchain capabilities but confers no acceleration authority.

The first experiment freezes the foundation and permits only non-neural L0–L5 mechanisms. L6–L9 remain proposals. The procedural sequential harness measures a narrow class of verified-repair learning and should not be read as proof of general lifelong learning.

The local llama.cpp research backend rehashes its checkpoint when the model guard checks identity. The HTTP research backend has only an independently supplied model manifest: it cannot remotely prove served weight immutability, and must not be treated as a cryptographically attested frozen model without separate server-side evidence. Neither backend is an alternate production substrate.

macOS lacks the Linux `posix_fadvise` page-cache-drop path. Native I/O reports degradation and continues with kernel page caching. The current macOS arena preallocation is logical; it does not guarantee physical disk reservation. Production operators still need measured memory/storage admission on their target hardware. Power-loss behavior beyond SQLite/fsync contracts and multiprocess production stress under real workloads have not been independently qualified.

The artifact mutation chain/checkpoint detects local corruption and ordering errors. Resistance to rollback of the entire mutable directory requires the existing externally protected production anchor and snapshot floor. Revocation requires fresh externally managed revocation state. The one-shot research consumption database is likewise inside this operator-protected state boundary.

The new release signature verifies the packaged source/metadata bytes using a newly generated local key. An upstream maintainer has not countersigned this build. The CycloneDX SBOM lists source components and declared requirement ranges; dependency resolution is not pinned into a production lockfile and optional ML toolchains have not been installed or qualified as part of this build.
