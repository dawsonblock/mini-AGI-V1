# RC10 implementation status

## Implemented and host-qualified in this archive

- Full RC9.2 native runtime retained.
- Strict native execution-cache identity primitive.
- Native/reference affine `(T,Z)` composition.
- Exact recurrent-layer-set validation.
- Native adaptive repair policy data structures.
- `KvMemBlock` metadata hooks for recurrent-summary slots/epochs/anchors.
- Hardened continual memory/control-plane package.
- Persistent and temporal SQLite memory.
- Transactional fact supersession.
- Candidate adapter digest sealing.
- Qualification-to-candidate digest binding.
- Rehash-before-promotion protection.
- Path-safe registry IDs.
- Atomic promotion metadata writes.
- Admin/control API separation.
- Non-duplicating weighted replay samples.

## Retained exact fallback

RC9.2 selected replay remains the authoritative coherent rebuild mode for
actual hybrid Qwen execution when sparse retrieval changes the historical
sequence.

## Model/GPU work still required

- Native GDN segment summary extraction.
- Structured transition storage format and size/performance characterization.
- Full-stack seam replay.
- Suffix replay and exact checkpoint hierarchy.
- Native LoRA kernels/application.
- GPU and model-level fidelity benchmarks.

This distinction is mandatory: the archive is a materially upgraded and
host-tested codebase, but it does not claim unexecuted CUDA/model integration as
finished.
