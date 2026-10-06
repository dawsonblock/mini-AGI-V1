# RC11.7 release notes

RC11.7 closes the largest remaining host-side data-plane gap from RC11.6: model-derived execution artifacts can now survive process restart in a content-addressed, integrity-checked store.

The persistent store uses immutable SHA-256 CAS objects plus a WAL SQLite lookup index. The artifact codec is deterministic ZIP/JSON/NumPy with `allow_pickle=False`; corrupt CAS objects fail closed and are quarantined.

Apple Metal/HYPIC qualification with real model weights remains target-hardware work and is intentionally not claimed by this release.
