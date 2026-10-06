# Integration audit

## Active donors

| Donor | Integration | Authority boundary |
|---|---|---|
| RC11 v11.7 | Sole C++/Metal source tree, unified Python execution/continual domains, persistent CAS artifact plane, signed production registry and deployment binding | Existing execution deployment authority remains authoritative |
| mini-AGI v12 | `minagi.egai`: three planes, evidence, beliefs, replay-prefix isolation, permanence proposal ladder, skill repository, improvement ledger, frozen-foundation protocol | Unsigned caller-built research bundles now fail closed |
| EGAI v3 RC8.2 | `egai`: benchmark/build/runner/evaluator/qualifier, artifact store, frozen backends and sequential experiment | Separate v3 promotion/applier/runtime registry removed; chain verified by v12 gate |
| RC10.25 | Target Mac preflight moved into `kvcontinual.execution.macos_qualification_preflight` | Preflight is a prerequisite, never a promotion authority or correctness certificate |
| RC10.11 | Missing invariants applied to RC11 persistent store and receipt verification; resource-budget contract ported | No donor control/promotion authority or competing journal imported |
| RC10.9 | Compared `src`, `include`, and `metal` against RC11.7 | All native paths are already present; identical or evolved RC11 versions retained |

`RC10_9_NATIVE_AUDIT.json` records per-file results. `SOURCE_PROVENANCE.json` records donor archive identities and their SHA-256 hashes. The older RC10.23, RC11.4, mini-AGI v7 and EGAI v2.2 trunks were not imported into the active package. The two original RC11.7 reference suites remain under `references/` and are excluded from installation. Retired v3 activation tests are preserved as text for provenance, and replaced by tests of the integrated signed chain and canonical authority.

## RC10.11 invariants

* CAS immutability/quarantine and SQLite snapshot lineage/floor modules were already in RC11.7 and were retained.
* Generation-fenced payload leases were missing from the persistent execution store. Ownership generations, acquire/renew/release checks and GC protection now use the same index transaction lock. Stale owners cannot regain leases after fencing or restart.
* The RC11 signed production transition journal already had ordered recovery and external anti-rollback anchors. It was not replaced. The artifact plane now commits its index mutation, monotonically sequenced mutation commitment and checkpoint together. Checkpoint/hash corruption rejects restart. CAS bytes written before a failed index transaction remain unreferenced and can be collected.
* RC10.11's admission controller depended on its old service/promotion/rollback objects. Importing it would have introduced a second service and authority graph. Its invariants were instead applied to the RC11 store: SQLite/checkpoint integrity, artifact decoding and identity closure, actual CAS byte budgets, block budgets and authoritative source-token counts. Admission failure chooses exact replay.
* Existing RC10-derived durability authority revocation code was present. Live execution Ed25519 receipts now also accept an operator-managed revocation callback. The research chain rejects revoked role keys at every signature check.
* Detached release signing was regenerated for the new source set using a separate Ed25519 release key. This key has no runtime promotion capability.

## Release and portability corrections

All installed distribution and module versions are 13.0.0, replacing the inconsistent RC10.24/RC10.25 and inherited authority v6 identities. RC11.7 is recorded as the execution lineage, not as a second installed version. Historical reference documents retain historical version names.

Old script imports and paths were updated to `src-python/kvcontinual.execution`. Legacy release manifests/build products were removed, and the release tooling and CI now use the consolidated layout.

macOS native validation exposed an inherited `fdatasync` call that cannot compile with this macOS toolchain. It now uses `fsync` on Apple platforms. Two native tests had Linux-only expectations: successful `posix_fadvise` page-cache dropping and physical preallocation. They now assert reported degradation when the platform lacks cache dropping and retain logical allocation, durable publication and content checks on macOS. Linux still requires the original physical-allocation/cache-drop assertions. The platform limitations remain explicit in the production limits document.

## Trust limits

Role-scoped signatures verify keys and object/content bindings. They cannot establish that key holders are different people or that an HTTP endpoint really serves immutable weights. Operators must keep benchmark, runner, evaluator, qualifier and promotion processes/keys independent. A caller with arbitrary Python code execution in the authority process is inside the trusted computing boundary. Source signatures authenticate this local build, not an upstream maintainer or real hardware qualification.

The standalone Metal parity test exposed unsupported shader `log1p`. Its replacement uses compensated `log(1+y)` with the existing large/small-value branches. The live Metal test now also compares gate outputs at six rounding/underflow boundaries against the independent double-precision oracle. CMake keeps native test assertions enabled in Release builds.
