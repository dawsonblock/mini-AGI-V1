# v16.4.5 Repair Report — Security and Evidence Closure

Base: `mini-AGI-V1-v16.4.4-Verified-Runtime-Closure.zip`
(sha256 `ad5993bd1e1e56f4f60c822cc95934b36710a8425d9978f905f095c101a0e7d5`,
preserved unmodified). Architecture preserved; the three release-blocking
trust-boundary defects are closed.

## SEC-401 — Unsafe worker-process IPC (closed)

`worker_backend.py` used `pickle.dumps`/`pickle.load` in both directions;
`_reader()` deserialized worker-controlled bytes inside the supervisor —
a compromised worker could execute code in the more privileged process.

Repair:

- `runtime/worker_protocol.py` — strict framed JSON v2 codec. 4-byte
  big-endian length prefix, maximum frame size checked *before*
  allocation, closed message schema (`protocol`, `request_id`, `type`,
  bounded `payload`), announced payload length and digest verification,
  bounded result size, bounded request concurrency. No pickle path remains.
- `runtime/worker_backend.py` — rewritten on the codec. Worker protocol
  violations subclass `WorkerDied`: malformed, truncated, oversized,
  unexpected, or hostile bytes kill the worker channel and never reach
  supervisor logic.
- `runtime/worker_isolation.py` — restricted launch: explicitly
  constructed environment (credential names matching KEY/TOKEN/SECRET/
  PASSWORD/CREDENTIAL are not propagated), private 0700 scratch dir
  bound to TMP/TMPDIR/TEMP, optional POSIX rlimits, optional uid/gid
  demotion, separate process group so termination covers descendants.
- `v161/immutable_snapshot.py` — `MeasuredSnapshot` refuses pickling.
  Cross-process transport is a descriptor resolved against staged files;
  the receiving process re-measures the bytes and compares them to the
  authorized digests. Missing, changed, unsafe, or mismatched trees fail.

## SEC-402 — Revoked promotion could be cold-restored (closed)

`recovery_manager.py` issued a fresh admission grant from a durable
pointer and an earlier decision digest without checking that decision
against current revocation evidence, and generated substitute digests
when historical authority fields were absent.

Repair:

- `runtime/authority_store.py` — retains the signed authority documents
  (plan, qualification, promotion decision, runtime manifest) presented
  at first admission; missing documents are missing evidence, never a
  prompt for substitutes.
- `security/restoration_authority.py` — `RestorationAuthorizationVerifier`
  re-verifies the complete historical chain under current policy:
  promotion signature valid, binding digests equal the restoration
  target, backend still qualified, policy epoch current, revocation
  snapshot valid/monotonic/not rolled back, artifact bytes re-measured.
- `runtime/recovery_manager.py` — no pointer-derived or fallback
  authority; eligible committed deployment only; fresh grant bound to
  the operative revocation epoch; `UNAVAILABLE` when nothing eligible.
- `runtime/supervisor.py` — `authorize()` and `commit_activation()`
  re-check the operative epoch and explicit decision/key revocation at
  commit time (authorize→prepare→commit race closed). Newly revoked
  actives are route-withdrawn and quarantined; revoked predecessors are
  retired before fallback selection so a revoked model cannot be
  restored as fallback.
- `v161/trusted_launcher.py` — retains authority documents at first
  admission so restoration has real evidence to verify.

## SEC-403 — Release integrity (closed)

The v16.4.4 zip shipped a manifest generated before part of its source:
18 governed files hashed differently and 246 evidence files were
unmanifested (233 tracked JSONs, 13 archives).

Repair:

- Governed set = `git ls-files` ∩ disk. `colab-evidence/` is untracked
  and gitignored; campaign evidence ships separately as
  `mini-AGI-Campaign3A-Evidence.zip` with its own
  `EVIDENCE_MANIFEST.json`, signature, and `EVIDENCE_PROVENANCE.json`.
  The historical Campaign 3A `REFUSED` result is preserved unmodified.
- `scripts/validation/reissue_release.py` — refuses to mint a signing
  key when the pinned release key is absent (`--generate-key` is an
  explicit development-only escape).
- `scripts/release/verify_final_package.py` — deterministic zip build
  (sorted entries, epoch timestamps, fixed modes), fresh-extraction
  verification of both artifacts.
- `RELEASE_CHANGE_MANIFEST.json` records every added/modified governed
  file with old/new sha256 and its repair id; pre-existing v16.4.4
  drift not otherwise touched by this release is reconciled under
  SEC-403.

## SEC-404 — Multiprocessing test portability (closed)

`test_two_processes_exactly_one_wins` defined its spawn child in the
test module, unimportable under `--import-mode=importlib`. The child
now lives in `tests-python/helpers/grant_ledger_worker.py`; the harness
distinguishes import failure, hang, silent exit, worker error, and true
double-consumption, and runs under spawn plus supported native start
methods (fork excluded on macOS — sqlite children cannot run after
fork; documented, not masked).

## Validation

`909 passed, 1 skipped` (Linux-only RLIMIT_AS test on the macOS host).
Release verification is performed on fresh extractions of the built
zips — see `RELEASE_ATTESTATION.json` and `TEST_RESULTS_V1645.md`.
