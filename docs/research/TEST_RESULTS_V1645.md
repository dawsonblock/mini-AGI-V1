# Test Results — v16.4.5

Environment: macOS (Apple Silicon), Python 3.12.0, CPU.

Full suite: **909 passed, 1 skipped, 0 failed** (~49s)
(`python3 -m pytest tests-python -x -q`)

Skipped: Linux-only `RLIMIT_AS` enforcement test on the macOS host —
a platform limitation, recorded, not masked.

## v16.4.5 additions

| Suite | Tests | Covers |
|---|---|---|
| `runtime/test_worker_protocol.py` | 27 | frame codec, schema/length/digest validation, oversized and truncated frames, EOF, unexpected types/ids, pending bounds |
| `runtime/test_worker_ipc_adversarial.py` | 9 | hostile worker bytes, spawned protocol attacks, worker death and supervisor usability |
| `runtime/test_worker_privilege_isolation.py` | 9 | credential filtering, private scratch permissions, limits config, env dump |
| `runtime/test_revocation_during_activation.py` | 5 | epoch advance authorize→commit, decision revoked during preparation, rollback under revocation |
| `security/test_restoration_authority.py` | 13 | verifier across the whole chain: signature, bindings, epoch, revoked decision/key, tampered artifacts, missing docs |
| `security/test_restoration_revocation.py` | 6 | e2e: valid restore, revoked promotion, revoked key, altered artifact, stale grant, all-revoked → UNAVAILABLE |
| `security/test_grant_ledger.py` | +1 param | spawn portability; hang/import/silent-exit failure modes |

Re-baselined for the new semantics (not weakened):
`test_activation_supervisor.py` (authorize-time epoch floor),
`test_v1644_runtime_closure.py` (restoration now requires the retained
authority chain), `test_v1644_real_model_e2e.py` (real signed chain +
trusted launcher so restore legitimately passes),
`test_worker_backend.py` (pickle-survival → descriptor transport).

## Release verification

`scripts/release/verify_final_package.py` builds both zips
deterministically and verifies fresh extractions:

- source: `scripts/verify_release.py --root <extract>` — manifest ==
  every on-disk governed file, signature under pinned fingerprint,
  attestation reconciled.
- evidence: manifest == every extracted file, signature under the same
  pinned key.

Final result is recorded in `RELEASE_ATTESTATION.json`
(`fresh_extraction_verification`).
