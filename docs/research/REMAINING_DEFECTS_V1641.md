# Remaining-defects register — v16.4.3

Defects known at the v16.4.3 release point, with severity, status, and
the reproducible test case (or the reason none exists yet). All ten
v16.4.2 authority/concurrency/filesystem/recovery/journal/backend/
service/routing findings (SEC-201..207, OPS-001..003) are closed by
this release. Controller correctness (SEC-005/006) is the v16.4.4
gate; real-model qualification (SEC-007) is the v16.5.0 gate.

| ID | Severity | Defect | Status | Reproducing test |
|---|---|---|---|---|
| SEC-001 | critical | Runtime admission not mandatory in every model-loading path | **CLOSED v16.4.1** | `test_v1641_trusted_launcher.py`, `test_v1641_cli.py` |
| SEC-002 | critical | Artifact hashing overlooked symbolic links; unlisted files indistinguishable | **CLOSED v16.4.1** | `test_v1641_artifact_closure.py` |
| SEC-003 | high | Unsigned activation receipts passed the verification helper | **CLOSED v16.4.1** | `test_v1641_trusted_launcher.py::test_unsigned_and_wrong_role_receipts_are_problems` |
| SEC-004 | high | Schema/protocol requirements optional (plan, qualification, protocol link, revocation freshness) | **CLOSED v16.4.1** | `test_v1641_strict_schema.py` |
| ACT-001 | critical | `TrustedRuntimeLauncher` loaded the backend before activation evidence was durable; a post-load persistence failure had no guaranteed unload | **CLOSED v16.4.2** | `test_v1642_activation_closure.py` rows 10-13; `test_activation_supervisor.py` |
| AUTH-001 | critical | `RuntimeAdmissionController` defaulted missing `runtime_backends` coverage to the supported-backend list | **CLOSED v16.4.2** | `test_v1642_activation_closure.py::test_row3/4` |
| AUTH-002 | high | A staged `ApprovedSnapshot` conflated artifact measurement with deployment authorization | **CLOSED v16.4.2** | `test_admission_grants.py`, `test_activation_supervisor.py::test_arbitrary_object_with_path_refused` |
| REV-001 | high | Revocation evidence was an unsigned, non-monotonic digest list that could not reject future-dated or replayed snapshots | **CLOSED v16.4.2** | `test_signed_revocations.py`, `test_v1642_activation_closure.py` rows 7-9 |
| SEC-201 | critical | Grant consumption tracked in process-local `_consumed_grants` set — lost on restart | **CLOSED v16.4.3** | `security/test_grant_ledger.py` (replay + restart persistence) |
| SEC-207 | critical | Grant check/insert not atomic — concurrent requests could both authorize | **CLOSED v16.4.3** | `security/test_grant_ledger.py` thread and subprocess races — exactly one winner |
| SEC-203 | critical | One uid-allowlist authorized every privileged operation — research clients held operator authority | **CLOSED v16.4.3** | `runtime/test_supervised_service.py` per-operation matrix |
| SEC-204 | critical | Caller-controlled `campaign_id`/`seed` built activation paths; `receipt_path` accepted from clients | **CLOSED v16.4.3** | `security/test_path_containment.py`, opaque-id launcher rewrite |
| SEC-202 | critical | Recovery restored `_active_id` from a pointer with no loaded model behind it | **CLOSED v16.4.3** | `runtime/test_activation_supervisor.py` crash matrix; cold start is `UNAVAILABLE` until re-admitted + probed |
| SEC-205 | high | `DurableJournal.records()` trusted digests/signatures/grammar without verification | **CLOSED v16.4.3** | `runtime/test_journal_integrity.py` + `migrate_journal_v1_to_v2.py` |
| SEC-206 | high | `backend_binary_digest`/`policy_epoch` permissive-defaulted and unenforced — grants did not bind implementation identity | **CLOSED v16.4.3** | `security/test_backend_identity.py`, `runtime/backend_manifest.py` |
| OPS-001 | high | Unbounded `readline()` before size check; unbounded thread-per-connection; no deadlines | **CLOSED v16.4.3** | `runtime/test_supervised_service.py` (bounded read, worker pool, quotas, deadlines) |
| OPS-002 | medium | Socket `0600` contradicted documented cross-uid research access; no endpoint separation | **CLOSED v16.4.3** | `research.sock`/`operator.sock` + socket-dir ownership; `test_supervised_service.py` |
| OPS-003 | high | Activation control never reached inference — no router, no backend `infer`, traffic could bypass admission | **CLOSED v16.4.3** | `runtime/test_serving_router.py`, `integration/test_full_admission_chain.py` |
| SEC-005 | high | Mechanism controller accepts unsupported prerequisite evidence: `prior_attempts` are unverified digest-shaped strings; `VerifiedAttemptRecordV2` (typed, signed, resolvable) does not exist | **OPEN — v16.4.4** | `tests-python/v161/test_v166_mechanism_controller.py` documents the current acceptance; the defect is that a fabricated `sha256:…` string satisfies the ladder check |
| SEC-006 | high | Mechanism selector may choose the highest-ranked intervention even when its expected utility is negative; no `NO_CHANGE` action and no preregistered minimum-utility threshold τ | **OPEN — v16.4.4** | `test_v166_mechanism_controller.py` (utility arithmetic) — extend with a negative-utility fixture in v16.4.4 |
| SEC-007 | medium | Real-model learning and end-to-end serving qualification incomplete: Campaign 3A executing (GPU), 3B corpus unsealed, 3C not run; no qualified artifact has been served through the real PEFT backend end to end | **OPEN — v16.5.0** | Campaign 3 execution + `trusted_launch.py --backend hf-peft` on a qualified adapter; `test_v1641_cli.py` uses a test backend |
| OPS-004 | medium | Operational role separation is not enforced by code: all role keys live under one storage root in the development scaffold | **PARTIAL v16.4.3** — the service now derives roles from authenticated uids and holds only the keys it needs; the dev scaffold still co-locates keys under one root | `test_supervised_service.py`; `minagi/v161/authority.py` module docstring; `docs/research/THREAT_MODEL_V1643.md` |
| OPS-005 | low | Snapshot immutability is process-level (read-only modes + staged copy + pre-load re-verification); a same-identity adversary can chmod and rewrite, which `verify_snapshot` detects but does not prevent | **ACCEPTED RISK** — mitigated: the supervisor re-verifies the grant AND the staged bytes at stage() | `test_v1641_artifact_closure.py::test_tampering_with_staged_snapshot_is_detected` |
| OPS-006 | low | `peer_uid()` relies on `SO_PEERCRED`/`getpeereid`, which CPython does not expose on macOS — the service fails closed there until a platform transport lands | **OPEN — platform coverage** | `test_supervised_service.py::test_peer_uid_extractable_on_socketpair` |
| OPS-007 | low | Event-log tail truncation by a storage-write adversary is detectable via the signed-checkpoint anchor, but an anchor kept on the same host is a weaker guarantee than an offline anchor | **ACCEPTED RISK** — `anchor_checkpoint()` + `integrity_check` close silent truncation; an external anchor is future work | `runtime/test_journal_integrity.py::test_truncated_tail_detected` |
