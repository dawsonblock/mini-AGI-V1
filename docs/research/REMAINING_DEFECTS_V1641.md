# Remaining-defects register — v16.4.2

Defects known at the v16.4.2 release point, with severity, status, and
the reproducible test case (or the reason none exists yet). The
runtime-security defects from the v16.4.0 audit are closed, and the
three activation-authority weaknesses confirmed in the v17 upgrade
spec (UPGRADE_PLAN_V17 §1/§3) are closed by this release. The
controller-correctness and scientific-qualification items below are
deliberately NOT addressed here — v16.4.3 and v16.5.0 have independent
gates.

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
| SEC-005 | high | Mechanism controller accepts unsupported prerequisite evidence: `prior_attempts` are unverified digest-shaped strings; `VerifiedAttemptRecordV2` (typed, signed, resolvable) does not exist | **OPEN — v16.4.3** | `tests-python/v161/test_v166_mechanism_controller.py` documents the current acceptance; the defect is that a fabricated `sha256:…` string satisfies the ladder check |
| SEC-006 | high | Mechanism selector may choose the highest-ranked intervention even when its expected utility is negative; no `NO_CHANGE` action and no preregistered minimum-utility threshold τ | **OPEN — v16.4.3** | `test_v166_mechanism_controller.py` (utility arithmetic) — extend with a negative-utility fixture in v16.4.3 |
| SEC-007 | medium | Real-model learning and end-to-end serving qualification incomplete: Campaign 3A executing (GPU), 3B corpus unsealed, 3C not run; no qualified artifact has been served through the real PEFT backend end to end | **OPEN — v16.5.0** | Campaign 3 execution + `trusted_launch.py --backend hf-peft` on a qualified adapter; `test_v1641_cli.py` uses a test backend |
| OPS-001 | medium | Operational role separation is not enforced by code: all role keys live under one storage root in the development scaffold | **PARTIAL v16.4.2** — `minagi.runtime.service` now runs the supervisor under a dedicated OS identity with peer-uid authentication; the dev scaffold still co-locates keys | `test_supervised_service.py`; `minagi/v161/authority.py` module docstring |
| OPS-002 | low | Snapshot immutability is process-level (read-only modes + staged copy + pre-load re-verification); a same-identity adversary can chmod and rewrite, which `verify_snapshot` detects but does not prevent | **ACCEPTED RISK** — mitigated: the supervisor re-verifies the grant AND the staged bytes at stage() | `test_v1641_artifact_closure.py::test_tampering_with_staged_snapshot_is_detected` |
| OPS-003 | low | `peer_uid()` relies on `SO_PEERCRED`/`getpeereid`, which CPython does not expose on macOS — the service fails closed there until a platform transport lands | **OPEN — platform coverage** | `test_supervised_service.py::test_peer_uid_extractable_on_socketpair` |
