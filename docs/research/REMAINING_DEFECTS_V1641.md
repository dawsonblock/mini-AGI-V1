# Remaining-defects register — v16.4.1

Defects known at the v16.4.1 release point, with severity, status, and
the reproducible test case (or the reason none exists yet). The
runtime-security defects from the v16.4.0 audit are closed; the
correctness and scientific-qualification items below are deliberately
NOT addressed by this release (see the roadmap in the release plan:
v16.4.2 and v16.5.0 have independent gates).

| ID | Severity | Defect | Status | Reproducing test |
|---|---|---|---|---|
| SEC-001 | critical | Runtime admission not mandatory in every model-loading path | **CLOSED v16.4.1** | `test_v1641_trusted_launcher.py`, `test_v1641_cli.py` |
| SEC-002 | critical | Artifact hashing overlooked symbolic links; unlisted files indistinguishable | **CLOSED v16.4.1** | `test_v1641_artifact_closure.py` |
| SEC-003 | high | Unsigned activation receipts passed the verification helper | **CLOSED v16.4.1** | `test_v1641_trusted_launcher.py::test_unsigned_and_wrong_role_receipts_are_problems` |
| SEC-004 | high | Schema/protocol requirements optional (plan, qualification, protocol link, revocation freshness) | **CLOSED v16.4.1** | `test_v1641_strict_schema.py` |
| SEC-005 | high | Mechanism controller accepts unsupported prerequisite evidence: `prior_attempts` are unverified digest-shaped strings; `MechanismAttemptRecord` (typed, signed, resolvable) does not exist | **OPEN — v16.4.2** | `tests-python/v161/test_v166_mechanism_controller.py` documents the current acceptance; the defect is that a fabricated `sha256:…` string satisfies the ladder check |
| SEC-006 | high | Mechanism selector may choose the highest-ranked intervention even when its expected utility is negative; no `NO_CHANGE` action and no preregistered minimum-utility threshold τ | **OPEN — v16.4.2** | `test_v166_mechanism_controller.py` (utility arithmetic) — extend with a negative-utility fixture in v16.4.2 |
| SEC-007 | medium | Real-model learning and end-to-end serving qualification incomplete: Campaign 3A executing (GPU), 3B corpus unsealed, 3C not run; no qualified artifact has been served through the real PEFT backend end to end | **OPEN — v16.5.0** | Campaign 3 execution + `trusted_launch.py --backend hf-peft` on a qualified adapter; `test_v1641_cli.py` uses a test backend |
| OPS-001 | medium | Operational role separation is not enforced by code: all six role keys live under one storage root in the development scaffold | **OPEN — deployment requirement** | `minagi/v161/authority.py` module docstring; `scripts/authority_bootstrap.py` |
| OPS-002 | low | Snapshot immutability is process-level (read-only modes + staged copy + pre-load re-verification); a same-identity adversary can chmod and rewrite, which `verify_snapshot` detects but does not prevent | **ACCEPTED RISK** | `test_v1641_artifact_closure.py::test_tampering_with_staged_snapshot_is_detected` |
| TEST-001 | medium | 4 sandbox-backend-dependent tests fail/skip where no OS sandbox backend is available; test portability classification (supported / intentionally unavailable / misconfigured / missing ML dependency) is v16.4.2 work | **OPEN — v16.4.2** | `tests-python/v161/test_v1622_evaluator_sandbox.py` |
| REL-001 | low | This release's Python matrix ran on macOS 3.12 only. Linux CPU/sandbox, Windows (where supported), and the GPU/PEFT campaign are not re-run here; passing Linux CPU tests must not be represented as CUDA/Metal qualification | **OPEN — CI matrix** | `docs/research/RELEASE_RC8_1.md`, `.github/` workflows |
| SCI-001 | medium | Campaign 3A execution is incomplete (GPU-blocked, VM reclamation recovered); 3B/3C unsealed. No scientific claim is made or changed by v16.4.1 | **OPEN — v16.5.0** | `scripts/campaign3a_finalize.py`, `docs/research/CAMPAIGN3_DESIGN.md` |

Severity: critical = unqualified artifact can be served or
authorization bypassed; high = authorization/evidence integrity
weakness; medium = correctness or qualification gap; low = hardening
or portability gap.
