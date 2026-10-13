# Mini-AGI v16.4.8 — Engineering Issue Register

Branch `hardening/v16.4.8` · baseline v16.4.7 Runtime Isolation
Enforcement · target v16.4.8 Linux Isolation and Security Closure.

| ID | Issue | Status | Fix landed | Regression evidence |
|----|-------|--------|------------|---------------------|
| SEC-701 | Incorrect capability detection — `os.geteuid()==0` short-circuited `_cap_eff`, so a root-in-container supervisor claimed CAP_SYS_ADMIN it did not hold | FIXED (source) | `worker_isolation._cap_eff` reads `/proc/self/status` CapEff only; `_has_demote_privilege` requires CAP_SETUID on Linux; preflight launches a real probe | `test_cap_eff_reads_status_not_root`, `test_demote_privilege_needs_cap_setuid_on_linux`, `test_validate_refuses_fake_linux_host` |
| SEC-702 | Invalid custom `pivot_root` setup — hand-rolled unshare/pivot/mount launcher on the production path | FIXED (source) | `worker_launcher` demoted to development executor only; production boundary is `nsjail_executor` + `worker_launch_broker` (systemd transient unit + NsJail) | `test_service_production_rejects_development_executor`, broker unit-argv tests |
| SEC-703 | Policy loses mount settings — env override reconstructed the policy field-by-field and dropped `mount_ro/mount_rw/no_new_privs` | FIXED | `dataclasses.replace` on the frozen policy; `policy_digest()` bound to every launch; frozen `IsolationProfile` + `with_runtime` whitelist | `test_env_overrides_cannot_drop_mount_policy`, `test_isolation_policy_frozen`, `test_policy_digest_binds_mounts`, `test_with_runtime_only_permitted_fields` |
| SEC-704 | Detached processes escape discovery — `/proc` ancestry + env markers lose reparented/setsid children | FIXED (source; needs Linux run) | kernel-owned cgroup: `KillMode=control-group` systemd unit per worker; teardown verifies `cgroup.populated==0`, never `/proc` scanning as the claim | `test_g4_detached_descendant_stays_owned`, `test_g4_spawn_backend_children_all_terminated` (Linux-gated) |
| SEC-705 | macOS Seatbelt invoked on Linux — `sandbox_exec` credited wherever a same-named binary existed on PATH | FIXED | `supported()["sandbox_exec"]` gated on Darwin; `production_policy` attaches the bundled profile only on Darwin; `wrap_argv` refuses a profile off-Darwin | `test_seatbelt_wrap_refused_off_darwin`, `test_seatbelt_not_credited_on_linux`, `test_production_policy_no_sandbox_profile_on_linux` |
| SEC-706 | Incorrect bind-mount targets — custom launcher mounted at wrong dst | FIXED (source; needs Linux run) | `resolve_mounts` computes broker-owned binds (runtime closure ro, artifacts ro at identical paths, scratch rw at `/work`, `/dev/{null,zero,urandom}`); `denied_paths` refuse authority mounts | `test_resolve_mounts_denies_authority`, `test_render_config_required_fields`, `test_g1_authority_dir_never_mounted` (Linux) |
| SEC-707 | Inconsistent release metadata | FIXED | `SOURCE_MANIFEST.json` regenerated over final tracked tree (1544 files) and re-signed with the pinned key; attestation/validation/change-manifest/SBOM/version reconciled to 16.4.8 | `verify_release.py --root` PASS on `git archive` extraction AND on the built ZIP (`836a65c4…`, manifest `7e46786a…`) — G9 |

## v16.4.9 — independent-audit repairs (SEC-801..805)

External audit of the shipped v16.4.8 ZIP reproduced the following;
all are fixed on this branch and re-qualified.

| ID | Defect (audit-reproduced) | Status | Fix | Tests |
|----|---------------------------|--------|-----|-------|
| SEC-801 | `terminate()` treated *unreadable* occupancy (`cgroup_populated` → `None`) as verified-empty; `cgroup.populated` never existed on cgroup v2 | FIXED | `cgroup_populated` parses `cgroup.events` populated field (`cgroup.procs` fallback); verification accepts ONLY explicit `populated 0`, or cgroup dir gone + unit inactive; `None` spins to deadline → unverified | `test_sec801_unreadable_occupancy_never_certifies_empty`, `test_sec801_populated_one_never_certifies_empty` |
| SEC-802 | `resolve_mounts` screened src-under-denied only — an ANCESTOR mount of a denied dir exposed it | FIXED | `_allowed` also refuses src that *contains* any denied path; broker `extra_ro` screening is bidirectional | `test_sec802_ancestor_mount_over_denied_path_refused`, `test_sec802_extra_ro_ancestor_refused` |
| SEC-803 | `launch()` refused a missing cgroup only when the client had exited — an alive-but-uncontained unit returned a valid record | FIXED | `record.cgroup_path is None` at the deadline ALWAYS refuses: unit stopped, client reaped, handles closed, workspace removed, record dropped (`_abort_launch`) | `test_sec803_launch_refuses_when_cgroup_never_appears` |
| SEC-804 | Preflight grader skipped absent evidence — a synthetic probe with readable `/etc/shadow` and host-identical namespaces could qualify | FIXED | `_grade` requires measured evidence for every mandatory property: capeff==0 (missing=fail), worker euid/egid match expected, identity distinct from supervisor, mnt/net/pid/user namespace ids differ from host, `deny:` paths unreachable, `read:` positive controls reachable | `test_sec804_grader_requires_evidence` |
| SEC-805 | `_validate_request` accepted `NOT-SIGNED-OR-DIGEST`; budget digest unchecked; nonce was check-then-write | FIXED | worker role REQUIRES `sha256:` digests for signed backend manifest + execution budget (probe role: format-checked when present); executor binds a real budget digest derived from the frozen profile and refuses worker launches with no manifest digest; nonce reservation is one atomic `O_CREAT\|O_EXCL` | `test_sec805_worker_requires_authority_digests`, `test_sec805_nonce_replay_refused_second_time` |

Additional audit items fixed: unit stderr now lands in a bounded
workspace log (`unit-stderr.log`) instead of an undrained pipe that
could stall the client lifecycle; `systemd stop`/`kill` return codes
are captured in the terminate report; `status()` re-resolves a
collected cgroup dir; the qualification script requires a worker uid
distinct from the supervisor (root → 65534 default, non-root →
explicit `MINAGI_WORKER_UID`).

## Exit-gate status

| Gate | Claim | Evidence |
|------|-------|----------|
| G1 NsJail restrictions | **qualified** (privileged Linux container) | strict preflight + 7/7 adversarial in the same run as below |
| G2 authority isolation | **qualified** | `deny:/etc/shadow`,`deny:/root`→missing; `read:` positive control read-ok |
| G3 network denial | **qualified** | live host listener `net:LISTENER`→`connect-denied`; `127.0.0.1:9` denied |
| G4 kernel containment | **qualified** | `cgroup.events populated` verified per-unit; terminate requires explicit `0` or gone+inactive |
| G5 job inputs cannot weaken isolation | verified by unit tests | frozen profile, `with_runtime` whitelist, digest binding |
| G6 production refuses missing/ineffective restrictions | verified by unit tests | `resolve_worker_executor` SystemExit paths |
| G7 lifecycle correctness | verified on dev executor | runtime tests incl. IPC adversarial + spawn/unload |
| G8 regression tests pass | verified | executor + audit-repair suites green |
| G9 release inventory | verified | clean-extraction `verify_release.py` PASS (1544 files, pinned key fp `0f9d7358`) |
| G10 real-model lifecycle | **qualified** (CPU) | real GPT2+LoRA load/infer/cancel/switch inside the jail, uid 65534; GPU + production scale on target HW remains |


## v16.4.10 independent-audit register (hardening/v16.4.10)

| ID | Defect (audit-reproduced) | Status | Fix | Tests |
|----|---------------------------|--------|-----|-------|
| SEC-901 | `cgroup_populated` used `partition(" ")` (malformed lines accepted), treated any non-"1" as empty, and fell back to `cgroup.procs` | FIXED | strict two-field parser; only explicit `populated 0` is EMPTY; missing/duplicate/malformed/unexpected → UNKNOWN; procs fallback removed | `test_v16410_cgroup_evidence.py` |
| SEC-902 | `containment_status()` ORed `not active and not client_alive` onto `verified_empty` — an occupied cgroup could be reported empty | FIXED | shared `ContainmentState` evaluator (RUNNING/EMPTY/REMOVED_VERIFIED/UNKNOWN) for broker status, executor status and teardown; populated 1 unoverridable; removed-verified requires InvocationID-bound teardown evidence | `test_v16410_executor_status.py` |
| SEC-903 | `_ID_RE` permitted `.`/`..`; `prepare()` accepted caller-supplied worker ids — workspace paths and unit names were caller-influenced | FIXED | ids are `w-<hex32>` generated via `secrets.token_hex(16)`; prepare/launch refuse caller ids; workspaces created descriptor-relative under the pinned root, stale/pre-existing dirs refused | `test_v16410_worker_identity.py` |
| SEC-904 | `validate_host()` had `or os.geteuid() == 0` — root inside a cap-restricted container could claim eligibility | FIXED | euid shortcut removed; `capability_preflight` (CapEff bit 21 or enabled unprivileged userns) vs `enforcement_qualification` (real probe only) separated | `test_v16410_host_capabilities.py` |
| SEC-905 | launch carried digest-shaped strings the broker never resolved — invented/substituted authority could pass | FIXED | `AuthorizedLaunchRecord` (runtime-role Ed25519 envelope) binds activation/backend/runtime-manifest/artifact-root/profile/budget/expiry/epochs/nonce; broker verifies signer authorization, signature, expiry, revocation floor, exact request binding and resolver-backed approval; signed nonce is the atomic reservation | `test_v16410_broker_authority.py` |
| SEC-906 | production startup called `validate_host()` (capability preflight) but never ran the enforcement qualification | FIXED | startup runs the real `run_preflight` probe through the executor and persists a host-bound qualification record (boot/kernel/nsjail/profile/worker-uid/verdict); service import bug (`DEFAULT_ISOLATION` from the wrong module) fixed | `test_v16410_production_preflight.py` |
| VAL-901 | prior cancel test armed the flag before generation — mid-decode interruption unproven | FIXED | real weighted GPT-2 decode inside the jail is cancelled in flight via the injected event; `metrics.cancelled` true and termination verified | `test_v16410_live_inference_cancel.py` |
| VAL-902 | no end-to-end signed activation/rollback/recovery through the real path | FIXED | full signed chain (decision/qualification/plan/manifest/grant) → supervisor prepare/health/commit on a jailed WorkerBackend → router inference → rollback to A → cold restart + RecoveryManager with B's promotion revoked | `test_v16410_activation_lifecycle.py` |
| REL-901 | release/scientific metadata carried stale counts and pending gates | FIXED | version 16.4.10; change manifest, validation, changelog, issue register, remaining-defects and qualification docs reconciled to measured results |
