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
| SEC-707 | Inconsistent release metadata | PENDING | regenerate manifest/signature/SBOM/source inventory from final source after code freeze | release verification run (final step) |

## Exit-gate status

| Gate | Claim | Evidence |
|------|-------|----------|
| G1 NsJail restrictions | source complete, **unqualified** | `test_v1648_linux_nsjail.py` skips off-Linux |
| G2 authority isolation | source complete, **unqualified** | `denied_paths` + `_allowed` mount screening; Linux test pending |
| G3 network denial | source complete, **unqualified** | `clone_newnet`+`iface_no_lo` in generated config; probe `net:` pending |
| G4 kernel containment | source complete, **unqualified** | cgroup.populated verification in `broker.terminate`; Linux test pending |
| G5 job inputs cannot weaken isolation | verified by unit tests | frozen profile, `with_runtime` whitelist, digest binding |
| G6 production refuses missing/ineffective restrictions | verified by unit tests | `resolve_worker_executor` SystemExit paths |
| G7 lifecycle correctness | verified on dev executor | 364 runtime tests incl. IPC adversarial + spawn/unload |
| G8 regression tests pass | verified | `test_v1648_executor_isolation.py` 42 tests |
| G9 release inventory | pending | final release step |
| G10 real-model lifecycle | pending | needs Linux host + HF/PEFT env |
