# Real Model Lifecycle (v16.4.9) — G10

## Status: PASSED — qualified in the privileged Linux container

Real HF/PEFT lifecycle executed INSIDE the systemd+cgroup-v2+NsJail
boundary, worker identity `nobody` (uid/gid 65534), supervisor root.

Environment: Debian bookworm, systemd 252, unified cgroup v2,
kernel linuxkit 7.0.14 (aarch64), NsJail pinned `4ff54a6`
(binary sha256 `a9408ab2…`), torch 2.14.1+cu130, transformers
5.19.0, peft 0.21.2.

Evidence: `tests-python/runtime/test_v1649_real_model_nsjail.py`

- `test_g10_real_model_load_infer_terminate_inside_jail` — real
  GPT2 + LoRA adapter staged as a hash-verified snapshot, loaded by
  the jailed `PeftServingBackend`, real inference (`completion`
  string + `prompt_tokens` metrics), `containment_status` reports
  `cgroup_populated is True`, then `terminate()` verifies the unit
  cgroup EMPTY. **PASSED**
- `test_g10_real_model_cancel_and_switch_inside_jail` — model A
  serving, `cancel()` then inference reports
  `metrics.cancelled is True`; model B loads in a **fresh** jailed
  worker (one proxy supervises one worker — the production
  activation shape) and serves. **PASSED**

## Latent defect found and fixed by this qualification

The G10 run exposed a real resource-boundary bug that the earlier
adversarial suite could not see:

- NsJail's `rlimit_as` default (4096 MiB *virtual* address space)
  capped the worker at ~61 pthreads — virtual accounting counts
  mmap'd regions (weights, glibc arenas ~64 MiB/thread), so torch /
  tokenizers' thread pools died with `EAGAIN`.
- Fix: generated configs now emit `rlimit_as_type: INF` and the
  profile carries `memory_mb: 8192` → the **cgroup `MemoryMax`**
  becomes the authoritative memory bound (virtual-AS rlimits are the
  wrong mechanism for mmap-based ML runtimes).
- `max_processes`/`TasksMax` remain 512 — verified 400/400 pthreads
  spawn inside the jail.

Also repaired during this qualification:

- `TOKENIZERS_PARALLELISM` (and other benign perf/verbosity knobs)
  were rejected by the credential-shaped environment screen — added
  to the explicit allowlist.
- Pytest `tmp_path` chains are `0700` — test helpers now chmod the
  ancestor chain `o+x` (traverse-only) so the 65534 worker can reach
  its scratch bind source.

## What remains unqualified

- GPU (CUDA) pass — the runs above are CPU. A device qualification
  requires an explicit, enumerated device/driver/library bind list,
  never a broad `/dev` expose.
- Large production-scale models (the fixture is deliberately tiny to
  exercise the real code paths without a download).
- Rootless `systemd-run --user` deployment mode.
