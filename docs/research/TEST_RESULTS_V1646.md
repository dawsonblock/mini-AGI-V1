# v16.4.6 Test Results

## Local (macOS dev host, Python 3.12)

- `pytest tests-python`: **946 passed, 4 skipped, 0 failed** (~51s)
- New adversarial suite `test_v1646_worker_isolation.py`: 36 tests —
  all runnable paths pass on this host.

## Skip classification (honest)

| Test | Reason |
|---|---|
| `test_rlimit_as_*` (v16.4.5 carry) | RLIMIT_AS absent on macOS — Linux-only |
| `test_unit_marker_survives_group_detach` | requires `/proc` (Linux) |
| `test_worker_scratch_worker_owned` | uid demotion requires privilege |
| `test_demoted_worker_cannot_read_supervisor_files` | uid demotion requires privilege |

The last three qualify on the Linux root profile (Colab) — PASSED there
(see below).

## Deterministic build + fresh-extraction verify

| Artifact | sha256 | Result |
|---|---|---|
| `mini-AGI-V1-v16.4.6-Worker-Isolation-and-Process-Closure.zip` | `f431b727ce6f152c0b9ee4172cd862e13397af7ec29036889c6148c013e153a3` | PASS — 1523 files, pinned key `0f9d7358` |
| `mini-AGI-Campaign3A-Evidence.zip` | `e458b6d666c65f6c2b787cc8a1f6ce8cadd3e7d96214322880f25e31631bb1db` | PASS — 246 files |

Manifest `e325f0f4…` signed under the pinned release key.
Report: `dist-v1646/VERIFICATION_REPORT.json`.


## Google Colab (T4 session `v1646-qual`, Ubuntu root, Python 3.13)

- Release verification on fresh extraction: **PASS** — 1522 files,
  manifest `66c53a77…`, pinned key `0f9d7358`.
- `test_v1646_worker_isolation.py`: **39/39 PASS** — including the
  uid-65534 demotion tests (worker cannot read the supervisor's
  0600 key/DB or write protected paths; worker-owned scratch
  writable) and the `/proc` env-marker detach test.
- Full suite: **931 passed, 20 skipped, 0 failed** (~65s).
- Colab skips: documented platform unavailability only.
- Note: `torchao 0.10.0` (preinstalled) conflicts with peft's >=0.16
  check — uninstalled in-session; not a code defect.

## Native CPU (macOS, QW3_ENABLE_CUDA=OFF)

- `ctest` over the test targets: **28/28 PASS**.
