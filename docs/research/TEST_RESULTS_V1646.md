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

The last three qualify on the Linux root profile (Colab) — recorded
in the Colab run section below once executed.

## Deterministic build + fresh-extraction verify

| Artifact | sha256 | Result |
|---|---|---|
| `mini-AGI-V1-v16.4.6-Worker-Isolation-and-Process-Closure.zip` | `f431b727ce6f152c0b9ee4172cd862e13397af7ec29036889c6148c013e153a3` | PASS — 1523 files, pinned key `0f9d7358` |
| `mini-AGI-Campaign3A-Evidence.zip` | `e458b6d666c65f6c2b787cc8a1f6ce8cadd3e7d96214322880f25e31631bb1db` | PASS — 246 files |

Manifest `e325f0f4…` signed under the pinned release key.
Report: `dist-v1646/VERIFICATION_REPORT.json`.
