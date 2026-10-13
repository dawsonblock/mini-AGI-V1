# Test results — Campaign 3B

Every count below is from an executed run on this host
(macOS, Python 3.12.0, torch 2.10.0). No result is fabricated, no
assertion was silently downgraded, and no skipped test is counted as
passing.

## Python suite

Command: `python3 -m pytest tests-python -q`

| group | passed | failed | skipped | collected |
|---|---|---|---|---|
| `tests-python/v161` | 481 | 0 | 1 | 482 |
| `tests-python/runtime` | 432 | 0 | 17 | 449 |
| `tests-python/security` | 77 | 0 | 0 | 77 |
| `tests-python/research` | 41 | 0 | 0 | 41 |
| `tests-python/v14` | 50 | 0 | 0 | 50 |
| `tests-python/governance` | 25 | 0 | 0 | 25 |
| `tests-python/apple` | 5 | 0 | 0 | 5 |
| `tests-python/integration` | 3 | 0 | 0 | 3 |
| `tests-python` (root modules) | 63 | 0 | 0 | 63 |
| **total** | **1183** | **0** | **18** | **1201** |

New tests added by Campaign 3B: **30** (see below).

## Native suite

Command: `cmake -S . -B build-c3b -G Ninja -DCMAKE_BUILD_TYPE=Release
-DQW3_ENABLE_CUDA=OFF -DQW3_BUILD_TESTS=ON && cmake --build build-c3b
&& ctest --test-dir build-c3b --output-on-failure`

```
100% tests passed, 0 tests failed out of 28
Total Test time (real) = 0.87 sec
```

**28/28 passed, 0 failed, 0 skipped** — macOS CPU profile
(`QW3_ENABLE_CUDA=OFF`), matching the v16.4.10 record.

## Campaign 3B test areas

Mandatory areas from the master specification, each with the test that
covers it:

| required area | test | result |
|---|---|---|
| exact prompt-token parity | `test_campaign_prompt_parity.py` | PASS |
| correct supervised-token masking | `test_campaign_prompt_parity.py`, `test_campaign_answer_boundaries.py` | PASS |
| EOS and generation limits | `test_campaign_eos_behavior.py` | PASS |
| strict vs diagnostic scoring separation | `test_campaign_output_scoring.py` | PASS |
| replay provenance, retrieval, leakage prevention | `test_replay_provenance.py`, `test_replay_nonempty.py` | PASS |
| positive-control training | `test_positive_control_learning.py` | PASS |
| negative-control training | `test_positive_control_learning.py`, `test_v166_scientific_gates.py` | PASS |
| undefined-denominator statistics | `test_false_activation_undefined.py` | PASS |
| family-aware uncertainty | `test_family_cluster_statistics.py` | PASS |
| off-target regressions | `test_v165_qualifier_chain.py`, `test_false_activation_undefined.py` | PASS |
| sealed evaluator isolation | `test_confirmation_access_denial.py` | PASS |
| confirmation execution on synthetic fixtures | `test_sealed_confirmation_execution.py` | PASS |
| adapter and evidence integrity | `test_v1641_artifact_closure.py`, `test_evidence_receipt_v3.py` | PASS |
| signed qualification | `test_v165_qualifier_chain.py` | PASS |
| governed runtime integration | `tests-python/runtime/*` | PASS |
| governed safety & retention suite | `test_safety_regression_suite.py` | PASS |
| end-to-end governed safety enforcement | `test_governed_safety_enforcement.py` | PASS |

## New tests in this campaign (30)

| file | tests | what they pin |
|---|---|---|
| `test_campaign_prompt_parity.py` | 9 | token-level parity; the historical mismatch is real; boundary merges refused |
| `test_campaign_answer_boundaries.py` | 12 | Unicode, empty output, punctuation, truncation, multi-turn, special tokens |
| `test_campaign_eos_behavior.py` | 7 | EOS stop, bounded generation, finish reasons |
| `test_campaign_output_scoring.py` | 6 | strict EM primary; diagnostics never substitute |
| `test_replay_nonempty.py` | 7 | verified replay produces a non-empty, auditable buffer |
| `test_replay_provenance.py` | 6 | provenance fields; out-of-split rows refused and never persisted |
| `test_false_activation_undefined.py` | 3 | empty denominator → NOT_ESTIMABLE, never 0.0 |
| `test_family_cluster_statistics.py` | 4 | family-aware CIs, leave-one-family-out, worst family |
| `test_positive_control_learning.py` | 13 | positive control learns; receipt readiness measurements |
| `test_sealed_confirmation_execution.py` | 13 | freeze, artifact verification, execution, scoring, signed refusal |
| `test_confirmation_access_denial.py` | 11 | no sealed leakage; research-worker denial; sealed-lane governed parity |
| `test_safety_regression_suite.py` | 28 | suite content, polarity, rollup, UNDEFINED handling, gate, inheritance |
| `test_governed_safety_enforcement.py` | 4 | real qualifier: compliant qualifies, violation refuses, legacy NOT_APPLICABLE |

## Skipped tests (18) — all environment-gated, none counted as passing

| reason | count |
|---|---|
| NsJail / systemd boundary exists only on Linux | 10 |
| uid demotion requires privilege (Linux/Colab root) | 2 |
| cgroup v2 unavailable on this host | 1 |
| netns + demotion need `CAP_SYS_ADMIN`/root | 1 |
| marker scan needs `/proc` (Linux) | 1 |
| kernel refuses `NOTE_TRACK` on this host | 1 |
| `RLIMIT_AS` is not enforceable on macOS | 1 |
| NsJail/systemd boundary (live-inference cancel) | 1 |

These are the runtime's own Linux-boundary qualifications. They are
**NOT RUN** here, not passed. The v16.4.10 record carries a privileged
container qualification (aarch64 Debian, systemd 252, cgroup v2) that
this work did not re-run.

## Not run

| suite | status |
|---|---|
| GPU / CUDA tests | **NOT RUN** — no CUDA device on this host |
| Real NsJail isolation qualification | **NOT RUN** — Linux only |
| Real sealed confirmation | **NOT RUN** — no independent evaluator custody |
| G10 real-model qualification inside NsJail | **NOT RUN** — Linux only |

## Verification of the changes themselves

| check | result |
|---|---|
| `ruff check scripts/ src-python/minagi/v161/ tests-python/v161/` | **All checks passed** |
| changed files flagged by ruff over the full tree | **none** |
| runtime security modules in the diff | **none** |
| qualifier gate reasons removed | **none** (17 → 19) |
| `scripts/release/source_audit.py` | **PASS** |

## Verification of the shipped archive

Run against a **fresh extraction** of
`mini-AGI-V1-Campaign3B-Full-Scientific-Hardening.zip`, never the
working tree:

| check | result |
|---|---|
| `scripts/verify_release.py` on the extracted tree | **PASS** — 1590 files verified, signature valid against the bundled key |
| archive entries | 1594 (1590 governed + 4 envelope files) |
| duplicate entries | 0 |
| absolute or path-traversal entries | 0 |
| symlink entries | 0 |
| `python3 -m pytest tests-python -q` on the extracted tree | **1183 passed, 18 skipped, 0 failed** |
| all 13 required deliverables present | **yes** |

## Acceptance gates

| gate | status | evidence |
|---|---|---|
| **G1** prompt correctness | **PASS** | `test_campaign_prompt_parity.py`; 30/30 train rows prove `training_input_ids[:prompt_length] == inference_prompt_input_ids` on the pinned tokenizer; boundary merges refused |
| **G2** output correctness | **PASS** | answer-only contract frozen and digest-bound; strict exact match primary; format/EOS/truncation deterministic and tested |
| **G3** baseline validity | **PASS** | L5b admitted 30/30 train records, 20/20 retrieval hits, 760 injected tokens; provenance tests prove hidden/sealed labels are refused and never persisted |
| **G4** learning readiness | **PASS (with a reported failure)** | the fixed-rank LoRA positive control *runs and learns within-family* (0.222 → 0.444 vs frozen 0.222; NC collapses to 0.067); the 95% fit gate itself **FAILS** (0.767) and is reported as such |
| **G5** negative-control validity | **PASS** | shuffled-label LoRA train accuracy 0.067 vs 0.767 for correct labels; NC forward transfer −0.167, within the preregistered bound |
| **G6** statistical integrity | **PASS** | undefined denominators are `NOT_ESTIMABLE`/`null` and block; family-clustered CIs, leave-one-family-out, worst-family and positive-family counts implemented and tested |
| **G7** safety and retention | **PASS** | governed suite enforced on the executed run: 0 undefined mandatory categories, 8 categories measured, violations block; the frozen model's two violations are reported as a baseline property |
| **G8** confirmation integrity | **PASS for executability; real confirmation NOT RUN** | the executor generates and scores real predictions under frozen rules and signs QUALIFIED/REFUSED (19 tests); no independent evaluator custody was available, so no candidate is reported as confirmed |
| **G9** runtime preservation | **PASS** | `git diff` over the runtime security modules is empty; qualifier gate reasons only added |
| **G10** regression | **PASS (host scope)** | Python 1183/0/18; native CTest 28/28. GPU and Linux-isolation suites **NOT RUN** here |
| **G11** release integrity | **PASS** | clean-extraction verification above; manifest + signature verify; separate evidence package verifies by SHA-256 and signature |
| **G12** scientific honesty | **PASS** | every claim traces to a recorded run; the executed result is REFUSED and reported as REFUSED; the sealed lane is NOT RUN and reported as NOT RUN |

Two gates deserve their qualifier spelled out rather than a bare PASS:

- **G4** passes on *readiness to learn* and fails on the fit threshold.
  The distinction matters: the pipeline can train, and the training did
  not reach the bar the diagnostic set.
- **G8** passes on the machinery and is **NOT RUN** on the real thing.
  Treating an unexecuted confirmation as a pass is exactly the error
  Campaign 3A made.
