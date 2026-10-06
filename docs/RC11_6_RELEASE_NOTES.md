# RC11.6 — Converged runtime

RC11.6 turns RC11.5 from an integrated collection into one installable Python runtime and restores the newer native Qwen GDN executor path.

## Active changes

- Unified `kvcontinual-runtime` distribution with `kvcontinual.continual` and `kvcontinual.execution` domains.
- Single macOS `.venv`; removed dual-package editable-install workaround.
- RC10.9 `qwen_executor.cpp/.hpp` promoted into the root native runtime, including `QW3_GDN_CAPTURE_DIR` capture and qualification-stage recurrent state support.
- RC10.11 durability primitives promoted into `kvcontinual.execution.durability`.
- RC10.23 qualification/deployment implementation remains active under the execution domain; passive reference copies are removed.
- Added one-interpreter cross-plane smoke validation.
- Release/document version identity normalized to 11.6.0.

## Validation performed in the build environment

- Unified Python runtime: 106 tests passed.
- Authority/control reference suite: passed.
- Oracle reference suite: 25 tests passed.
- Native portable `qw3_core` compiled with the RC10.9 Qwen executor transplant.
- GDN reference, segment I/O, summary runtime and runtime bundle tests passed.
- Metal interface failed closed as expected on non-Apple host.

## Remaining qualification requirement

Real Apple Silicon Metal execution with real Qwen hybrid weights has not been certified in this environment. RC11.6 must remain fail-closed until target-hardware differential qualification succeeds and a signed deployment binding authorizes that exact execution identity.
