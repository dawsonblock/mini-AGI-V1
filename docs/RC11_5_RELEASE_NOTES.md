# RC11.5 Integrated Runtime

RC11.5 is a convergence release built on RC11.4.

## Integrated changes
- Restores the RC10.9 native Gated-DeltaNet segment I/O, summary runtime, runtime bundle, CLI tools, Apple Metal bridge, non-Apple fail-closed stub, and qualification tests into the canonical native CMake graph.
- Preserves the RC10.9 Qwen executor capture implementation under `reference_integrations/rc10.9-native/` for deliberate porting without silently replacing RC11 authority-aware runtime code.
- Carries RC10.23 content-bound qualification/deployment implementation and tests as explicit provenance under `reference_integrations/rc10.23-qualification/`; non-conflicting deployment modules are also present in execution memory for convergence work.
- Carries RC10.11 CAS/snapshot/promotion durability modules under `reference_integrations/rc10.11-durability/` for regression auditing.
- Fixes the RC11.4 Mac installer collision where two distinct distributions named `kvcontinual` were editable-installed into one environment. RC11.5 installs isolated continual and execution-memory environments until the Python APIs are deliberately unified.
- Adds `scripts/rc11/validate_integrated.sh`, which runs all RC11 Python/reference regressions and compiles/tests the restored GDN native targets.

## Deliberate non-claim
RC11.5 does not claim real Apple Metal HYPIC acceleration is production-qualified merely because the host/reference targets pass. Production authorization still requires target Apple-Silicon qualification against the exact model, adapter, backend, runtime build, qualification suite, and signed RC11 authority state.
