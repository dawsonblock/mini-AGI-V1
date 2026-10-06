# Validation Summary — v15.8.0

- Python/governance/runtime: **304 passed, 0 failed** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).
- Focused v15.8 preregistered neural campaign: **7 passed, 0 failed**.
- Portable native C++: **28 passed, 0 failed**.
- Host QW3 target compiled successfully with CUDA and Metal GDN disabled.
- Real-weight Qwen + trained-adapter campaign: **not run / not claimed**.

The v15.8 tests prove experiment-plan preregistration, A0/A1 arm immutability, task-battery immutability, gate immutability, deterministic governed launch-spec generation, and evidence-closed result bundling. Deterministic model doubles are used for the campaign PASS fixture; they do not constitute neural-learning evidence.
