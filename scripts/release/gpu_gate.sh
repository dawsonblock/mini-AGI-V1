#!/usr/bin/env bash
# Mini-AGI GPU release gate.
#
# Runs the real-model qualification on a CUDA host: actual PEFT weights
# through the governed serving path (admit -> load -> infer -> rollback
# -> cold restoration). An NVIDIA GPU must be visible and the test must
# RUN — a skip fails the gate, the same posture as release verification:
# a gate that quietly skips did not qualify anything.
#
# Required env on the host: working CUDA + drivers, torch with CUDA,
# transformers, peft, and model access (HF_HOME may be pre-populated or
# outbound fetch allowed).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo 'nvidia-smi not found — GPU gate requires an NVIDIA host' >&2
  exit 2; }
nvidia-smi -L | grep -q 'GPU ' || {
  echo 'no NVIDIA GPU visible' >&2
  exit 2; }

LOG="$(mktemp -t minagi-gpu-gate.XXXXXX.log)"
set +e
(cd "${ROOT}" && PYTHONPATH=src-python:tests-python \
  python3 -m pytest tests-python/runtime/test_v1644_real_model_e2e.py \
  -v 2>&1) | tee "${LOG}"
status=${PIPESTATUS[0]}
set -e
(( status == 0 )) || exit "${status}"

if grep -Eiq '(^|[^a-z])(SKIP|skipped|s skipped)([^a-z]|$)' "${LOG}"; then
  echo 'GPU release gate rejected an apparent skipped real-model test' >&2
  exit 3
fi
echo "gpu release gate: PASS"
