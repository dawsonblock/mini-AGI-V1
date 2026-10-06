#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${QW3_MODEL:?set QW3_MODEL to a compatible local qwen35 GGUF/HF model}"
: "${QW3_CUDA_ARCH:?set QW3_CUDA_ARCH for the target GPU, e.g. 80, 86, 89, 90, or 120a-real}"
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi not found' >&2; exit 2; }
nvidia-smi -L | grep -q 'GPU ' || { echo 'no NVIDIA GPU visible' >&2; exit 2; }
[[ -e "${QW3_MODEL}" ]] || { echo "model not found: ${QW3_MODEL}" >&2; exit 2; }

BUILD="${QW3_CUDA_BUILD_DIR:-${ROOT}/build-release-cuda}"
LOG="${BUILD}/ctest-verbose.log"
rm -rf "${BUILD}"
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DQW3_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${QW3_CUDA_ARCH}"
cmake --build "${BUILD}" -j"${QW3_BUILD_JOBS:-2}"
mkdir -p "${BUILD}"
set +e
ctest --test-dir "${BUILD}" -V 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}
set -e
(( status == 0 )) || exit "${status}"
if grep -Eiq '(^|[^a-z])(skip|skipped|no cuda device|no gpu)([^a-z]|$)' "${LOG}"; then
  echo 'GPU release gate rejected an apparent skipped CUDA test' >&2
  exit 3
fi

PORT="${QW3_RELEASE_PORT:-18080}"
SERVER_LOG="${BUILD}/release-server.log"
"${BUILD}/qw3" serve \
  --model "${QW3_MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --kvmem-profile agent-safe \
  --ctx "${QW3_RELEASE_CTX:-32768}" \
  --kvmem-budget "${QW3_RELEASE_BUDGET:-8192}" \
  --kvmem-prefill-budget "${QW3_RELEASE_PREFILL_BUDGET:-8192}" \
  --kvmem-gen-budget "${QW3_RELEASE_GEN_BUDGET:-512}" \
  >"${SERVER_LOG}" 2>&1 &
pid=$!
trap 'kill ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then break; fi
  if ! kill -0 "${pid}" 2>/dev/null; then cat "${SERVER_LOG}" >&2; exit 4; fi
  sleep 1
done
status_json="$(curl -fsS "http://127.0.0.1:${PORT}/v1/kvmem/status")"
python3 - "${status_json}" <<'PY'
import json, sys
x=json.loads(sys.argv[1])
assert x['enabled'] is True
assert x['profile'] == 'agent-safe'
assert x['state_coherence'] == 'selected-replay'
assert x['strict_retrieval'] is True
assert x['method'] == 'retrieval'
assert x['retrieval_method'] == 'mean-k'
assert x['kvmi_012_applies'] is False
assert x['session_policy'] == 'detachable-host-state'
assert x['session_max'] >= 1
assert x['session_scheduler'] == 'lru-affinity-fail-fast'
assert x['executor_slots_configured'] == 1
assert x['executor_slots_certified'] == 1
assert x['physical_executor_runtime_abi'] == 'executor-slot-runtime-v2'
assert x['physical_executor_runtimes_installed'] == 1
assert x['registered_session_host_bytes'] >= 0
assert x['resource_policy'] == 'fail-fast-byte-envelope'
assert x['resource_max_inflight'] >= 1
assert x['resource_rejections'] >= 0
assert x['scheduler_faulted_releases'] >= 0
assert x['scheduler_forced_cold_resets'] >= 0
print('KVMem RC9.2 status contract: PASS')
PY

scheduler_json="$(curl -fsS "http://127.0.0.1:${PORT}/v1/kvmem/scheduler")"
python3 - "${scheduler_json}" <<'PY'
import json, sys
x=json.loads(sys.argv[1])
assert x['object'] == 'kvmem.scheduler'
assert x['policy'] == 'lru-affinity-fail-fast'
assert x['slot_count'] == 1
assert x['busy_slots'] == 0
assert len(x['slots']) == 1
assert x['faulted_releases'] >= 0
assert x['forced_cold_resets'] >= 0
assert 'dirty' in x['slots'][0]
assert 'active_lease_id' in x['slots'][0]
print('KVMem RC9.2 scheduler contract: PASS')
PY

executor_json="$(curl -fsS "http://127.0.0.1:${PORT}/v1/kvmem/executors")"
python3 - "${executor_json}" <<'PY'
import json, sys
x=json.loads(sys.argv[1])
assert x['object'] == 'kvmem.executor_pool'
assert x['scope'] == 'persistent-session-runtime'
assert x['runtime_abi'] == 'executor-slot-runtime-v2'
assert x['configured_slots'] == 1
assert x['certified_slots'] == 1
assert x['installed_runtimes'] == 1
assert len(x['runtimes']) == 1
assert x['runtimes'][0]['installed'] is True
assert x['runtimes'][0]['runtime_id'] == 'qw3-executor-slot-0'
assert 'active_lease_id' in x['runtimes'][0]
assert 'lease_sequence' in x['runtimes'][0]
assert 'linked_scheduler_lease_id' in x['runtimes'][0]
print('KVMem RC9.2 physical executor contract: PASS')
PY

resource_json="$(curl -fsS "http://127.0.0.1:${PORT}/v1/kvmem/resources")"
python3 - "${resource_json}" <<'PY'
import json, sys
x=json.loads(sys.argv[1])
assert x['object'] == 'kvmem.resources'
assert x['policy'] == 'fail-fast-byte-envelope'
assert x['max_inflight'] >= 1
assert x['inflight'] == 0
assert x['admissions'] >= 0
assert x['rejections'] >= 0
print('KVMem RC9.2 resource contract: PASS')
PY

if [[ "${QW3_SKIP_SELECTED_REPLAY_GATE:-0}" != "1" ]]; then
  python3 "${ROOT}/scripts/release/selected_replay_gate.py" \
    "http://127.0.0.1:${PORT}"
fi

echo "GPU release gate: PASS (component tests + agent-safe selected-replay preflight)"
