#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${QW3_HOST_BUILD_DIR:-${ROOT}/build-release-host}"
rm -rf "${BUILD}"
python3 "${ROOT}/scripts/release/source_audit.py"
cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${QW3_HOST_GATE_BUILD_TYPE:-Debug}" \
  -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g0" \
  -DQW3_ENABLE_CUDA=OFF -DQW3_BUILD_TESTS=ON
cmake --build "${BUILD}" -j"${QW3_BUILD_JOBS:-2}"
ctest --test-dir "${BUILD}" --output-on-failure
if command -v pytest >/dev/null 2>&1; then
  (cd "${ROOT}/continual" && PYTHONPATH=src pytest -q)
else
  python3 -m pytest --version >/dev/null 2>&1 || { echo 'pytest is required for RC10 continual-plane qualification' >&2; exit 6; }
  (cd "${ROOT}/continual" && PYTHONPATH=src python3 -m pytest -q)
fi
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-strict-retrieval'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-exactmass-raw-key-max-mib'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-profile'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-state-coherence'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-executor-slots'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-resource-vram-bytes'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-resource-host-bytes'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-resource-nvme-bytes'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-resource-max-inflight'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-session-max'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-session-host-token-limit'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-session-host-byte-limit'
"${BUILD}/qw3" --help 2>&1 | grep -q -- '--kvmem-session-snapshot-dir'
set +e
slot_err="$("${BUILD}/qw3" serve --model /nonexistent/model.gguf \
  --kvmem --kvmem-executor-slots 2 2>&1)"
slot_status=$?
set -e
(( slot_status != 0 )) || { echo 'uncertified multi-executor configuration unexpectedly passed' >&2; exit 5; }
grep -q 'certifies exactly one CUDA slot' <<<"${slot_err}"
set +e
resource_err="$("${BUILD}/qw3" serve --model /nonexistent/model.gguf \
  --kvmem --kvmem-resource-max-inflight -1 2>&1)"
resource_status=$?
set -e
(( resource_status != 0 )) || { echo 'negative KVMem resource inflight limit unexpectedly passed' >&2; exit 5; }
grep -q -- '--kvmem-resource-max-inflight must be >= 0' <<<"${resource_err}"
set +e
exactmass_cap_err="$("${BUILD}/qw3" serve --model /nonexistent/model.gguf \
  --kvmem --kvmem-exactmass-raw-key-max-mib -1 2>&1)"
exactmass_cap_status=$?
set -e
(( exactmass_cap_status != 0 )) || { echo 'negative ExactMass raw-key cap unexpectedly passed' >&2; exit 5; }
grep -q -- '--kvmem-exactmass-raw-key-max-mib must be >= 0' <<<"${exactmass_cap_err}"
set +e
resource_slots_err="$("${BUILD}/qw3" serve --model /nonexistent/model.gguf \
  --kvmem --kvmem-resource-max-inflight 2 2>&1)"
resource_slots_status=$?
set -e
(( resource_slots_status != 0 )) || { echo 'resource inflight > executor slots unexpectedly passed' >&2; exit 5; }
grep -q 'resource max_inflight cannot exceed certified executor slots' <<<"${resource_slots_err}"
set +e
profile_err="$("${BUILD}/qw3" serve --model /nonexistent/model.gguf \
  --kvmem-profile agent-safe --continuous-batching 2>&1)"
profile_status=$?
set -e
(( profile_status != 0 )) || { echo 'agent-safe incompatible override unexpectedly passed' >&2; exit 5; }
grep -Eq 'agent-safe profile contract was overridden|state coherence query-replay/selected-replay is single-request only' <<<"${profile_err}"
python3 "${ROOT}/scripts/release/verify_manifest.py"
echo "host release gate: PASS"
