#!/usr/bin/env bash
# Mini-AGI host release gate.
#
# The full CPU-side qualification a release must pass before packaging:
#   1. source audit        — release identity files, version coherence,
#                            no private keys / weights / dev paths
#   2. native build+ctest  — the bundled C++ tree (host profile, no CUDA)
#   3. full python suite   — tests-python against src-python
#   4. manifest verify     — governed digests + Ed25519 signature +
#                            attestation reconciliation
#
# GPU qualification is a separate gate (gpu_gate.sh) — this script is
# the host profile and must pass on every release host.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${MINAGI_HOST_BUILD_DIR:-${ROOT}/build-release-host}"
JOBS="${MINAGI_BUILD_JOBS:-4}"

rm -rf "${BUILD}"

python3 "${ROOT}/scripts/release/source_audit.py"

CMAKE_GEN=()
if command -v ninja >/dev/null 2>&1; then
  CMAKE_GEN=(-G Ninja)
fi
cmake -S "${ROOT}" -B "${BUILD}" "${CMAKE_GEN[@]}" \
  -DCMAKE_BUILD_TYPE="${MINAGI_HOST_GATE_BUILD_TYPE:-Release}" \
  -DQW3_ENABLE_CUDA=OFF -DQW3_BUILD_TESTS=ON
cmake --build "${BUILD}" -j"${JOBS}"
ctest --test-dir "${BUILD}" --output-on-failure

(cd "${ROOT}" && PYTHONPATH=src-python:tests-python \
  python3 -m pytest tests-python -q)

# Manifest+signature verification runs against a git-archive extraction
# — exactly the tracked, governed file set. The live dev tree carries
# ungoverned debris (.vscode/, evidence downloads, dist/) that is not
# part of the release contract and must not fail the gate.
STAGE="$(mktemp -d -t minagi-host-gate.XXXXXX)"
trap 'rm -rf "${STAGE}"' EXIT
git -C "${ROOT}" archive HEAD | tar -x -C "${STAGE}"
python3 "${ROOT}/scripts/release/verify_manifest.py" --root "${STAGE}"
echo "host release gate: PASS"
