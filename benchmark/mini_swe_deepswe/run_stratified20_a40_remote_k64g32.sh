#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${QW3_REMOTE_SSH_HOST:?set QW3_REMOTE_SSH_HOST (for example user@gpu-host)}"
: "${QW3_REMOTE_BINARY:?set QW3_REMOTE_BINARY to the remote qw3 binary}"

export QW3_DEEPSWE_BINARY="${QW3_DEEPSWE_BINARY:-${SCRIPT_DIR}/remote_qw3_ssh_wrapper.py}"
export QW3_REMOTE_SSH_PORT="${QW3_REMOTE_SSH_PORT:-22}"
export QW3_REMOTE_WORKDIR="${QW3_REMOTE_WORKDIR:-~}"
export QW3_REMOTE_LISTEN_HOST="${QW3_REMOTE_LISTEN_HOST:-127.0.0.1}"
export QW3_REMOTE_LISTEN_PORT="${QW3_REMOTE_LISTEN_PORT:-8000}"
PYTHON_BIN="${QW3_DEEPSWE_PYTHON:-python3}"

exec "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/run_requestplan10.py" \
  --run-name "${RUN_NAME:-stratified20_kvmem_k64g32_official}" \
  --tasks-file "${SCRIPT_DIR}/stratified20_seed20260831.json" \
  --mode kvmem \
  --seed 1000 \
  --guided-query-tokens 4096 \
  --kvmem-budget 65536 \
  --kvmem-prefill-budget 65536 \
  --kvmem-gen-budget 32768 \
  --host "${QW3_DEEPSWE_HOST:-172.17.0.1}" \
  --port "${QW3_DEEPSWE_PORT:-8010}" \
  --api-host "${QW3_DEEPSWE_API_HOST:-172.17.0.1.nip.io}" \
  --api-port "${QW3_DEEPSWE_API_PORT:-443}" \
  "$@"
