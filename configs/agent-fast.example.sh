#!/usr/bin/env bash
set -euo pipefail
: "${QW3_MODEL:?set QW3_MODEL to a supported Qwen model}"
exec ./build/qw3 serve \
  --model "$QW3_MODEL" \
  --kvmem-profile agent-fast \
  --ctx "${QW3_CTX:-262144}" \
  --kvmem-budget "${QW3_KVMEM_BUDGET:-65536}" \
  --kvmem-gen-budget "${QW3_KVMEM_GEN_BUDGET:-8192}" \
  --host 127.0.0.1 \
  --port "${QW3_PORT:-8080}"
