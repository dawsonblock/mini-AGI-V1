#!/usr/bin/env bash
set -euo pipefail

: "${QW3_MODEL:?set QW3_MODEL to a supported Qwen model}"

exec ./build/qw3 serve \
  --model "$QW3_MODEL" \
  --kvmem-profile agent-safe \
  --ctx "${QW3_CTX:-262144}" \
  --kvmem-budget "${QW3_KVMEM_BUDGET:-65536}" \
  --kvmem-gen-budget "${QW3_KVMEM_GEN_BUDGET:-8192}" \
  --kvmem-cpu-gb "${QW3_KVMEM_CPU_GB:-16}" \
  --kvmem-nvme-dir "${QW3_KVMEM_NVME_DIR:-/mnt/nvme/kvmem}" \
  --kvmem-nvme-gb "${QW3_KVMEM_NVME_GB:-64}" \
  --host 127.0.0.1 \
  --port "${QW3_PORT:-8080}"
