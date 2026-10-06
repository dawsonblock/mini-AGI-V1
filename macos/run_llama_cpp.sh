#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
if [[ -z "$LLAMA_SERVER" ]]; then
  echo "llama-server not found. Run macos/install.sh or set LLAMA_SERVER=/path/to/llama-server." >&2
  exit 2
fi

PORT="${QW3_MAC_BACKEND_PORT:-8080}"
CTX="${QW3_MAC_CONTEXT:-32768}"
PARALLEL="${QW3_MAC_PARALLEL:-1}"

LORA_ARGS=()
if [[ -n "${QW3_LLAMA_LORA:-}" ]]; then
  LORA_ARGS+=(--lora "$QW3_LLAMA_LORA")
fi
if [[ "${QW3_LLAMA_LORA_HOT_SWITCH:-0}" == "1" ]]; then
  LORA_ARGS+=(--lora-init-without-apply)
fi

if [[ -n "${QW3_MODEL:-}" ]]; then
  exec "$LLAMA_SERVER" \
    -m "$QW3_MODEL" \
    -ngl "${QW3_MAC_GPU_LAYERS:-999}" \
    -c "$CTX" -np "$PARALLEL" \
    "${LORA_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT"
elif [[ -n "${QW3_HF_MODEL:-}" ]]; then
  exec "$LLAMA_SERVER" \
    -hf "$QW3_HF_MODEL" \
    -ngl "${QW3_MAC_GPU_LAYERS:-999}" \
    -c "$CTX" -np "$PARALLEL" \
    "${LORA_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT"
else
  echo "Set QW3_MODEL=/path/to/model.gguf or QW3_HF_MODEL=<repo-or-repo:file>." >&2
  exit 2
fi
