#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/.venv-mac/bin/activate"
MODEL="${QW3_MLX_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set QW3_MLX_MODEL to a current MLX Qwen model or local converted model." >&2
  exit 2
fi
PORT="${QW3_MAC_BACKEND_PORT:-8080}"
exec mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$PORT"
