#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer targets Apple Silicon macOS (Darwin arm64)." >&2; exit 2
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and rerun." >&2; exit 2
fi
brew install cmake ninja python@3.12 curl jq openssl@3 || true
PYTHON="${QW3_MAC_PYTHON:-/opt/homebrew/bin/python3.12}"; [[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
rm -rf "$ROOT/.venv"
"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT/runtime[dev,security]"
if [[ "${QW3_MAC_INSTALL_MLX:-1}" == "1" ]]; then "$ROOT/.venv/bin/python" -m pip install --upgrade "mlx-lm[train]"; fi
if [[ "${QW3_MAC_INSTALL_LLAMA_CPP:-1}" == "1" ]] && ! command -v llama-server >/dev/null 2>&1; then brew install llama.cpp || true; fi
echo "Installed RC11.6 unified runtime environment: .venv"
echo "Next: $ROOT/macos/doctor.sh"
