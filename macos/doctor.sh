#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$ROOT/.venv-mac/bin/python" ]]; then
  exec "$ROOT/.venv-mac/bin/python" "$ROOT/macos/doctor.py"
fi
exec python3 "$ROOT/macos/doctor.py"
