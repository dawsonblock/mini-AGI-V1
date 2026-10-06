#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
"$ROOT/scripts/rc11/validate_all.sh"
python3 "$ROOT/tools/verify_release.py" "$ROOT"
