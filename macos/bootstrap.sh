#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/macos/install.sh"
"$ROOT/macos/doctor.sh"
mkdir -p "$ROOT/data/training" "$ROOT/artifacts/adapters/candidates" "$ROOT/artifacts/adapters/promoted"
echo "Bootstrap complete. Start a backend with: macos/qw3-mac backend-llama"
echo "Then start the continual gateway with: macos/qw3-mac gateway"
