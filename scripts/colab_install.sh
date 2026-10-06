#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -U pip setuptools wheel
python -m pip install -e "$ROOT[colab]"
echo "Installed mini-AGI v16.1 Colab convergence build."
