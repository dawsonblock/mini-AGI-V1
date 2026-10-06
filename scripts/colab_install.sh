#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install -U pip setuptools wheel
python -m pip install -e "$ROOT[colab]"
# Colab images ship torchao 0.10.x; peft>=0.14 hard-requires >=0.16 when
# torchao is installed (LoRA injection dispatches through peft.tuners.torchao).
if python -c "import torchao" 2>/dev/null; then
  python -m pip install -U "torchao>=0.16"
fi
echo "Installed mini-AGI v16.1 Colab convergence build."
