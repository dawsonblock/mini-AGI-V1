#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${QW3_MAC_BUILD_DIR:-$ROOT/build-macos-host}"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=OFF \
  -DQW3_BUILD_TESTS=ON
cmake --build "$BUILD" -j "${QW3_BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}"
ctest --test-dir "$BUILD" --output-on-failure

echo "Host-only macOS build passed. Native generation remains delegated to Metal/MLX." 
