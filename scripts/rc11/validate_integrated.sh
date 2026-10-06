#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
"$ROOT/scripts/rc11/validate_all.sh"
rm -rf "$ROOT/build-integration"
cmake -S "$ROOT" -B "$ROOT/build-integration" -DQW3_BUILD_TESTS=ON -DQW3_ENABLE_CUDA=OFF
cmake --build "$ROOT/build-integration" --target qw3-gdn-reference qw3-gdn-segment-io qw3-gdn-metal-interface qw3-gdn-summary-runtime qw3-gdn-runtime-bundle qw3-gdn-summarize qw3-gdn-runtime-cli -j2
for t in qw3-gdn-reference qw3-gdn-segment-io qw3-gdn-metal-interface qw3-gdn-summary-runtime qw3-gdn-runtime-bundle; do "$ROOT/build-integration/$t"; done
python3 -m compileall -q "$ROOT/src-python" "$ROOT/scripts/rc11"
python3 "$ROOT/scripts/rc11/integration_smoke.py"
echo 'RC11.7 integrated validation passed.'
