#!/usr/bin/env bash
# v16.4.8 — run the real-boundary qualification suite on a prepared
# Linux host. Requires: host_check.sh PASS, nsjail built+pinned via
# build_nsjail.sh, MINAGI_NSJAIL[_DIGEST] set.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MINAGI_NSJAIL:=/usr/local/bin/nsjail}"
export MINAGI_NSJAIL
if [ -z "${MINAGI_NSJAIL_DIGEST:-}" ] && [ -f /usr/local/bin/nsjail-build-record.json ]; then
  MINAGI_NSJAIL_DIGEST=$(python3 -c "import json;print(json.load(open('/usr/local/bin/nsjail-build-record.json'))['binary_sha256'])")
  export MINAGI_NSJAIL_DIGEST
fi
export PYTHONPATH="src-python:tests-python/helpers"

echo "== host check =="
bash scripts/qualification/host_check.sh || {
  echo "host check failed — fix before qualifying" >&2; exit 1; }

echo "== executor preflight (real boundary) =="
python3 - <<'PY'
import os, sys
sys.path.insert(0, "src-python")
from minagi.runtime.isolation_profile import load_profile
from minagi.runtime.nsjail_executor import NsJailWorkerExecutor
from minagi.runtime.isolation_preflight import run_preflight
from minagi.runtime.worker_isolation import DEFAULT_ISOLATION
import hashlib
binary = os.environ["MINAGI_NSJAIL"]
digest = os.environ.get("MINAGI_NSJAIL_DIGEST") or (
    "sha256:" + hashlib.sha256(open(binary,"rb").read()).hexdigest())
euid = os.geteuid()
# The worker identity MUST differ from the supervisor's — a same-uid
# boundary is no boundary. Root supervisors default to 65534 (nobody);
# non-root supervisors must set MINAGI_WORKER_UID explicitly.
wuid = os.environ.get("MINAGI_WORKER_UID")
if not wuid:
    if euid == 0:
        wuid = "65534"
    else:
        raise SystemExit("MINAGI_WORKER_UID is required — it must name "
                         "a dedicated uid distinct from the supervisor")
wgid = os.environ.get("MINAGI_WORKER_GID") or wuid
if int(wuid) == euid:
    raise SystemExit("MINAGI_WORKER_UID must differ from the "
                     "supervisor uid")
ex = NsJailWorkerExecutor.from_profile(
    load_profile("configs/isolation/linux-production-offline-v1.json"),
    worker_state_root="/var/lib/minagi/workers",
    nsjail_path=binary, nsjail_digest=digest,
    worker_uid=int(wuid), worker_gid=int(wgid),
    user_manager=euid != 0,
    python_path_entries=tuple(p for p in sys.path if p))
report = run_preflight(ex, isolation=DEFAULT_ISOLATION, timeout=180.0,
                       probe_args=["deny:/etc/shadow", "deny:/root",
                                   "read:/proc/self/status",
                                   "net:LISTENER",   # live host listener
                                   "detach:/work/detached.marker"])
import json; print(json.dumps(report, indent=2, sort_keys=True))
sys.exit(0 if report["qualified"] else 1)
PY

# The gated suites inherit the SAME distinct worker identity — the
# boundary under test is the one production will run.
if [ "$(id -u)" = "0" ]; then
    export MINAGI_WORKER_UID="${MINAGI_WORKER_UID:-65534}"
    export MINAGI_WORKER_GID="${MINAGI_WORKER_GID:-$MINAGI_WORKER_UID}"
fi
echo "== adversarial boundary tests =="
python3 -m pytest tests-python/runtime/test_v1648_linux_nsjail.py -v
echo "== real-model lifecycle (G10) =="
python3 -m pytest tests-python/runtime/test_v1649_real_model_nsjail.py -v
echo "QUALIFICATION: PASS"
