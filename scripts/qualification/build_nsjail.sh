#!/usr/bin/env bash
# v16.4.8 Step 4 — build NsJail from a REVIEWED, PINNED commit and
# record the binary hash. Usage:
#   ./build_nsjail.sh <APPROVED_COMMIT_SHA> [install_prefix]
# Records: commit, binary sha256, build options, dep versions to
# nsjail-build-record.json next to the installed binary.
set -euo pipefail
COMMIT="${1:?usage: build_nsjail.sh <APPROVED_COMMIT_SHA> [prefix]}"
PREFIX="${2:-/usr/local}"
case "$COMMIT" in *[!0-9a-f]*|"") echo "refusing non-SHA ref: pin an exact commit" >&2; exit 2;; esac

if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y git build-essential autoconf automake \
    bison flex libprotobuf-dev protobuf-compiler \
    libnl-route-3-dev libtool pkg-config
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone https://github.com/google/nsjail.git "$WORK/nsjail"
cd "$WORK/nsjail"
git checkout "$COMMIT"
make -j"$(nproc)"
./nsjail --help >/dev/null
SHA=$(sha256sum ./nsjail | awk '{print $1}')
sudo install -m 0755 ./nsjail "$PREFIX/bin/nsjail"

python3 - "$COMMIT" "$SHA" "$PREFIX" <<'PY'
import json, subprocess, sys, time
commit, sha, prefix = sys.argv[1:4]
def ver(cmd):
    try: return subprocess.run(cmd, capture_output=True,
                               text=True).stdout.splitlines()[0]
    except Exception: return "?"
rec = {
  "nsjail_commit": commit,
  "binary": f"{prefix}/bin/nsjail",
  "binary_sha256": f"sha256:{sha}",
  "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "deps": {"protoc": ver(["protoc","--version"]),
           "gcc": ver(["gcc","--version"]),
           "libnl": ver(["pkg-config","--modversion","libnl-route-3.0"])},
}
out = f"{prefix}/bin/nsjail-build-record.json"
with open("/tmp/nsjail-build-record.json","w") as f: json.dump(rec,f,indent=2)
print(json.dumps(rec, indent=2))
print(f"record written: /tmp/nsjail-build-record.json -> {out}")
PY
sudo install -m 0644 /tmp/nsjail-build-record.json \
  "$PREFIX/bin/nsjail-build-record.json"
echo
echo "PIN THIS DIGEST in service config:"
echo "  MINAGI_NSJAIL_DIGEST=sha256:$SHA"
echo "  (or --nsjail-digest sha256:$SHA)"
