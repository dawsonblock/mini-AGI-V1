#!/usr/bin/env bash
# v16.4.8 Step 3 — Linux qualification-host capability check.
# Run on the candidate host BEFORE building NsJail. Prints a
# PASS/FAIL line per requirement; exits non-zero if any hard
# requirement is absent.
set -u
fail=0
say() { printf '%-52s %s\n' "$1" "$2"; }
req() { # req <name> <ok?> <detail>
  if [ "$2" -eq 0 ]; then say "$1" "PASS  $3"
  else say "$1" "FAIL  $3"; fail=1; fi
}

echo "== kernel =="
uname -a; uname -m

echo "== identity / capabilities =="
python3 - <<'PY'
import re
s = open("/proc/self/status").read()
for k in ("Uid","Gid","CapEff","NoNewPrivs","Seccomp"):
    m = re.search(rf"^{k}:\s*(.*)$", s, re.M)
    print(f"{k}: {m.group(1) if m else '?'}")
PY
CAPEFF=$(python3 -c "print(int(open('/proc/self/status').read().split('CapEff:')[1].split()[0],16))")
[ $((CAPEFF & (1 << 21))) -ne 0 ] && CAP=0 || CAP=1
req "CAP_SYS_ADMIN (bit21) effective" $CAP \
    "userns fallback possible if unset" ; CAP=0

echo "== cgroup =="
CGTYPE=$(stat -fc %T /sys/fs/cgroup)
findmnt /sys/fs/cgroup | tail -2
[ "$CGTYPE" = "cgroup2fs" ] && CG=0 || CG=1
req "cgroup v2 unified hierarchy" $CG "stat -f: $CGTYPE"
[ -f /sys/fs/cgroup/cgroup.controllers ] && C2=0 || C2=1
req "cgroup.controllers present" $C2 ""
[ -w /sys/fs/cgroup ] && CW=0 || CW=1
req "cgroup root writable (or delegated via MINAGI_CGROUP_ROOT)" $CW \
    "MINAGI_CGROUP_ROOT may name a delegated subtree"

echo "== systemd =="
systemctl --version | head -1 || true
command -v systemd-run >/dev/null && SR=0 || SR=1
req "systemd-run on PATH" $SR "$(command -v systemd-run || echo missing)"
command -v systemctl >/dev/null && SC=0 || SC=1
req "systemctl on PATH" $SC "$(command -v systemctl || echo missing)"
systemctl --user is-active default.target >/dev/null 2>&1 && UM=0 || UM=1
[ $UM -eq 0 ] || [ "$(id -u)" -eq 0 ] && UMR=0 || UMR=1
req "system manager reachable (system or user)" $UMR \
    "user manager: $([ $UM -eq 0 ] && echo active || echo inactive)"

echo "== namespaces =="
ls -l /proc/self/ns/ | awk '{print $9, $11}' | tail -n +2
for n in user net pid mnt ipc uts; do
  [ -e "/proc/self/ns/$n" ] || { say "ns/$n" "FAIL  missing"; fail=1; }
done
UCLONE=$(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || echo "?")
MAXNS=$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo "?")
say "unprivileged_userns_clone" "${UCLONE:-?}"
say "max_user_namespaces" "${MAXNS:-?}"

echo "== python =="
python3 --version

echo
[ $fail -eq 0 ] && echo "HOST CHECK: PASS" || echo "HOST CHECK: FAIL"
exit $fail
