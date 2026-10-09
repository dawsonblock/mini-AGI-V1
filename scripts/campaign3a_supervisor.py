#!/usr/bin/env python3
"""Campaign 3A Colab lane supervisor (operational, resumable).

Keeps one Colab session executing the preregistered campaign3a-v166
seeds, banks each completed seed's evidence + adapters + arms as a
tarball on the VM, downloads them locally, and relaunches the runner if
it dies while seeds remain (the runner skips already-COMPLETE seeds, so
relaunch is safe — per-seed atomicity survives VM reclamation).

The holdout file is NOT needed here: the runner never opens it; only
qualification consumes it (`qualify_campaign1.py --holdout`).

Usage:
  python scripts/campaign3a_supervisor.py --session v166-c3a-1 \
      --seeds 0,1,2,3,4,5,6,7,8,9 --ckpt-dir /tmp/c3a_ckpt

Exit codes: 0 all requested seeds banked, 3 session unreachable /
stopped with seeds outstanding, 4 wait budget exhausted.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

REMOTE = "/content/minagi_campaign3a"
CID = "campaign3a-v166"
SRC = "/content/miniagi-src"
LOG = "/content/campaign3a.log"

VM_BASH = r'''
CD={remote}
CID={cid}
mkdir -p /content/ck3a
echo "== status =="
if pgrep -f "scripts/run_campaign1.py" >/dev/null; then
  echo "runner: RUNNING ($(ps -o etime= -p $(pgrep -f 'scripts/run_campaign1.py' | head -1) | tr -d ' '))"
else
  echo "runner: NOT RUNNING"
fi
echo "complete: $(ls $CD/campaigns/$CID/seed-*/COMPLETE 2>/dev/null | wc -l)/{n_seeds}"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)"
for s in $(ls -d $CD/campaigns/$CID/seed-* 2>/dev/null | grep -v staging); do
  name=$(basename $s)
  if [ -f "$s/COMPLETE" ] && [ ! -f "/content/ck3a/ck_${{name}}.tar.gz" ]; then
    members="campaigns/$CID/$name"
    for a in L6 NC; do
      [ -d "$CD/adapters/$CID/$a/$name" ] && members="$members adapters/$CID/$a/$name"
    done
    [ -d "$CD/arms/$CID/$name" ] && members="$members arms/$CID/$name"
    tar -czf "/content/ck3a/ck_${{name}}.tar.gz" -C $CD $members && echo "banked $name"
  fi
done
if [ ! -f /content/ck3a/ck_meta.tar.gz ]; then
  tar -czf /content/ck3a/ck_meta.tar.gz -C $CD .keys campaigns/$CID/CAMPAIGN_PLAN.json \
      campaigns/$CID/DATASET_PROOF.json campaigns/$CID/EXPERIMENT_PROTOCOL.json \
      campaigns/$CID/EXECUTION_PUBLIC_KEY.bin campaigns/$CID/EXECUTION_KEY_ID.txt \
      trust_root.json 2>/dev/null && echo "banked meta"
fi
ls /content/ck3a/ 2>/dev/null | sed 's/ck_//;s/.tar.gz//' | tr '\n' ' '; echo
'''

# colab exec runs the uploaded file as a Python cell — wrap the bash.
VM_SCRIPT = (
    "import subprocess\n"
    "BASH = r'''" + VM_BASH + "'''\n"
    "r = subprocess.run(['bash','-lc',BASH], capture_output=True, "
    "text=True, timeout=600)\n"
    "print(r.stdout)\n"
    "if r.stderr.strip(): print('stderr:', r.stderr[-300:])\n"
)

RELAUNCH_BASH = (
    f"cd {SRC} && nohup python3 scripts/run_campaign1.py "
    f"--config configs/campaign3a.yaml --storage {REMOTE} "
    "--execute-seeds {seeds} > " + LOG + " 2>&1 < /dev/null & echo RELAUNCHED"
)
# Popen with start_new_session detaches; the nohup'd runner survives the
# cell and keeps logging to LOG.
RELAUNCH = (
    "import subprocess\n"
    "BASH = r'''" + RELAUNCH_BASH + "'''\n"
    "subprocess.Popen(['bash','-lc',BASH], start_new_session=True, "
    "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "print('RELAUNCHED')\n"
)


def _run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


def vm_exec(session: str, script: str, timeout=600):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    r = _run(["colab", "exec", "--session", session, "--file", path,
              "--timeout", str(min(timeout, 540))], timeout=timeout + 60)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ckpt-dir", default="/tmp/c3a_ckpt")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between cycles")
    ap.add_argument("--max-wait", type=int, default=0,
                    help="give up after this many seconds (0 = forever)")
    ap.add_argument("--relaunch-dead-runner", action="store_true",
                    default=True)
    args = ap.parse_args(argv)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    wanted = {f"seed-{s}" for s in seeds}
    ckpt = Path(args.ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    script = VM_SCRIPT.format(remote=REMOTE, cid=CID, n_seeds=len(seeds))
    started = time.time()

    while True:
        try:
            rc, out, err = vm_exec(args.session, script)
        except subprocess.TimeoutExpired:
            rc, out, err = 1, "", "colab exec timed out"
        if rc != 0 and "not found" in (out + err).lower():
            print(f"[supervisor] session unreachable: {out[-200:]}{err[-200:]}")
            return 3
        print(f"----- cycle @ {time.strftime('%H:%M:%S')} -----")
        print(out.strip())
        if err.strip():
            print("stderr:", err.strip()[-300:])

        # download any tarballs not yet local
        rc, out, _ = vm_exec(args.session, "import os; print(' '.join("
                                          "f for f in os.listdir('/content/ck3a') "
                                          "if f.endswith('.tar.gz')))")
        remote_names = out.strip().split() if rc == 0 else []
        for name in remote_names:
            local = ckpt / name
            if not local.is_file():
                r = _run(["colab", "download", "--session", args.session,
                          f"/content/ck3a/{name}", str(local)], timeout=900)
                if r.returncode == 0:
                    print(f"[supervisor] downloaded {name}")

        banked = {p.name for p in ckpt.glob("ck_seed-*.tar.gz")}
        done = {n for n in banked
                if n.removeprefix("ck_").removesuffix(".tar.gz") in wanted}
        if len(done) == len(wanted):
            print(f"[supervisor] all {len(wanted)} seeds banked locally")
            return 0

        if "runner: NOT RUNNING" in out and args.relaunch_dead_runner:
            print("[supervisor] runner dead with seeds outstanding — "
                  "relaunching (COMPLETE seeds are skipped)")
            vm_exec(args.session, RELAUNCH.format(seeds=args.seeds))

        if args.max_wait and time.time() - started > args.max_wait:
            print(f"[supervisor] wait budget exhausted; "
                  f"{len(done)}/{len(wanted)} seeds banked")
            return 4
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
