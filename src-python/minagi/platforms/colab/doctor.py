from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from .environment import probe_environment

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--output",default="READINESS_REPORT.json"); args=p.parse_args(argv)
    env=probe_environment(); reasons=[]
    if env.torch_version=="unavailable": reasons.append("torch_unavailable")
    if not env.cuda_available: reasons.append("cuda_unavailable")
    report={"schema":"mini-agi-v16.1-readiness-report-v1","environment":asdict(env),"environment_digest":env.digest,"ready_for_gpu_campaign":env.cuda_available and not reasons,"reasons":reasons}
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)); print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not reasons else 2
if __name__=="__main__": raise SystemExit(main())
