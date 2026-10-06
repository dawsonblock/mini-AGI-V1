#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src-python'))
from kvcontinual.execution.macos_qualification_preflight import run_macos_qualification_preflight,write_preflight
p=argparse.ArgumentParser();p.add_argument('--backend-fingerprint',required=True);p.add_argument('--require-mlx',action='store_true');p.add_argument('--out')
a=p.parse_args();r=run_macos_qualification_preflight(backend_fingerprint=a.backend_fingerprint,require_mlx=a.require_mlx)
if a.out:write_preflight(r,a.out)
print(json.dumps(r.to_dict(),indent=2));sys.exit(0 if r.passed else 2)
