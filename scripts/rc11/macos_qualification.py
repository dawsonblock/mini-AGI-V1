#!/usr/bin/env python3
"""Collect a canonical Apple-Silicon qualification fingerprint.

This does not claim HYPIC correctness by itself. It records the machine/toolchain
identity that must be bound into later differential qualification receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def cmd(argv):
    try: return subprocess.check_output(argv,stderr=subprocess.STDOUT,text=True,timeout=15).strip()
    except Exception as e: return f'UNAVAILABLE:{type(e).__name__}'

def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def torch_mps_available():
    try:
        import torch
        return bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    except Exception:
        return False

def package_version(name):
    try: return importlib.metadata.version(name)
    except Exception: return None

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default=None); ap.add_argument('--allow-non-macos-probe',action='store_true'); a=ap.parse_args()
    is_macos=sys.platform=='darwin'; machine=platform.machine().lower(); is_apple=is_macos and machine in ('arm64','aarch64')
    if not is_apple and not a.allow_non_macos_probe:
        raise SystemExit('qualification requires macOS on Apple Silicon; use --allow-non-macos-probe only to test the probe')
    metal_path=cmd(['xcrun','--find','metal']) if shutil.which('xcrun') else 'UNAVAILABLE:not-found'
    data={
      'schema_version':3,'platform':sys.platform,'machine':machine,'python':platform.python_version(),
      'macos_version':platform.mac_ver()[0] if is_macos else None,
      'sw_vers':cmd(['sw_vers']) if is_macos else 'UNAVAILABLE:not-macos',
      'os_build':cmd(['sw_vers','-buildVersion']) if is_macos else 'UNAVAILABLE:not-macos',
      'hardware_model':cmd(['sysctl','-n','hw.model']) if is_macos else 'UNAVAILABLE:not-macos',
      'cpu_brand':cmd(['sysctl','-n','machdep.cpu.brand_string']) if is_macos else platform.processor(),
      'hw_memsize':cmd(['sysctl','-n','hw.memsize']) if is_macos else 'UNAVAILABLE:not-macos',
      'xcodebuild':cmd(['xcodebuild','-version']) if shutil.which('xcodebuild') else 'UNAVAILABLE:not-found',
      'metal_compiler':metal_path,
      'clang':cmd(['clang','--version']).splitlines()[0] if shutil.which('clang') else 'UNAVAILABLE:not-found',
      'torch_version':package_version('torch'),
      'mlx_version':package_version('mlx'),
      'apple_silicon':is_apple,
      'metal_available':is_apple and not metal_path.startswith('UNAVAILABLE:'),
      'mps_available':is_apple and torch_mps_available(),
      'mlx_installed':importlib.util.find_spec('mlx') is not None,
    }
    data['hardware_toolchain_fingerprint']='sha256:'+hashlib.sha256(canonical(data)).hexdigest()
    text=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if a.out: Path(a.out).write_text(text)
    print(text,end='')
    return 0 if is_apple or a.allow_non_macos_probe else 2
if __name__=='__main__': raise SystemExit(main())
