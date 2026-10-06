#!/usr/bin/env python3
"""Build a reproducible ZIP from a verified RC11 tree."""
from __future__ import annotations
import argparse, hashlib, os, stat, zipfile
from pathlib import Path
FIXED_TIME=(2026,1,1,0,0,0)
EXCLUDED={'.git','.pytest_cache','__pycache__','.mypy_cache','.ruff_cache','build','dist'}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path(a.root).resolve(); out=Path(a.out).resolve()
    files=[]
    for p in sorted(root.rglob('*'), key=lambda x:x.as_posix()):
        rel=p.relative_to(root)
        if any(part in EXCLUDED for part in rel.parts): continue
        if p.is_symlink(): raise SystemExit(f'forbidden symlink: {rel}')
        if p.is_file(): files.append((p,rel))
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+'.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        prefix=root.name+'/'
        for p,rel in files:
            info=zipfile.ZipInfo(prefix+rel.as_posix(),FIXED_TIME)
            mode=stat.S_IMODE(p.stat().st_mode)
            info.create_system=3; info.external_attr=(stat.S_IFREG|mode)<<16; info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    os.replace(tmp,out)
    h=hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'{out}\nsha256:{h}\nbytes:{out.stat().st_size}')
    return 0
if __name__=='__main__': raise SystemExit(main())
