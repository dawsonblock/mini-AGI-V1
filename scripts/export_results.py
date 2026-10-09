#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--campaign-dir',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    root=Path(a.campaign_dir).resolve(); out=Path(a.output).resolve(); files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest_bytes=json.dumps({'schema_version':1,'hash_algorithm':'sha256','files':manifest},sort_keys=True,separators=(',',':')).encode()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p,p.relative_to(root).as_posix())
        z.writestr('EXPORT_MANIFEST.json',manifest_bytes)
    print(json.dumps({'output':str(out),'files':len(files),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__': main()
