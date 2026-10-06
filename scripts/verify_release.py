#!/usr/bin/env python3
from __future__ import annotations
import base64, hashlib, json, sys
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
ROOT=Path(__file__).resolve().parents[1]

def canonical_manifest_bytes(doc):
    return json.dumps(doc,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    manifest_path=ROOT/'SOURCE_MANIFEST.json'; sig_path=ROOT/'RELEASE_SIGNATURE.bin'; pub_path=ROOT/'RELEASE_PUBLIC_KEY.pem'
    doc=json.loads(manifest_path.read_text())
    if doc.get('schema_version')!=1 or doc.get('hash_algorithm')!='sha256' or not isinstance(doc.get('files'),dict):
        raise SystemExit('FAIL: unsupported SOURCE_MANIFEST schema')
    expected=doc['files']; excluded={'SOURCE_MANIFEST.json','RELEASE_SIGNATURE.bin','RELEASE_PUBLIC_KEY.pem','RELEASE_ATTESTATION.json'}
    def skip(p):
        rel=p.relative_to(ROOT).as_posix()
        parts=p.relative_to(ROOT).parts
        return (rel in excluded or '__pycache__' in parts or '.git' in parts
                or '.pytest_cache' in parts or p.name.endswith('.pyc')
                or any(part.endswith('.egg-info') for part in parts))
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink() and not skip(p)}
    if expected!=actual:
        missing=sorted(set(expected)-set(actual)); extra=sorted(set(actual)-set(expected)); changed=sorted(k for k in expected.keys()&actual.keys() if expected[k]!=actual[k])
        print(json.dumps({'status':'FAIL','missing':missing[:20],'extra':extra[:20],'changed':changed[:20]},indent=2)); return 2
    key=serialization.load_pem_public_key(pub_path.read_bytes()); sig=sig_path.read_bytes(); key.verify(sig,canonical_manifest_bytes(doc))
    print(json.dumps({'status':'PASS','files_verified':len(actual),'manifest_sha256':hashlib.sha256(canonical_manifest_bytes(doc)).hexdigest()},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
