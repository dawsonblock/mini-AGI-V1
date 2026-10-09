"""Verify exact source membership, hashes, and the detached Ed25519 attestation."""
import argparse
import hashlib
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization

ENVELOPE={'SOURCE_MANIFEST.json','RELEASE_ATTESTATION.json','RELEASE_SIGNATURE.bin','RELEASE_PUBLIC_KEY.pem'}


def verify(root):
    root=Path(root)
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    expected=manifest['files']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in ENVELOPE}
    if actual!=set(expected):raise ValueError('source file membership mismatch')
    for name,row in expected.items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe source path')
        data=p.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('source digest mismatch: '+name)
    key=serialization.load_pem_public_key((root/'RELEASE_PUBLIC_KEY.pem').read_bytes())
    key.verify((root/'RELEASE_SIGNATURE.bin').read_bytes(),(root/'RELEASE_ATTESTATION.json').read_bytes())
    att=json.loads((root/'RELEASE_ATTESTATION.json').read_text())
    for field,name in [('manifest_sha256','SOURCE_MANIFEST.json'),('sbom_sha256','SBOM.cdx.json'),('validation_sha256','RELEASE_VALIDATION.json')]:
        if att[field]!=hashlib.sha256((root/name).read_bytes()).hexdigest():raise ValueError('attestation binding mismatch')
    pub=key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    if att['release_key_sha256']!=hashlib.sha256(pub).hexdigest():raise ValueError('release key identity mismatch')
    import tomllib
    metadata=tomllib.loads((root/'pyproject.toml').read_text())['project']
    if metadata['version']!=manifest['version'] or metadata['version']!=att['version'] or metadata['version']!=(root/'VERSION').read_text().strip():raise ValueError('release version mismatch')
    return {'verified_files':len(expected),'detached_signature':'verified','version':metadata['version']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parents[1]))
    print(json.dumps(verify(p.parse_args().root),sort_keys=True))
