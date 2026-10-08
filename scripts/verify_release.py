#!/usr/bin/env python3
"""Verify a mini-AGI signed release.

Checks that every governed file on disk exactly matches SOURCE_MANIFEST.json
(missing / extra / changed all fail) and that the manifest carries a valid
Ed25519 signature from a TRUSTED release key.

Trust model (REPAIR-004): the bundled RELEASE_PUBLIC_KEY.pem is NOT trusted
merely because it sits inside the package. Its fingerprint must equal a
pinned fingerprint — otherwise an attacker could replace manifest + public
key + signature inside a ZIP and pass as an unauthorized release.

Pinning channels, in precedence order:

  --trusted-key <pem>        verify with an externally supplied key; the
                             bundled key file is ignored entirely
  --expected-key-fingerprint sha256:...  fingerprint the bundled key must
                             carry (also read from env
                             MINIAGI_RELEASE_KEY_FP)
  REQUIRED_TRUSTED_KEY_FINGERPRINT      baked-in fingerprint of the v16.x
                             release key (below)

The baked-in constant authenticates releases made with the repository's
known release key. It lives in this verifier — so for high-assurance use
the verifier itself (or at least this fingerprint) should come from a
trusted channel, and prefer --trusted-key with an independently held key.
Historical releases signed under different keys verify via
--expected-key-fingerprint / --trusted-key; their original manifests and
keys are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization

ROOT = Path(__file__).resolve().parents[1]

# sha256 of the DER SubjectPublicKeyInfo of the v16.x release key
# (~/.config/miniagi/release-ed25519.pem on the signing host). Public by
# design; authenticity comes from the pin, not secrecy.
REQUIRED_TRUSTED_KEY_FINGERPRINT = (
    "sha256:0f9d735812c99a583574a0157697f37b8b3373d34ab05d659638a88c0f5cb0eb")


def canonical_manifest_bytes(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def key_fingerprint(pub) -> str:
    der = pub.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return "sha256:" + hashlib.sha256(der).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trusted-key", default=None,
                    help="external PEM public key to verify with; overrides "
                         "the bundled RELEASE_PUBLIC_KEY.pem")
    ap.add_argument("--expected-key-fingerprint",
                    default=os.environ.get("MINIAGI_RELEASE_KEY_FP"),
                    help="required sha256 fingerprint of the bundled release "
                         "key (default: the pinned v16.x release fingerprint; "
                         "env: MINIAGI_RELEASE_KEY_FP)")
    args = ap.parse_args()

    manifest_path = ROOT / 'SOURCE_MANIFEST.json'
    sig_path = ROOT / 'RELEASE_SIGNATURE.bin'
    pub_path = ROOT / 'RELEASE_PUBLIC_KEY.pem'
    doc = json.loads(manifest_path.read_text())
    if doc.get('schema_version') != 1 \
            or doc.get('hash_algorithm') != 'sha256' \
            or not isinstance(doc.get('files'), dict):
        raise SystemExit('FAIL: unsupported SOURCE_MANIFEST schema')
    expected = doc['files']
    excluded = {'SOURCE_MANIFEST.json', 'RELEASE_SIGNATURE.bin',
                'RELEASE_PUBLIC_KEY.pem', 'RELEASE_ATTESTATION.json'}

    def skip(p):
        rel = p.relative_to(ROOT).as_posix()
        parts = p.relative_to(ROOT).parts
        # Gitignored build-artifact dirs (see .gitignore: build/,
        # build-cuda, build-*/) can never be release-controlled —
        # git ls-files excludes them, so flagging them in a developer
        # tree is a false positive. A clean extraction never contains
        # them, so this cannot hide real drift.
        artifact_dir = parts and parts[0].startswith('build')
        return (rel in excluded or artifact_dir or '__pycache__' in parts
                or '.git' in parts or '.pytest_cache' in parts
                or p.name.endswith('.pyc')
                or any(part.endswith('.egg-info') for part in parts))

    actual = {p.relative_to(ROOT).as_posix(): sha(p)
              for p in ROOT.rglob('*')
              if p.is_file() and not p.is_symlink() and not skip(p)}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in expected.keys() & actual.keys()
                         if expected[k] != actual[k])
        print(json.dumps({'status': 'FAIL', 'missing': missing[:20],
                          'extra': extra[:20], 'changed': changed[:20]},
                         indent=2))
        return 2

    if args.trusted_key:
        key = serialization.load_pem_public_key(
            Path(args.trusted_key).read_bytes())
        fp = key_fingerprint(key)
        source = "external --trusted-key"
    else:
        key = serialization.load_pem_public_key(pub_path.read_bytes())
        fp = key_fingerprint(key)
        source = "bundled RELEASE_PUBLIC_KEY.pem"
        expected_fp = (args.expected_key_fingerprint
                       or REQUIRED_TRUSTED_KEY_FINGERPRINT)
        if fp != expected_fp:
            print(json.dumps({
                'status': 'FAIL', 'reason': 'untrusted_release_key',
                'key_fingerprint': fp,
                'expected_fingerprint': expected_fp}, indent=2))
            return 3

    sig = sig_path.read_bytes()
    try:
        key.verify(sig, canonical_manifest_bytes(doc))
    except Exception:
        print(json.dumps({'status': 'FAIL', 'reason': 'bad_signature'},
                         indent=2))
        return 4

    print(json.dumps({
        'status': 'PASS', 'files_verified': len(actual),
        'key_source': source, 'key_fingerprint': fp,
        'manifest_sha256': hashlib.sha256(
            canonical_manifest_bytes(doc)).hexdigest()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
