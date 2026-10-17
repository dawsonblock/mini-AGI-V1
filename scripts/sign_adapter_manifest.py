#!/usr/bin/env python3
"""Sign a QualifiedAdapterManifest with an authority role key (V4-22).

A routing manifest is only a CLAIM until the qualification authority
signs it: ``VerifiedAdapterRouter`` configured with a verifier or an
``AuthorityRegistry`` refuses unsigned or wrongly-signed manifests
outright. This tool is the authority-side half of that loop — it runs
where the qualification role key lives (a different custody domain from
the training worker; see authority_bootstrap.py's deployment note) and
produces the signed manifest the router may admit.

Usage:
    python scripts/sign_adapter_manifest.py MANIFEST.json \
        --keys-dir <authority-keys> [--role qualification] [--out OUT.json]

MANIFEST.json may be a single manifest object or a list of them; the
output has the same shape, each entry carrying signer_key_id +
signature_b64 over the unsigned body. The tool refuses to re-sign an
already-signed manifest (a signature is an authorization event, not a
refreshable label — issue a new manifest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.v161.authority import provision_role  # noqa: E402
from minagi.v161.verified_routing import (  # noqa: E402
    QualifiedAdapterManifest)


def _sign_one(doc: dict, role: str, keys_dir: Path) -> dict:
    m = QualifiedAdapterManifest(**doc)
    if m.signed:
        raise SystemExit(
            f"manifest {m.adapter_id!r} is already signed — refusing to "
            "re-sign; authorization is an event, not a refreshable label")
    signer = provision_role(keys_dir, role)
    from dataclasses import asdict
    return asdict(m.sign(signer))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="manifest JSON file (object or list)")
    ap.add_argument("--keys-dir", required=True,
                    help="authority keys directory (provision_role)")
    ap.add_argument("--role", default="qualification",
                    help="authority role to sign under "
                         "(default: qualification — the router's "
                         "required_role)")
    ap.add_argument("--out", default=None,
                    help="output path (default: stdout)")
    args = ap.parse_args()

    doc = json.loads(Path(args.manifest).read_text())
    if isinstance(doc, list):
        signed = [_sign_one(d, args.role, Path(args.keys_dir))
                  for d in doc]
    else:
        signed = _sign_one(doc, args.role, Path(args.keys_dir))

    out = json.dumps(signed, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
