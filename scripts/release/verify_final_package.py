#!/usr/bin/env python3
"""Build and verify the v16.4.5 release artifacts (SEC-403).

Two separately verifiable artifacts:

  mini-AGI-V1-v16.4.5-Security-and-Evidence-Closure.zip
      governed source = `git ls-files` intersected with disk (exactly the
      set a fresh clone receives), written deterministically and then
      verified on fresh extraction with scripts/verify_release.py's
      pinned-fingerprint signature check.

  mini-AGI-Campaign3A-Evidence.zip
      the historical colab-evidence tree — experiment receipts,
      predictions, result files, raw archives — plus its own
      EVIDENCE_MANIFEST.json, signature under the same pinned release
      key, and EVIDENCE_PROVENANCE.json recording the source archive it
      was split from. The historical Campaign 3A REFUSED result is
      preserved unmodified.

Verification is run against fresh extractions, never the working tree:
a clean extraction contains no stray developer files, so any mismatch
between manifest and disk is real drift.

Usage:
    python3 scripts/release/verify_final_package.py \
        --out-dir /path/to/release --key ~/.config/miniagi/release-ed25519.pem
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENVELOPE = {"SOURCE_MANIFEST.json", "RELEASE_SIGNATURE.bin",
            "RELEASE_PUBLIC_KEY.pem", "RELEASE_ATTESTATION.json"}
EVIDENCE_ENVELOPE = {"EVIDENCE_MANIFEST.json", "EVIDENCE_SIGNATURE.bin",
                     "RELEASE_PUBLIC_KEY.pem", "EVIDENCE_PROVENANCE.json"}
EVIDENCE_ROOT = "colab-evidence"
SOURCE_ZIP = "mini-AGI-V1-v16.4.5-Security-and-Evidence-Closure.zip"
EVIDENCE_ZIP = "mini-AGI-Campaign3A-Evidence.zip"
# sha256 of the v16.4.4 source archive this release repairs, recorded at
# freeze time for provenance (never re-signed or rewritten).
BASE_ARCHIVE_SHA256 = (
    "ad5993bd1e1e56f4f60c822cc95934b36710a8425d9978f905f095c101a0e7d5")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def _tracked_files(root: Path):
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True,
        capture_output=True).stdout.decode().split("\0")
    return sorted(root / t for t in tracked
                  if t and (root / t).is_file()
                  and not (root / t).is_symlink())


def _write_zip(zip_path: Path, root: Path, files, arc_root: str):
    """Deterministic zip: sorted paths, epoch timestamp, fixed modes."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for p in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
            rel = p.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{arc_root}/{rel}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if p.suffix == ".sh" else 0o644
            info.external_attr = (0o100000 | mode) << 16
            z.writestr(info, p.read_bytes())


def _extract(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return next(dest.iterdir())


def _verify_source(extracted: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_release.py"),
         "--root", str(extracted)],
        capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {"status": "FAIL", "reason": "verifier_crashed",
                  "stderr": proc.stderr[-2000:]}
    return report


def _load_release_pub(pub_pem: bytes):
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_public_key(pub_pem)


def _key_fp(pub) -> str:
    from cryptography.hazmat.primitives import serialization
    der = pub.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return "sha256:" + hashlib.sha256(der).hexdigest()


def _verify_evidence(extracted: Path, expected_fp: str) -> dict:
    doc = json.loads((extracted / "EVIDENCE_MANIFEST.json").read_text())
    expected = doc["files"]
    actual = {p.relative_to(extracted).as_posix(): _sha(p)
              for p in extracted.rglob("*")
              if p.is_file() and not p.is_symlink()
              and p.relative_to(extracted).as_posix()
              not in EVIDENCE_ENVELOPE}
    if expected != actual:
        return {"status": "FAIL",
                "missing": sorted(set(expected) - set(actual))[:20],
                "extra": sorted(set(actual) - set(expected))[:20],
                "changed": sorted(k for k in expected.keys() & actual.keys()
                                  if expected[k] != actual[k])[:20]}
    pub = _load_release_pub((extracted / "RELEASE_PUBLIC_KEY.pem").read_bytes())
    fp = _key_fp(pub)
    if fp != expected_fp:
        return {"status": "FAIL", "reason": "untrusted_release_key",
                "key_fingerprint": fp}
    try:
        pub.verify((extracted / "EVIDENCE_SIGNATURE.bin").read_bytes(),
                   _canonical(doc))
    except Exception:
        return {"status": "FAIL", "reason": "bad_signature"}
    return {"status": "PASS", "files_verified": len(actual),
            "key_fingerprint": fp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key",
                    default=str(Path.home() / ".config" / "miniagi"
                                / "release-ed25519.pem"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from cryptography.hazmat.primitives import serialization
    key_path = Path(args.key)
    if not key_path.is_file():
        raise SystemExit(f"release key not found at {key_path} — refusing "
                         "to package an unsigned release")
    priv = serialization.load_pem_private_key(key_path.read_bytes(),
                                              password=None)
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)

    report = {"artifacts": {}}

    # ---- source artifact -------------------------------------------------
    src_zip = out / SOURCE_ZIP
    _write_zip(src_zip, ROOT, _tracked_files(ROOT),
               SOURCE_ZIP.removesuffix(".zip"))
    with tempfile.TemporaryDirectory() as td:
        extracted = _extract(src_zip, Path(td))
        src_report = _verify_source(extracted)
    report["artifacts"][SOURCE_ZIP] = {
        "sha256": _sha(src_zip), "verification": src_report}

    # ---- evidence artifact ----------------------------------------------
    ev_root = ROOT / EVIDENCE_ROOT
    ev_files = sorted(p for p in ev_root.rglob("*")
                      if p.is_file() and not p.is_symlink())
    manifest = {"schema_version": 1, "hash_algorithm": "sha256",
                "release": "mini-AGI-Campaign3A-Evidence",
                "files": {f"{EVIDENCE_ROOT}/{p.relative_to(ev_root).as_posix()}":
                          _sha(p) for p in ev_files}}
    provenance = {
        "schema": "mini-agi-evidence-provenance-v1",
        "campaign": "3A",
        "result": "REFUSED (historical; preserved unmodified)",
        "split_from": {
            "archive": "mini-AGI-V1-v16.4.4-Verified-Runtime-Closure.zip",
            "sha256": BASE_ARCHIVE_SHA256},
        "separated_in": "v16.4.5 (SEC-403): experiment evidence ships as a "
                        "separately verified artifact, not inside the "
                        "governed source tree",
        "files": len(ev_files)}
    ev_zip = out / EVIDENCE_ZIP
    with zipfile.ZipFile(ev_zip, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for p in ev_files:
            rel = p.relative_to(ev_root).as_posix()
            info = zipfile.ZipInfo(
                f"{EVIDENCE_ZIP.removesuffix('.zip')}/"
                f"{EVIDENCE_ROOT}/{rel}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100000 | 0o644) << 16
            z.writestr(info, p.read_bytes())
        for name, payload in {
                "EVIDENCE_MANIFEST.json":
                    json.dumps(manifest, indent=1, sort_keys=True).encode()
                    + b"\n",
                "EVIDENCE_SIGNATURE.bin":
                    priv.sign(_canonical(manifest)),
                "RELEASE_PUBLIC_KEY.pem": pub_pem,
                "EVIDENCE_PROVENANCE.json":
                    json.dumps(provenance, indent=1, sort_keys=True).encode()
                    + b"\n"}.items():
            info = zipfile.ZipInfo(
                f"{EVIDENCE_ZIP.removesuffix('.zip')}/{name}",
                (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100000 | 0o644) << 16
            z.writestr(info, payload)

    with tempfile.TemporaryDirectory() as td:
        extracted = _extract(ev_zip, Path(td))
        ev_report = _verify_evidence(
            extracted,
            _key_fp(_load_release_pub(
                (ROOT / "RELEASE_PUBLIC_KEY.pem").read_bytes())))
    report["artifacts"][EVIDENCE_ZIP] = {
        "sha256": _sha(ev_zip), "verification": ev_report}

    ok = all(a["verification"].get("status") == "PASS"
             for a in report["artifacts"].values())
    report["status"] = "PASS" if ok else "FAIL"
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
