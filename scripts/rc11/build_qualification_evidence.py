#!/usr/bin/env python3
"""Build a deterministic, self-contained RC11 Mac qualification evidence ZIP."""
from __future__ import annotations
import argparse, hashlib, json, stat, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.qualification_bundle import verify_qualification_bundle

FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_file(path: Path) -> bytes:
    return canonical(json.loads(path.read_text(encoding="utf-8")))


def add_bytes(z: zipfile.ZipFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--capture-manifest", required=True)
    ap.add_argument("--observations", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bundle_path = Path(a.bundle)
    probe_path = Path(a.probe)
    capture_path = Path(a.capture_manifest)
    observations_path = Path(a.observations)
    kernel_path = Path(a.kernel)
    bundle = json.loads(bundle_path.read_text())
    if not verify_qualification_bundle(bundle):
        raise SystemExit("qualification bundle is invalid or did not pass")

    probe = canonical_file(probe_path)
    capture = canonical_file(capture_path)
    obs_text = observations_path.read_text().strip()
    if obs_text.startswith("["):
        observations_obj = json.loads(obs_text)
    else:
        observations_obj = [json.loads(line) for line in obs_text.splitlines() if line.strip()]
    observations = canonical(observations_obj)
    kernel = kernel_path.read_bytes()

    checks = {
        "hardware_probe_digest": sha(probe),
        "capture_manifest_digest": sha(capture),
        "oracle_observations_digest": sha(observations),
        "kernel_build_digest": sha(kernel),
    }
    for field, actual in checks.items():
        if bundle.get(field) != actual:
            raise SystemExit(f"{field} does not match qualification bundle")

    files = {
        "qualification_bundle.json": canonical(bundle) + b"\n",
        "hardware_probe.json": probe + b"\n",
        "capture_manifest.json": capture + b"\n",
        "observations.json": observations + b"\n",
        "artifacts/kernel.bin": kernel,
    }
    entries = [
        {"path": name, "bytes": len(data), "sha256": sha(data)}
        for name, data in sorted(files.items())
    ]
    evidence_body = {
        "schema_version": 1,
        "qualification_bundle_digest": bundle["bundle_digest"],
        "qualification_context_digest": bundle.get("qualification_context_digest"),
        "entries": entries,
    }
    evidence_body["evidence_tree_digest"] = sha(canonical(entries))
    manifest_bytes = canonical(evidence_body) + b"\n"
    files["EVIDENCE_MANIFEST.json"] = manifest_bytes

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(files.items()):
            add_bytes(z, name, data)
    print(json.dumps({
        "ok": True,
        "out": str(out),
        "bundle_digest": bundle["bundle_digest"],
        "evidence_tree_digest": evidence_body["evidence_tree_digest"],
        "zip_sha256": sha(out.read_bytes()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
