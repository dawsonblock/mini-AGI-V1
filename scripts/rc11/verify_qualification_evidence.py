#!/usr/bin/env python3
"""Verify a deterministic RC11 Mac qualification evidence ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.qualification_bundle import verify_qualification_bundle
from kvcontinual.execution.qualification_harness import MacQualificationThresholds, QualificationObservation, summarize_observations


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and all(part not in ("", ".", "..") for part in p.parts)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("evidence_zip"); a = ap.parse_args()
    path = Path(a.evidence_zip)
    try:
        with zipfile.ZipFile(path, "r") as z:
            infos = z.infolist()
            names = [i.filename for i in infos]
            if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
                raise ValueError("unsafe or duplicate ZIP member")
            for i in infos:
                mode = (i.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ValueError("symlink member is forbidden")
            required = {
                "EVIDENCE_MANIFEST.json", "qualification_bundle.json", "hardware_probe.json",
                "capture_manifest.json", "observations.json", "artifacts/kernel.bin",
            }
            if set(names) != required:
                raise ValueError("evidence ZIP member set is incomplete or contains extras")
            data = {name: z.read(name) for name in names}
        manifest = json.loads(data["EVIDENCE_MANIFEST.json"])
        bundle = json.loads(data["qualification_bundle.json"])
        if not verify_qualification_bundle(bundle):
            raise ValueError("qualification bundle is invalid")
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported evidence manifest schema")
        expected_entries = manifest.get("entries")
        if not isinstance(expected_entries, list):
            raise ValueError("evidence entries are missing")
        actual_entries = [
            {"path": name, "bytes": len(data[name]), "sha256": sha(data[name])}
            for name in sorted(required - {"EVIDENCE_MANIFEST.json"})
        ]
        if expected_entries != actual_entries:
            raise ValueError("evidence file digest manifest mismatch")
        if manifest.get("evidence_tree_digest") != sha(canonical(expected_entries)):
            raise ValueError("evidence tree digest mismatch")
        if manifest.get("qualification_bundle_digest") != bundle.get("bundle_digest"):
            raise ValueError("bundle digest mismatch")
        if manifest.get("qualification_context_digest") != bundle.get("qualification_context_digest"):
            raise ValueError("qualification context mismatch")
        cross = {
            "hardware_probe_digest": sha(canonical(json.loads(data["hardware_probe.json"]))),
            "capture_manifest_digest": sha(canonical(json.loads(data["capture_manifest.json"]))),
            "oracle_observations_digest": sha(canonical(json.loads(data["observations.json"]))),
            "kernel_build_digest": sha(data["artifacts/kernel.bin"]),
        }
        for field, actual in cross.items():
            if bundle.get(field) != actual:
                raise ValueError(f"{field} mismatch")
        obs_obj = json.loads(data["observations.json"])
        observations = [QualificationObservation(**x) for x in obs_obj]
        thresholds = MacQualificationThresholds(**bundle["thresholds"])
        recomputed_summary = summarize_observations(observations, thresholds)
        if bundle.get("summary") != {
            "case_count": recomputed_summary.case_count,
            "max_logit_kl": recomputed_summary.max_logit_kl,
            "mean_top1_agreement": recomputed_summary.mean_top1_agreement,
            "failure_rate": recomputed_summary.failure_rate,
            "passed": recomputed_summary.passed,
            "receipt_digest": recomputed_summary.receipt_digest,
        }:
            raise ValueError("qualification summary does not match packaged observations")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2)); return 1
    print(json.dumps({
        "ok": True,
        "bundle_digest": bundle["bundle_digest"],
        "evidence_tree_digest": manifest["evidence_tree_digest"],
        "zip_sha256": sha(path.read_bytes()),
    }, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
