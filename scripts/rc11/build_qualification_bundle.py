#!/usr/bin/env python3
"""Build a deterministic fail-closed Apple-Silicon qualification bundle."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
from kvcontinual.execution.qualification_bundle import build_qualification_bundle, sha256_file
from kvcontinual.execution.qualification_harness import MacQualificationThresholds, QualificationObservation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--capture-manifest", required=True)
    ap.add_argument("--observations", required=True, help="JSON array or JSONL observations")
    ap.add_argument("--model-weights-digest", required=True)
    ap.add_argument("--tokenizer-digest", required=True)
    ap.add_argument("--execution-identity-digest", required=True,
                    help="sha256 digest of the exact ModelIdentity+ExecutionIdentity used for the run")
    ap.add_argument("--kernel", required=True, help="compiled kernel/library/binary whose bytes are qualified")
    ap.add_argument("--runtime-build-digest", default=None,
                    help="sha256 digest of the exact runtime build; defaults to release_tree_sha256 from --release-manifest")
    ap.add_argument("--release-manifest", default=str(ROOT / "SOURCE_MANIFEST.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-logit-kl", type=float, default=0.02)
    ap.add_argument("--min-top1-agreement", type=float, default=0.995)
    ap.add_argument("--max-failure-rate", type=float, default=0.01)
    ap.add_argument("--min-cases", type=int, default=100)
    args = ap.parse_args()

    probe = json.loads(Path(args.probe).read_text())
    capture = json.loads(Path(args.capture_manifest).read_text())
    text = Path(args.observations).read_text().strip()
    if text.startswith("["):
        raw_obs = json.loads(text)
    else:
        raw_obs = [json.loads(line) for line in text.splitlines() if line.strip()]
    observations = [QualificationObservation(**x) for x in raw_obs]
    thresholds = MacQualificationThresholds(
        max_logit_kl=args.max_logit_kl,
        min_top1_agreement=args.min_top1_agreement,
        max_failure_rate=args.max_failure_rate,
        min_cases=args.min_cases,
    )
    runtime_build_digest = args.runtime_build_digest
    if runtime_build_digest is None:
        release_manifest = json.loads(Path(args.release_manifest).read_text())
        runtime_build_digest = release_manifest.get("release_tree_sha256")
        if not isinstance(runtime_build_digest, str):
            raise SystemExit("release manifest does not contain release_tree_sha256")
    bundle = build_qualification_bundle(
        hardware_probe=probe,
        capture_manifest=capture,
        observations=observations,
        model_weights_digest=args.model_weights_digest,
        tokenizer_digest=args.tokenizer_digest,
        kernel_build_digest=sha256_file(args.kernel),
        execution_identity_digest=args.execution_identity_digest,
        runtime_build_digest=runtime_build_digest,
        thresholds=thresholds,
        require_apple_silicon=True,
    )
    payload = asdict(bundle)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"ok": payload["summary"]["passed"], "bundle_digest": payload["bundle_digest"], "out": args.out}, indent=2))
    return 0 if payload["summary"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
