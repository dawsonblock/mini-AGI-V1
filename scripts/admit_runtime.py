#!/usr/bin/env python3
"""Runtime admission CLI (Phase 3 / v16.4.0; hardened v16.4.1).

Deployment entry point for ADMISSION EVIDENCE: the serving runtime
independently admits the exact artifact before loading it. Verifies the
full authorization chain (plan -> qualification -> promotion decision ->
runtime manifest -> adapter bytes) with strict schemas and physical
artifact measurement, then writes a runtime-signed ActivationReceipt.

v16.4.1 hardening:
  * model and tokenizer are MEASURED from --runtime-model-path /
    --runtime-tokenizer-path; a caller-supplied digest is only a
    cross-check and a disagreement is refused (the receipt is evidence,
    so it must bind measured bytes);
  * the receipt is always signed by the runtime role — an unsigned
    receipt is not authorization evidence and is no longer producible
    here;
  * --revocation-list accepts the versioned RevocationList document
    (with --max-revocation-age-days enforcing freshness).

This CLI performs admission only. STARTING a serving backend goes
through scripts/trusted_launch.py, which additionally stages the
immutable snapshot, loads the backend, and emits the production
receipt binding the bytes measured at load.

    python scripts/admit_runtime.py \
        --storage-root STORAGE --campaign-id CID --seed seed-0 \
        --adapter-dir STORAGE/adapters/CID/L6/seed-0 \
        --decision STORAGE/evidence/CID/RUNTIME_MANIFEST.json \
        --runtime-model-path MODEL_DIR --runtime-tokenizer-path TOK_DIR \
        --out STORAGE/evidence/CID/ACTIVATION_RECEIPT.json

Exit codes: 0 admitted, 2 refused (fail closed), 3 configuration error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import AuthorityRegistry  # noqa: E402
from minagi.v161.runtime_admission import (  # noqa: E402
    AdmissionRefused, RevocationList, RuntimeAdmissionController,
    write_activation_receipt)


def _fail(msg: str) -> int:
    print(f"[ADMISSION REFUSED] {msg}", file=sys.stderr)
    return 2


def _config_error(msg: str) -> int:
    print(f"[CONFIG ERROR] {msg}", file=sys.stderr)
    return 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--decision", required=True,
                    help="promotion decision document (promote.py --out)")
    ap.add_argument("--out", required=True, help="activation receipt path")
    ap.add_argument("--backend", default="hf-peft")
    ap.add_argument("--runtime-model-path", required=True,
                    help="model snapshot directory (physically measured)")
    ap.add_argument("--runtime-tokenizer-path", required=True,
                    help="tokenizer artifact directory (physically measured)")
    ap.add_argument("--runtime-model-digest", default=None,
                    help="optional cross-check against the measurement")
    ap.add_argument("--runtime-tokenizer-digest", default=None,
                    help="optional cross-check against the measurement")
    ap.add_argument("--revoked-decisions", default=None,
                    help="versioned revocation list document (or a bare "
                         "JSON list of revoked decision digests)")
    ap.add_argument("--max-revocation-age-days", type=float, default=None,
                    help="refuse a revocation list older than this")
    ap.add_argument("--runtime-key", default=None,
                    help="runtime role private key (default: "
                         "<storage>/.keys/runtime.pem)")
    args = ap.parse_args(argv)

    storage = Path(args.storage_root).resolve()
    trust_path = storage / "trust_root.json"
    if not trust_path.is_file():
        return _config_error("trust_root.json missing")
    try:
        registry = AuthorityRegistry.load(trust_path)
    except Exception as exc:  # noqa: BLE001
        return _config_error(f"trust root load: {exc}")

    decision_path = Path(args.decision)
    if not decision_path.is_file():
        return _config_error(f"decision document missing: {decision_path}")
    decision_doc = json.loads(decision_path.read_text())
    manifest = (decision_doc.get("runtime_manifests") or {}).get(args.seed)
    if manifest is None:
        return _config_error(f"decision carries no runtime manifest for "
                             f"{args.seed!r}")

    cdir = storage / "evidence" / args.campaign_id
    qual_path = cdir / "QUALIFICATION_RECORD.json"
    plan_path = cdir / "CAMPAIGN_PLAN.json"
    for p in (qual_path, plan_path):
        if not p.is_file():
            return _config_error(f"incomplete chain: {p.name} missing")
    qualification_doc = json.loads(qual_path.read_text())
    plan_doc = json.loads(plan_path.read_text())

    for label, path in (("model", args.runtime_model_path),
                        ("tokenizer", args.runtime_tokenizer_path)):
        if not Path(path).exists():
            return _config_error(f"{label} path missing: {path}")

    revocation_list = None
    max_age = None
    if args.max_revocation_age_days is not None:
        max_age = int(float(args.max_revocation_age_days) * 86400)
    if args.revoked_decisions:
        try:
            revocation_list = RevocationList.load(args.revoked_decisions)
        except Exception as exc:  # noqa: BLE001
            return _config_error(f"revocation list parse: {exc}")

    controller = RuntimeAdmissionController(
        registry, revocation_list=revocation_list,
        max_revocation_age_seconds=max_age)
    try:
        receipt = controller.admit(
            decision_doc=decision_doc, qualification_doc=qualification_doc,
            plan_doc=plan_doc, runtime_manifest=manifest,
            adapter_dir=args.adapter_dir, seed=args.seed,
            runtime_model_digest=args.runtime_model_digest,
            runtime_tokenizer_digest=args.runtime_tokenizer_digest,
            runtime_model_path=args.runtime_model_path,
            runtime_tokenizer_path=args.runtime_tokenizer_path,
            expected_backend=args.backend)
    except AdmissionRefused as exc:
        return _fail(str(exc))

    key_path = Path(args.runtime_key) if args.runtime_key else \
        storage / ".keys" / "runtime.pem"
    if not key_path.is_file():
        return _config_error(
            f"runtime key missing: {key_path} — an unsigned receipt is not "
            "authorization evidence")
    signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())
    if not registry.is_authorized("runtime", signer.key_id):
        return _config_error("runtime key not registered in trust root")

    try:
        doc = write_activation_receipt(receipt, args.out, signer=signer)
    except Exception as exc:  # noqa: BLE001 - no receipt, no admission
        return _config_error(f"receipt write failed: {exc}")
    print(json.dumps({
        "status": "ADMITTED",
        "decision_digest": receipt.decision_digest,
        "adapter_digest": receipt.adapter_digest,
        "backend": receipt.backend,
        "receipt_digest": doc["digest"],
        "production": False,
        "note": "admission evidence only; serving must go through "
                "scripts/trusted_launch.py, which emits the production "
                "receipt binding the bytes measured at load",
        "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
