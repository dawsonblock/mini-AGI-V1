#!/usr/bin/env python3
"""Runtime admission CLI (Phase 3 / v16.4.0).

Deployment entry point: the serving runtime independently admits the
exact artifact before loading it. Verifies the full authorization chain
(plan -> qualification -> promotion decision -> runtime manifest ->
adapter bytes) and writes a signed ActivationReceipt.

    python scripts/admit_runtime.py \
        --storage-root STORAGE --campaign-id CID --seed seed-0 \
        --adapter-dir STORAGE/adapters/CID/L6/seed-0 \
        --decision STORAGE/evidence/CID/RUNTIME_MANIFEST.json \
        --runtime-model-digest sha256:... --runtime-tokenizer-digest sha256:... \
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
    AdmissionRefused, RuntimeAdmissionController, write_activation_receipt)
from minagi.v161.runtime_closure3 import sha256_path  # noqa: E402


def _fail(msg: str) -> int:
    print(f"[ADMISSION REFUSED] {msg}", file=sys.stderr)
    return 2


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
    ap.add_argument("--runtime-model-digest", default=None)
    ap.add_argument("--runtime-tokenizer-digest", default=None)
    ap.add_argument("--runtime-model-path", default=None,
                    help="compute the model digest from this path instead")
    ap.add_argument("--runtime-tokenizer-path", default=None,
                    help="compute the tokenizer digest from this path")
    ap.add_argument("--revoked-decisions", default=None,
                    help="JSON file listing revoked decision digests")
    ap.add_argument("--runtime-key", default=None,
                    help="runtime role private key (default: "
                         "<storage>/.keys/runtime.pem); omit to write an "
                         "unsigned receipt")
    ap.add_argument("--no-sign", action="store_true",
                    help="write the receipt without a signature")
    args = ap.parse_args(argv)

    storage = Path(args.storage_root).resolve()
    trust_path = storage / "trust_root.json"
    if not trust_path.is_file():
        return _fail("trust_root.json missing")
    try:
        registry = AuthorityRegistry.load(trust_path)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"trust root load: {exc}")

    decision_path = Path(args.decision)
    if not decision_path.is_file():
        return _fail(f"decision document missing: {decision_path}")
    decision_doc = json.loads(decision_path.read_text())
    manifest = (decision_doc.get("runtime_manifests") or {}).get(args.seed)
    if manifest is None:
        return _fail(f"decision carries no runtime manifest for "
                     f"{args.seed!r}")

    cdir = storage / "evidence" / args.campaign_id
    qual_path = cdir / "QUALIFICATION_RECORD.json"
    plan_path = cdir / "CAMPAIGN_PLAN.json"
    for p in (qual_path, plan_path):
        if not p.is_file():
            return _fail(f"incomplete chain: {p.name} missing")
    qualification_doc = json.loads(qual_path.read_text())
    plan_doc = json.loads(plan_path.read_text())

    if args.runtime_model_path:
        model_digest = sha256_path(args.runtime_model_path)
    elif args.runtime_model_digest:
        model_digest = args.runtime_model_digest
    else:
        return _fail("configuration: --runtime-model-digest or "
                     "--runtime-model-path required")
    if args.runtime_tokenizer_path:
        tokenizer_digest = sha256_path(args.runtime_tokenizer_path)
    elif args.runtime_tokenizer_digest:
        tokenizer_digest = args.runtime_tokenizer_digest
    else:
        return _fail("configuration: --runtime-tokenizer-digest or "
                     "--runtime-tokenizer-path required")

    revoked = ()
    if args.revoked_decisions:
        try:
            revoked = json.loads(Path(args.revoked_decisions).read_text())
        except Exception as exc:  # noqa: BLE001
            return _fail(f"revocation list parse: {exc}")

    controller = RuntimeAdmissionController(registry,
                                            revoked_decision_digests=revoked)
    try:
        receipt = controller.admit(
            decision_doc=decision_doc, qualification_doc=qualification_doc,
            plan_doc=plan_doc, runtime_manifest=manifest,
            adapter_dir=args.adapter_dir, seed=args.seed,
            runtime_model_digest=model_digest,
            runtime_tokenizer_digest=tokenizer_digest,
            expected_backend=args.backend)
    except AdmissionRefused as exc:
        return _fail(str(exc))

    signer = None
    if not args.no_sign:
        key_path = Path(args.runtime_key) if args.runtime_key else \
            storage / ".keys" / "runtime.pem"
        if not key_path.is_file():
            return _fail(f"runtime key missing: {key_path} (use --no-sign "
                         "to emit an unsigned receipt)")
        signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())
        if not registry.is_authorized("runtime", signer.key_id):
            return _fail("runtime key not registered in trust root")

    doc = write_activation_receipt(receipt, args.out, signer=signer)
    print(json.dumps({
        "status": "ADMITTED",
        "decision_digest": receipt.decision_digest,
        "adapter_digest": receipt.adapter_digest,
        "backend": receipt.backend,
        "receipt_digest": doc["digest"],
        "signed": signer is not None,
        "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
