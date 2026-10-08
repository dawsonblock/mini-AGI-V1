#!/usr/bin/env python3
"""Trusted runtime launcher CLI (v16.4.1) — the supported model-loading
path.

This is the enforcement point the v16.4.0 audit found missing: every
launch verifies the full signed chain, measures the model, tokenizer,
and adapter bytes physically, stages an immutable snapshot, loads the
selected serving backend ONLY from that snapshot, and emits a signed
production activation receipt (recorded in the authority ledger) after
the load succeeds.

    python scripts/trusted_launch.py \
        --storage-root STORAGE --campaign-id CID --seed seed-0 \
        --decision STORAGE/evidence/CID/RUNTIME_MANIFEST.json \
        --adapter-dir STORAGE/adapters/CID/L6/seed-0 \
        --model-path MODEL_DIR --tokenizer-path TOKENIZER_DIR \
        --revocation-list STORAGE/REVOCATIONS.json \
        --receipt STORAGE/evidence/CID/ACTIVATION_RECEIPT.json

The backend factory (module:callable) must return an object with a
`backend_id` and a `load(snapshot)` method; the default `hf-peft`
factory is `minagi.v161.peft_serving.PeftServingBackend`, which refuses
anything but an ApprovedSnapshot.

Exit codes: 0 launched, 2 refused (fail closed), 3 configuration error.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import AuthorityRegistry  # noqa: E402
from minagi.v161.runtime_admission import RevocationList  # noqa: E402
from minagi.v161.trusted_launcher import (  # noqa: E402
    LaunchRefused, LaunchRequest, TrustedRuntimeLauncher)


def _refuse(msg: str) -> int:
    print(f"[LAUNCH REFUSED] {msg}", file=sys.stderr)
    return 2


def _config_error(msg: str) -> int:
    print(f"[CONFIG ERROR] {msg}", file=sys.stderr)
    return 3


def _resolve_factory(spec: str | None, backend: str):
    if spec is None:
        if backend != "hf-peft":
            raise ValueError(
                f"no built-in factory for backend {backend!r} — pass "
                "--backend-factory module:callable")
        from minagi.v161.peft_serving import PeftServingBackend
        return PeftServingBackend()
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError("--backend-factory must be module:callable")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    return factory() if callable(factory) else factory


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--decision", required=True,
                    help="promotion decision document (promote.py --out)")
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--backend", default="hf-peft")
    ap.add_argument("--backend-factory", default=None,
                    help="module:callable returning a serving backend")
    ap.add_argument("--revocation-list", required=True)
    ap.add_argument("--max-revocation-age-days", type=float, default=7.0)
    ap.add_argument("--snapshot-root", default=None)
    ap.add_argument("--nonce-journal", default=None)
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--runtime-key", default=None)
    args = ap.parse_args(argv)

    storage = Path(args.storage_root).resolve()
    trust_path = storage / "trust_root.json"
    if not trust_path.is_file():
        return _config_error("trust_root.json missing")
    try:
        registry = AuthorityRegistry.load(trust_path)
    except Exception as exc:  # noqa: BLE001
        return _config_error(f"trust root load: {exc}")

    cdir = storage / "evidence" / args.campaign_id
    decision_path = Path(args.decision)
    qual_path = cdir / "QUALIFICATION_RECORD.json"
    plan_path = cdir / "CAMPAIGN_PLAN.json"
    for p in (decision_path, qual_path, plan_path):
        if not p.is_file():
            return _config_error(f"incomplete chain: {p} missing")
    decision_doc = json.loads(decision_path.read_text())
    manifest = (decision_doc.get("runtime_manifests") or {}).get(args.seed)
    if manifest is None:
        return _config_error(
            f"decision carries no runtime manifest for {args.seed!r}")

    key_path = Path(args.runtime_key) if args.runtime_key else \
        storage / ".keys" / "runtime.pem"
    if not key_path.is_file():
        return _config_error(f"runtime key missing: {key_path}")
    signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())
    if not registry.is_authorized("runtime", signer.key_id):
        return _config_error("runtime key not registered in trust root")

    try:
        revocation_list = RevocationList.load(args.revocation_list)
    except Exception as exc:  # noqa: BLE001
        return _config_error(f"revocation list load: {exc}")

    try:
        backend = _resolve_factory(args.backend_factory, args.backend)
    except Exception as exc:  # noqa: BLE001
        return _config_error(f"backend factory: {exc}")

    launcher = TrustedRuntimeLauncher(
        registry, revocation_list=revocation_list, runtime_signer=signer,
        snapshot_root=args.snapshot_root or (storage / "snapshots"),
        nonce_journal=args.nonce_journal
        or (storage / "activation_nonces.jsonl"),
        max_revocation_age_seconds=int(
            float(args.max_revocation_age_days) * 86400),
        ledger_path=args.ledger or (storage / "AUTHORITY_LEDGER.jsonl"))

    request = LaunchRequest(
        campaign_id=args.campaign_id, seed=args.seed,
        decision_doc=decision_doc,
        qualification_doc=json.loads(qual_path.read_text()),
        plan_doc=json.loads(plan_path.read_text()),
        runtime_manifest=manifest, adapter_dir=args.adapter_dir,
        model_path=args.model_path, tokenizer_path=args.tokenizer_path,
        expected_backend=args.backend)

    receipt_path = args.receipt or (
        cdir / "ACTIVATION_RECEIPT.json")
    try:
        result = launcher.launch(request, backend, receipt_path=receipt_path)
    except LaunchRefused as exc:
        return _refuse(str(exc))
    except Exception as exc:  # noqa: BLE001 - never report success on error
        print(f"[LAUNCH ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(json.dumps({
        "status": "LAUNCHED",
        "backend": result.backend_id,
        "receipt_digest": result.receipt_doc["digest"],
        "receipt": str(receipt_path),
        "snapshot_root": str(result.snapshot.root),
        "loaded_artifact_digests": dict(
            result.receipt.loaded_artifact_digests),
        "activation_nonce": result.receipt.activation_nonce,
        "production": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
