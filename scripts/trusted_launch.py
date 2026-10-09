#!/usr/bin/env python3
"""Trusted runtime launcher CLI (v16.4.3) — supervised transactional
activation.

Every launch verifies the full signed chain plus an authenticated
RevocationSnapshotV2, issues a short-lived AdmissionGrant under the
`admission` role, stages an immutable measured snapshot under a
server-generated activation id, and hands the transactional lifecycle
to the ServingSupervisor over the SQLite authority store: durable grant
reservation → prepare → health probe → atomic commit+pointer → signed
completion. Any failed transition unloads the prepared handle and
restores traffic to the previous live runtime; a restart reconciles the
verified event log and never serves a model that is not resident.

    python scripts/trusted_launch.py \
        --storage-root STORAGE --campaign-id CID --seed seed-0 \
        --decision STORAGE/evidence/CID/RUNTIME_MANIFEST.json \
        --adapter-dir STORAGE/adapters/CID/L6/seed-0 \
        --model-path MODEL_DIR \
        --revocation-store STORAGE/revocations

The tokenizer identity is bound by the signed plan as the
tokenizer-named files inside the model snapshot, re-measured from the
staged model and served from that verified artifact — `--tokenizer-path`
is an optional cross-check root (the same tokenizer files elsewhere),
never the served copy.

Receipt paths and snapshot destinations are generated inside the
trusted side (`storage/receipts/<activation_id>.json`,
`storage/snapshots/<id>`); campaign/seed are metadata, never path
components. The backend factory (module:callable) must return an object
with a `backend_id` plus `load`/`health_probe`/`unload` lifecycle
methods; the default `hf-peft` factory is
`minagi.v161.peft_serving.PeftServingBackend`, which refuses anything
but a MeasuredSnapshot.

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
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.supervisor import ServingSupervisor  # noqa: E402
from minagi.security.signed_revocations import RevocationStore  # noqa: E402
from minagi.v161.authority import AuthorityRegistry  # noqa: E402
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
    ap.add_argument("--tokenizer-path", default=None,
                    help="optional tokenizer artifact root to cross-check "
                         "against the plan's binding (the served tokenizer "
                         "is the one inside the verified model snapshot)")
    ap.add_argument("--backend", default="hf-peft")
    ap.add_argument("--backend-factory", default=None,
                    help="module:callable returning a serving backend")
    ap.add_argument("--revocation-store", required=True,
                    help="directory of signed revocation snapshots "
                         "(RevocationStore); unsigned lists are refused")
    ap.add_argument("--max-revocation-age-days", type=float, default=7.0)
    ap.add_argument("--snapshot-root", default=None)
    ap.add_argument("--state-dir", default=None,
                    help="authority store directory "
                         "(default <storage>/state)")
    ap.add_argument("--policy-epoch", type=int, default=0)
    ap.add_argument("--receipt", default=None,
                    help="deprecated: receipt destinations are "
                         "service-owned; ignored")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--runtime-key", default=None)
    ap.add_argument("--admission-key", default=None)
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

    admission_key_path = Path(args.admission_key) if args.admission_key \
        else storage / ".keys" / "admission.pem"
    if not admission_key_path.is_file():
        return _config_error(
            f"admission key missing: {admission_key_path}")
    admission_signer = Ed25519Signer.from_private_bytes(
        admission_key_path.read_bytes())
    if not registry.is_authorized("admission", admission_signer.key_id):
        return _config_error(
            "admission key not registered in trust root")

    revocation_store = RevocationStore(args.revocation_store)

    try:
        backend = _resolve_factory(args.backend_factory, args.backend)
    except Exception as exc:  # noqa: BLE001
        return _config_error(f"backend factory: {exc}")

    store = AuthorityStore(
        (args.state_dir or (storage / "state")) / "authority.sqlite")
    supervisor = ServingSupervisor(
        store, runtime_signer=signer, registry=registry,
        min_policy_epoch=args.policy_epoch)
    supervisor.recover()

    launcher = TrustedRuntimeLauncher(
        registry, runtime_signer=signer,
        admission_signer=admission_signer,
        supervisor=supervisor,
        authority_store=store,
        revocation_store=revocation_store,
        snapshot_root=args.snapshot_root or (storage / "snapshots"),
        receipts_dir=storage / "receipts",
        policy_epoch=args.policy_epoch,
        max_revocation_age_seconds=int(
            float(args.max_revocation_age_days) * 86400),
        ledger_path=args.ledger or (storage / "AUTHORITY_LEDGER.jsonl"))

    request = LaunchRequest(
        campaign_id=args.campaign_id, seed=args.seed,
        decision_doc=decision_doc,
        qualification_doc=json.loads(qual_path.read_text()),
        plan_doc=json.loads(plan_path.read_text()),
        runtime_manifest=manifest, adapter_dir=args.adapter_dir,
        model_path=args.model_path, tokenizer_path=args.tokenizer_path or "",
        expected_backend=args.backend)

    try:
        result = launcher.launch(request, backend)
    except LaunchRefused as exc:
        return _refuse(str(exc))
    except Exception as exc:  # noqa: BLE001 - never report success on error
        print(f"[LAUNCH ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print(json.dumps({
        "status": "LAUNCHED",
        "backend": result.backend_id,
        "activation_id": result.activation_id,
        "grant_digest": result.grant_digest,
        "receipt_digest": result.receipt_doc["digest"],
        "receipt": str(result.receipt_path),
        "snapshot_root": str(result.snapshot.root),
        "loaded_artifact_digests": dict(
            result.receipt.loaded_artifact_digests),
        "activation_nonce": result.receipt.activation_nonce,
        "production": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
