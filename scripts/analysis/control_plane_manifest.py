#!/usr/bin/env python3
"""Campaign 3B.3 Priority 1 — sign the control plane so it is a fact.

The control-plane map answers "which authority is authoritative" by
reachability, which means the answer is only available to someone who
runs the tool or reads the report. That is not good enough for the
priority, which asks for ONE authoritative control plane: the answer
should be a signed artifact that a reviewer can check without trusting
either the report or the tool.

This emits CONTROL_PLANE_MANIFEST.json — the LIVE authority set, the
declared superseded/orphan implementations, and the deployment entry
point — and signs it. A change that adds a competing authority, removes
the live one, or moves the deployment path changes the digest, so the
claim cannot drift silently.

Usage:
  python3 scripts/analysis/control_plane_manifest.py \
      --output CONTROL_PLANE_MANIFEST.json \
      --key ~/.config/miniagi/release-ed25519.pem
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

#: The single authority. Declared here, asserted by test, and signed.
LIVE_AUTHORITY = "minagi.v161.authority"

#: The live control plane, module by module. Each entry states its role so
#: a reviewer can see WHY it is in the set, not just that it is.
LIVE_CONTROL_PLANE = {
    "minagi.v161.authority": "the single authority: registry, ledger, roles",
    "minagi.v161.runtime_admission": "admission of artifacts at load time",
    "minagi.v161.trusted_launcher": "launch authorization and activation",
    "minagi.v161.evidence_receipt_v3": "evidence binding (receipts, bundles)",
    "minagi.v161.strict_schema": "fail-closed schema validation",
    "minagi.v161.peft_serving": "adapter closure and serving identity",
    "minagi.security.admission_grants": "signed admission grants",
    "minagi.security.signed_revocations": "signed revocation snapshots",
    "minagi.runtime.supervisor": "transactional activation lifecycle",
}

#: Implementations retained for provenance that no live path reaches, with
#: the reason they are retained and what supersedes each.
SUPERSEDED = {
    "dream_rsi_governed.promotion": "competing PromotionAuthority class",
    "dream_rsi_governed.qualification": "competing qualification engine",
    "dream_rsi_governed.activation": "competing activation step",
    "kvcontinual.execution.durability.promotion_authority":
        "competing promotion/rollback authorization",
    "minagi.rc14.qualification": "competing qualification engine (RC14)",
    "minagi.egai.authority": "competing authority (donor tree)",
    "minagi.v16.control_plane": "superseded by minagi.v161.converged_evidence",
}

#: The production runtime is OPERATOR-LAUNCHED, not import-reachable. This
#: is the boundary a static analysis cannot close, so it is declared here
#: explicitly rather than left to inference.
DEPLOYMENT_ENTRY_POINT = {
    "module": "minagi.runtime.service",
    "invocation": "python3 -m minagi.runtime.service",
    "launches": ("the isolation chain: nsjail_executor, "
                 "worker_launch_broker, isolation_preflight, "
                 "worker_executor, worker_isolation"),
    "status": ("DECLARED, NOT VERIFIED: nothing in this repository "
               "establishes that production runs this entry point rather "
               "than the kvcontinual gateway or the native qw3 binary. "
               "Pinning it is the remaining Priority 1 action."),
}


def _canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--key",
                    default=str(Path.home() /
                                ".config/miniagi/release-ed25519.pem"))
    ap.add_argument("--unsigned", action="store_true",
                    help="emit the manifest without signing it")
    args = ap.parse_args()

    # the declared live set must match what the map actually finds
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpm", ROOT / "scripts" / "analysis" / "control_plane_map.py")
    cpm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpm)

    index: dict = {}
    for p in (ROOT / "src-python").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        n = cpm._module_name(p)
        if n:
            index[n] = cpm._imports(p)
    for base in ("scripts", "tests-python"):
        for p in (ROOT / base).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            n = cpm._module_name(p)
            if n:
                index[n] = cpm._imports(p)
    seeds = [cpm._module_name(p) for pat in cpm.LIVE_ENTRY_GLOBS
             for p in ROOT.glob(pat)]
    live = cpm._reachable([s for s in seeds if s], index)

    problems = []
    if LIVE_AUTHORITY not in live:
        problems.append(f"{LIVE_AUTHORITY} is not reachable from any live path")
    for mod in LIVE_CONTROL_PLANE:
        if mod not in live:
            problems.append(f"declared live module is not reachable: {mod}")
    for mod in SUPERSEDED:
        if mod in live:
            problems.append(
                f"{mod} is declared superseded but a live path reaches it")
    if problems:
        for p in problems:
            print(f"REFUSED: {p}", file=sys.stderr)
        return 2

    doc = {
        "schema": "mini-agi-campaign3b3-control-plane-manifest-v1",
        "live_authority": LIVE_AUTHORITY,
        "live_control_plane": LIVE_CONTROL_PLANE,
        "n_live": len(LIVE_CONTROL_PLANE),
        "superseded": SUPERSEDED,
        "n_superseded": len(SUPERSEDED),
        "deployment_entry_point": DEPLOYMENT_ENTRY_POINT,
        "live_entry_points": sorted(set(s for s in seeds if s)),
        "statement": (
            "One authority governs promotion: minagi.v161.authority. Every "
            "other promotion/qualification implementation in this tree is "
            "retained for provenance and no live path reaches it. A module "
            "named PromotionAuthority exists and is NOT that authority."),
        "limits": (
            "Reachability is proven by static import analysis from the "
            "repository's own entry points. It does not prove what a "
            "deployment runs; the operator-launched service is declared "
            "above precisely because static analysis cannot decide it."),
    }
    body = _canonical(doc)
    doc["manifest_sha256"] = hashlib.sha256(body).hexdigest()

    if args.unsigned:
        doc["signing"] = "UNSIGNED (development)"
    else:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey)
        key_path = Path(args.key)
        if not key_path.is_file():
            print(f"REFUSED: signing key not found at {key_path}; pass "
                  "--unsigned for a development artifact", file=sys.stderr)
            return 2
        priv = serialization.load_pem_private_key(key_path.read_bytes(),
                                                  password=None)
        assert isinstance(priv, Ed25519PrivateKey)
        sig = priv.sign(body)
        pub = priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo)
        doc["signing"] = "Ed25519"
        doc["signature_b64"] = __import__("base64").b64encode(
            sig).decode()
        doc["public_key_pem_sha256"] = hashlib.sha256(pub).hexdigest()
        (Path(args.output).parent / "CONTROL_PLANE_PUBLIC_KEY.pem"
         ).write_bytes(pub)

    Path(args.output).write_text(json.dumps(doc, indent=2, sort_keys=True)
                                 + "\n")
    print(f"live authority      : {LIVE_AUTHORITY}")
    print(f"live control plane  : {len(LIVE_CONTROL_PLANE)} modules, "
          f"all reachable")
    print(f"superseded declared : {len(SUPERSEDED)} modules, "
          f"none reachable")
    print(f"manifest_sha256     : {doc['manifest_sha256']}")
    print(f"signing             : {doc['signing']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
