"""v16.5 PEFT serving path (REPAIR-025..029, Route A).

The first complete learning->serving loop trains, saves, reloads, and
serves the SAME adapter through the HF/PEFT stack. The runtime manifest
binds the exact base-model snapshot, tokenizer artifact set, and adapter
directory digests — promotion applies to those bytes, not a name.

Checks enforced here:

  * adapter_config.json exists and its LoRA config matches the signed
    protocol (rank/alpha/dropout/every declared target module)
  * the adapter directory digest matches the digest the campaign
    evidence committed (exact bytes, not just a path)
  * reload parity: a saved adapter re-loaded onto the base model must
    reproduce the training-instance outputs within the preregistered
    tolerance (verified by the caller's generation/forward probe)

Native-runtime (qw3) parity is Route B and remains a separate gate —
PEFT qualification does NOT transitively qualify the native path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from egai.common.canonical import digest, sha256_bytes
from minagi.v161.runtime_closure3 import sha256_path

RUNTIME_MANIFEST_SCHEMA = "mini-agi-v16.5-peft-runtime-manifest-v1"


class AdapterClosureError(ValueError):
    """Adapter artifact fails closure checks (never silently serve)."""


@dataclass(frozen=True)
class AdapterClosure:
    adapter_digest: str
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    files: int

    @property
    def digest(self) -> str:
        return digest(self)


def adapter_closure(adapter_dir) -> AdapterClosure:
    """Read + verify the PEFT adapter artifact on disk."""
    d = Path(adapter_dir)
    if not d.is_dir():
        raise AdapterClosureError(f"adapter dir missing: {d}")
    cfg_path = d / "adapter_config.json"
    if not cfg_path.is_file():
        raise AdapterClosureError("adapter_config.json missing — "
                                  "not a PEFT adapter artifact")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("peft_type") != "LORA" or cfg.get("task_type") not in (
            "CAUSAL_LM", "SEQ_2_SEQ_LM", None):
        raise AdapterClosureError(
            f"unsupported peft artifact: {cfg.get('peft_type')}")
    targets = tuple(sorted(str(t) for t in cfg.get("target_modules") or ()))
    if not targets:
        raise AdapterClosureError("adapter declares no target modules")
    weights = [p for p in d.iterdir()
               if p.name.startswith("adapter_model")]
    if not weights:
        raise AdapterClosureError("adapter weight files missing")
    return AdapterClosure(
        adapter_digest=sha256_path(d),
        rank=int(cfg.get("r", 0)),
        alpha=float(cfg.get("lora_alpha", 0)),
        dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=targets,
        files=sum(1 for p in d.rglob("*") if p.is_file()))


def assert_adapter_matches_protocol(adapter_dir,
                                    protocol) -> AdapterClosure:
    """Verify the on-disk adapter IS the protocol's declared footprint:
    rank/alpha/dropout equal and every declared target present."""
    c = adapter_closure(adapter_dir)
    if c.rank != int(protocol.lora_rank):
        raise AdapterClosureError(
            f"adapter rank {c.rank} != protocol {protocol.lora_rank}")
    if abs(c.alpha - float(protocol.lora_alpha)) > 1e-9:
        raise AdapterClosureError(
            f"adapter alpha {c.alpha} != protocol {protocol.lora_alpha}")
    if abs(c.dropout - float(protocol.lora_dropout)) > 1e-9:
        raise AdapterClosureError(
            f"adapter dropout {c.dropout} != protocol "
            f"{protocol.lora_dropout}")
    declared = set(c.target_modules)
    expected = set(str(t) for t in protocol.lora_target_modules)
    missing = expected - declared
    extra = declared - expected
    if missing or extra:
        raise AdapterClosureError(
            f"adapter targets {sorted(declared)} != protocol targets "
            f"{sorted(expected)} (missing={sorted(missing)}, "
            f"extra={sorted(extra)})")
    return c


def runtime_manifest(*, model_id: str, model_revision: str,
                     model_digest: str, tokenizer_digest: str,
                     adapter_dir, protocol=None,
                     expected_adapter_digest: str | None = None,
                     qualification_record_digest: str | None = None,
                     campaign_digest: str | None = None) -> dict:
    """Build the deployment-bound manifest for the PEFT serving path.

    Binds: exact model snapshot/config identity, tokenizer identity,
    the adapter bytes, the LoRA footprint actually on disk, the
    protocol digest (when the adapter was produced under a signed
    protocol), and the qualification record authorizing it. Promotion
    and serving must both consume this — never a mutable name like
    'latest'."""
    closure = adapter_closure(adapter_dir)
    if expected_adapter_digest is not None \
            and closure.adapter_digest != expected_adapter_digest:
        raise AdapterClosureError(
            f"adapter bytes digest {closure.adapter_digest} != "
            f"qualified digest {expected_adapter_digest}")
    if protocol is not None:
        assert_adapter_matches_protocol(adapter_dir, protocol)
    body = {"schema": RUNTIME_MANIFEST_SCHEMA,
            "model_id": str(model_id),
            "model_revision": str(model_revision),
            "model_digest": model_digest,
            "tokenizer_digest": tokenizer_digest,
            "adapter_digest": closure.adapter_digest,
            "lora": {"rank": closure.rank, "alpha": closure.alpha,
                     "dropout": closure.dropout,
                     "target_modules": list(closure.target_modules)},
            "protocol_digest": (getattr(protocol, "digest", None)
                                if protocol is not None else None),
            "campaign_digest": campaign_digest,
            "qualification_record_digest": qualification_record_digest,
            "serving_stack": "hf-peft",
            "note": "exact-artifact binding; runtime admission control "
                    "must re-verify adapter_digest before serving"}
    body["digest"] = digest(body)
    return body


def logits_digest(model, tokenizer, prompts: Iterable[str],
                  device=None) -> str:
    """Deterministic probe: digest over next-token logits for each
    prompt — used to prove a reloaded adapter reproduces the training
    instance's outputs bit-for-bit (fp32) / within tolerance."""
    import torch
    rows = []
    if device is None:
        device = next(model.parameters()).device
    with torch.inference_mode():
        for p in prompts:
            b = tokenizer(str(p), return_tensors="pt")
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b).logits[0, -1]
            rows.append({"prompt_sha256": sha256_bytes(str(p).encode()),
                         "logits_sha256": sha256_bytes(
                             logits.float().cpu().numpy().tobytes())})
    return digest(rows)
