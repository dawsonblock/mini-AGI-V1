"""v16.5/v16.4.1 PEFT serving path (REPAIR-025..029, Route A).

The first complete learning->serving loop trains, saves, reloads, and
serves the SAME adapter through the HF/PEFT stack. The runtime manifest
binds the exact base-model snapshot, tokenizer artifact set, and adapter
directory digests — promotion applies to those bytes, not a name.

Checks enforced here:

  * adapter_config.json exists and its LoRA config matches the signed
    protocol (rank/alpha/dropout/every declared target module)
  * the adapter directory closure matches the digest the campaign
    evidence committed (exact bytes, not just a path); symbolic links
    and special files anywhere in the tree are refused (v16.4.1)
  * manifest v2 additionally binds the complete authorized file listing
    (`adapter_files`): missing, unexpected, resized, or modified files
    are refused individually, not just via the aggregate digest
  * reload parity: a saved adapter re-loaded onto the base model must
    reproduce the training-instance outputs within the preregistered
    tolerance (verified by the caller's generation/forward probe)

Loading for serving goes through `PeftServingBackend.load()`, which
accepts ONLY an `MeasuredSnapshot` staged by the trusted launcher — a
raw directory path is not a loadable artifact. There is no public
function here that opens a mutable adapter directory for serving.

Native-runtime (qw3) parity is Route B and remains a separate gate —
PEFT qualification does NOT transitively qualify the native path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from egai.common.canonical import digest, sha256_bytes
from minagi.v161.artifact_closure import (ArtifactClosureError, TreeEntry,
                                          close_tree, expected_from_manifest,
                                          verify_entries)
from minagi.v161.immutable_snapshot import MeasuredSnapshot

RUNTIME_MANIFEST_SCHEMA = "mini-agi-v16.5-peft-runtime-manifest-v2"


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
    entries: tuple[TreeEntry, ...] = ()

    @property
    def digest(self) -> str:
        return digest(self)

    def manifest(self) -> list[dict]:
        """The complete authorized file listing (v2 manifests bind
        this; admission verifies it entry by entry)."""
        return [e.to_doc() for e in self.entries]


def adapter_closure(adapter_dir, *,
                    expected_files: Iterable[dict] | None = None
                    ) -> AdapterClosure:
    """Read + verify the PEFT adapter artifact on disk.

    v16.4.1: the tree is closed canonically — symlinks/special files are
    refused (never skipped), and when `expected_files` (the signed
    listing) is supplied the on-disk entries must match it exactly."""
    d = Path(adapter_dir)
    if not d.is_dir():
        raise AdapterClosureError(f"adapter dir missing: {d}")
    cfg_path = d / "adapter_config.json"
    if cfg_path.is_symlink():
        raise AdapterClosureError(
            "adapter_config.json is a symbolic link — refused")
    if not cfg_path.is_file():
        raise AdapterClosureError("adapter_config.json missing — "
                                  "not a PEFT adapter artifact")
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as exc:
        raise AdapterClosureError(
            f"adapter_config.json unreadable: {exc}") from exc
    if cfg.get("peft_type") != "LORA" or cfg.get("task_type") not in (
            "CAUSAL_LM", "SEQ_2_SEQ_LM", None):
        raise AdapterClosureError(
            f"unsupported peft artifact: {cfg.get('peft_type')}")
    targets = tuple(sorted(str(t) for t in cfg.get("target_modules") or ()))
    if not targets:
        raise AdapterClosureError("adapter declares no target modules")
    try:
        closure = close_tree(d)
    except ArtifactClosureError as exc:
        raise AdapterClosureError(f"adapter closure failed: {exc}") from exc
    weights = [e for e in closure.entries
               if Path(e.path).name.startswith("adapter_model")]
    if not weights:
        raise AdapterClosureError("adapter weight files missing")
    if expected_files is not None:
        try:
            verify_entries(closure.entries,
                           expected_from_manifest(expected_files),
                           what="adapter")
        except ArtifactClosureError as exc:
            raise AdapterClosureError(str(exc)) from exc
    return AdapterClosure(
        adapter_digest=closure.digest,
        rank=int(cfg.get("r", 0)),
        alpha=float(cfg.get("lora_alpha", 0)),
        dropout=float(cfg.get("lora_dropout", 0.0)),
        target_modules=targets,
        files=len(closure.entries),
        entries=closure.entries)


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
    protocol), the qualification record authorizing it, and — since
    v16.4.1 — the complete authorized adapter file listing. Promotion
    and serving must both consume this; never a mutable name like
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
            "adapter_files": closure.manifest(),
            "lora": {"rank": closure.rank, "alpha": closure.alpha,
                     "dropout": closure.dropout,
                     "target_modules": list(closure.target_modules)},
            "protocol_digest": (getattr(protocol, "digest", None)
                                if protocol is not None else None),
            "campaign_digest": campaign_digest,
            "qualification_record_digest": qualification_record_digest,
            "serving_stack": "hf-peft",
            "note": "exact-artifact binding; the trusted runtime launcher "
                    "must re-verify adapter_digest and adapter_files "
                    "against a staged immutable snapshot before serving"}
    body["digest"] = digest(body)
    return body


class PeftServingBackend:
    """The only supported PEFT serving entry point.

    `load()` accepts a `MeasuredSnapshot` (staged, verified, frozen by
    the supervisor's launch path) and refuses anything else — including
    a raw path, a str, or a mutable directory. The model, tokenizer,
    and adapter are opened exclusively from the snapshot root; the
    tokenizer is loaded from the verified model artifact, where the
    signed plan binds it (there is no separately substitutable
    tokenizer path in the served set).

    v16.4.2: loading and probing are distinct supervised steps — a
    loaded model is PREPARED (not serving); `health_probe` moves it to
    READY; `unload` releases a handle on demand so a failed activation
    cannot leave a resident model behind.

    v16.4.4 (WP-E/SEC-306): `infer` enforces an
    `InferenceBudgetPolicyV1` — the prompt is tokenized and its ACTUAL
    token count checked, requested ``max_new_tokens`` is clamped to the
    authorized bound, generation runs under a cooperative deadline and
    a route-level cancellation event (a network timeout is never
    mistaken for cancellation of GPU computation)."""

    backend_id = "hf-peft"

    def __init__(self, *, budget=None):
        from minagi.runtime.inference_policy import \
            InferenceBudgetPolicyV1
        self.budget = budget or InferenceBudgetPolicyV1()

    def load(self, snapshot: MeasuredSnapshot):
        if not isinstance(snapshot, MeasuredSnapshot):
            raise PermissionError(
                "PeftServingBackend.load() requires a MeasuredSnapshot "
                "staged by the supervised launch path — raw paths are "
                "not loadable artifacts")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot.path("model")), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot.path("model")), local_files_only=True)
        served = PeftModel.from_pretrained(
            model, str(snapshot.path("adapter")))
        served.eval()
        return {"model": served, "tokenizer": tokenizer,
                "snapshot_manifest_digest": snapshot.manifest_digest}

    def health_probe(self, handle) -> None:
        """The PREPARED -> READY gate: the loaded model must actually
        answer an inference probe before it may be committed."""
        import torch
        model, tokenizer = handle["model"], handle["tokenizer"]
        with torch.inference_mode():
            probe = tokenizer("runtime admission probe",
                              return_tensors="pt")
            out = model(**probe)
        if out.logits is None:
            raise RuntimeError("health probe produced no logits")

    def infer(self, handle, request) -> dict:
        """Bounded routed inference (WP8/OPS-003 + v16.4.4 WP-E): the
        serving router dispatches requests only to the live handle of
        the committed, healthy activation — the backend is never handed
        a raw path — and every request executes inside the authorized
        `InferenceBudgetPolicyV1` envelope."""
        import contextlib
        try:
            import torch
            inference_ctx = torch.inference_mode
        except ImportError:  # tests drive mock models without torch
            inference_ctx = contextlib.nullcontext
        model, tokenizer = handle["model"], handle["tokenizer"]
        req = dict(request or {})
        prompt = str(req.get("prompt", ""))
        request_id = str(req.get("request_id", ""))
        cancel_event = req.get("_cancel_event")
        with inference_ctx():
            ids = tokenizer(prompt, return_tensors="pt")
            prompt_tokens = self._token_count(ids)
            self.budget.check_prompt_tokens(prompt_tokens)
            requested = int(req.get("max_new_tokens", 8))
            applied = self.budget.clamp_max_new_tokens(requested)
            gen_kwargs = {"max_new_tokens": applied}
            deadline_applied = False
            if self.budget.execution_deadline_seconds:
                gen_kwargs["max_time"] = float(
                    self.budget.execution_deadline_seconds)
                deadline_applied = True
            stopping = self._stopping_criteria(cancel_event)
            if stopping:
                gen_kwargs["stopping_criteria"] = stopping
            try:
                out = model.generate(**ids, **gen_kwargs)
            except TypeError:
                # older generate() without max_time/stopping_criteria —
                # the token bound still holds; record the gap
                gen_kwargs.pop("max_time", None)
                gen_kwargs.pop("stopping_criteria", None)
                deadline_applied = False
                out = model.generate(**ids, **gen_kwargs)
        return {"completion": tokenizer.decode(
                    out[0], skip_special_tokens=True),
                "manifest_digest": handle.get(
                    "snapshot_manifest_digest", ""),
                "metrics": {
                    "request_id": request_id,
                    "prompt_tokens": int(prompt_tokens),
                    "max_new_tokens_requested": int(requested),
                    "max_new_tokens_applied": int(applied),
                    "deadline_seconds": float(
                        self.budget.execution_deadline_seconds),
                    "deadline_enforced": bool(deadline_applied),
                    "cancelled": bool(
                        cancel_event is not None and
                        cancel_event.is_set())}}

    @staticmethod
    def _token_count(ids) -> int:
        tensor = ids.get("input_ids") if hasattr(ids, "get") else None
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[-1])
        try:
            return len(ids["input_ids"][0])
        except Exception:  # noqa: BLE001
            return len(ids["input_ids"])

    @staticmethod
    def _stopping_criteria(cancel_event):
        """A cooperative StoppingCriteria that halts generation when
        the route's cancellation event is set — without corrupting the
        backend by killing the process."""
        if cancel_event is None:
            return None
        try:
            from transformers import StoppingCriteria, \
                StoppingCriteriaList
        except ImportError:
            return None

        class _RouteCancellation(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(cancel_event.is_set())

        return StoppingCriteriaList([_RouteCancellation()])

    def cancel(self, handle) -> None:
        """Cooperative cancellation hook used by the router's
        `cancel_requests`: signals in-flight generation to stop at the
        next produced token. GPU work is not force-killed — the flag is
        checked cooperatively."""
        event = handle.get("_cancel_event") if hasattr(
            handle, "get") else None
        if event is not None:
            event.set()

    def unload(self, handle) -> None:
        """Release a loaded model. Idempotent — a supervisor may call
        this during abort and again during recovery."""
        import gc
        handle["model"] = None
        handle["tokenizer"] = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache hygiene is best effort
            pass


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
