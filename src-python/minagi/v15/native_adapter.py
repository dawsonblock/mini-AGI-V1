from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import re
import struct
from typing import Mapping

import numpy as np

from egai.common.canonical import validate_digest
from minagi.integration.qw3_state import ServedArtifactManifest
from minagi.v15.adapter_learning import (
    AdapterArtifactManifest,
    AdapterSetBundle,
    AdapterTrainingPlan,
    QualifiedAdapterSetCandidate,
)

_HEX = set("0123456789abcdef")


def _hex64(value: str, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(obj: Mapping) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Minimal read-only safetensors loader for the native compiler.

    It supports the numeric dtypes normally emitted by LoRA trainers and does
    not execute any code from the artifact. The source adapter's byte digest is
    verified by the v15.2 artifact manifest before this parser is entered.
    """
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"invalid safetensors file: {path}")
    header_len = struct.unpack("<Q", data[:8])[0]
    if header_len <= 1 or 8 + header_len > len(data):
        raise ValueError(f"invalid safetensors header length: {path}")
    header = json.loads(data[8:8 + header_len])
    body = memoryview(data)[8 + header_len:]
    out: dict[str, np.ndarray] = {}
    dtype_map = {
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
        "F64": np.dtype("<f8"),
        "I32": np.dtype("<i4"),
        "I64": np.dtype("<i8"),
        "U8": np.dtype("u1"),
    }
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = str(meta["dtype"])
        shape = tuple(int(x) for x in meta["shape"])
        start, end = (int(x) for x in meta["data_offsets"])
        if start < 0 or end < start or end > len(body):
            raise ValueError(f"invalid tensor offsets for {name}")
        raw = body[start:end]
        if dtype == "BF16":
            vals = np.frombuffer(raw, dtype=np.dtype("<u2")).astype(np.uint32) << 16
            arr = vals.view(np.float32)
        elif dtype in dtype_map:
            arr = np.frombuffer(raw, dtype=dtype_map[dtype])
        else:
            raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
        expected = int(np.prod(shape, dtype=np.int64))
        if arr.size != expected:
            raise ValueError(f"tensor byte/shape mismatch for {name}")
        out[name] = np.asarray(arr.reshape(shape), dtype=np.float32)
    return out


def _lora_role(key: str) -> tuple[str, str] | None:
    lower = key.lower()
    suffixes = (
        (".lora_a.weight", "a"), (".lora_b.weight", "b"),
        (".lora_a", "a"), (".lora_b", "b"),
        (".lora_down.weight", "a"), (".lora_up.weight", "b"),
        (".lora_down", "a"), (".lora_up", "b"),
    )
    for suffix, role in suffixes:
        if lower.endswith(suffix):
            return key[: len(key) - len(suffix)], role
    return None


def _is_output_target(prefix: str) -> bool:
    lower = prefix.lower()
    return "lm_head" in lower or lower.endswith("output") or "output.weight" in lower


def native_adapter_supports_target(module_name: str) -> bool:
    """Config-level check: a PEFT LoRA trained on this module name must
    produce an artifact NativeAdapter1 can compile and serve. The
    qualified artifact and the served artifact must be the same object;
    unsupported targets are refused before training rather than
    discovered at promotion time."""
    return _is_output_target(str(module_name))


_NATIVE2_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _attention_target_kind(prefix: str) -> tuple[int, str] | None:
    """Match a PEFT module prefix to (layer_index, kind) for attention
    projections: '...layers.12.self_attn.q_proj' -> (12, 'q_proj')."""
    m = re.search(r"\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
                  prefix.lower())
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def native_adapter2_supports_target(module_name: str) -> bool:
    """NativeAdapter2 (v2 bundles): the v1 LM-head surface plus the four
    attention projections, identified by a layers.N.self_attn.*_proj
    module path. Anything else is refused."""
    name = str(module_name)
    if _is_output_target(name):
        return True
    return _attention_target_kind(name) is not None


def _normalize_a(arr: np.ndarray, rank: int, in_features: int) -> np.ndarray:
    if arr.shape == (rank, in_features):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape == (in_features, rank):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"LoRA A shape {arr.shape} does not match rank={rank}, in={in_features}")


def _normalize_b(arr: np.ndarray, rank: int, out_features: int) -> np.ndarray:
    if arr.shape == (out_features, rank):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape == (rank, out_features):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"LoRA B shape {arr.shape} does not match out={out_features}, rank={rank}")


@dataclass(frozen=True)
class NativeAdapterFile:
    file: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _hex64(self.sha256, "native adapter file sha256")
        if self.bytes <= 0:
            raise ValueError("native adapter file must be non-empty")


@dataclass(frozen=True)
class NativeAdapterTensor:
    target: str
    rank: int
    in_features: int
    out_features: int
    scale: float
    a: NativeAdapterFile
    b: NativeAdapterFile


@dataclass(frozen=True)
class NativeAdapterBundle:
    adapter_set_root: str
    bundle_root: str
    model_sha256: str
    foundation_model_digest: str
    source_adapter_manifest_digest: str
    source_adapter_set_bundle_digest: str
    qualified_candidate_digest: str
    tensors: tuple[NativeAdapterTensor, ...]
    directory: str
    schema: str = "qw3-native-lora-bundle-v1"

    def __post_init__(self) -> None:
        _hex64(self.adapter_set_root, "adapter_set_root")
        _hex64(self.bundle_root, "bundle_root")
        _hex64(self.model_sha256, "model_sha256")
        for d in (
            self.foundation_model_digest,
            self.source_adapter_manifest_digest,
            self.source_adapter_set_bundle_digest,
            self.qualified_candidate_digest,
        ):
            validate_digest(d)
        if not self.tensors:
            raise ValueError("native adapter bundle requires at least one tensor")


class NativeQW3AdapterCompiler:
    """Compile one qualified LM-head LoRA candidate into QW3-native FP32 form.

    NativeAdapter1 is intentionally narrow: every LoRA tensor in the source
    safetensors must belong to the LM head/output projection. If the source
    contains layer adapters, compilation fails rather than serving a partial
    representation while claiming the whole adapter_set_root.
    """

    def compile(
        self,
        *,
        plan: AdapterTrainingPlan,
        adapter: AdapterArtifactManifest,
        adapter_set: AdapterSetBundle,
        qualified: QualifiedAdapterSetCandidate,
        artifact_dir: str | Path,
        output_dir: str | Path,
        model_sha256: str,
        input_features: int,
        output_features: int,
    ) -> NativeAdapterBundle:
        _hex64(model_sha256, "model_sha256")
        if qualified.decision != "PASS":
            raise PermissionError("blocked adapter candidate cannot be compiled for native serving")
        if adapter.training_plan_digest != plan.digest:
            raise ValueError("adapter is not bound to the supplied training plan")
        if adapter.digest != qualified.adapter_manifest_digest:
            raise ValueError("qualified candidate references a different adapter artifact")
        if adapter_set.digest != qualified.adapter_set_bundle_digest:
            raise ValueError("qualified candidate references a different adapter set")
        if adapter_set.root_hex != qualified.adapter_set_root:
            raise ValueError("qualified adapter-set root mismatch")
        if len(adapter_set.adapters) != 1 or adapter_set.adapters[0].digest != adapter.digest:
            raise ValueError("NativeAdapter1 requires a single fully represented adapter artifact")
        if input_features <= 0 or output_features <= 0:
            raise ValueError("native adapter dimensions must be positive")

        root = Path(artifact_dir).resolve()
        tensors: dict[str, np.ndarray] = {}
        found_safetensors = False
        for row in adapter.files:
            path = (root / row.relative_path).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError("adapter payload path is missing or unsafe")
            raw = path.read_bytes()
            if "sha256:" + _sha256_bytes(raw) != row.sha256_digest or len(raw) != row.bytes:
                raise RuntimeError("adapter payload changed after qualification")
            if path.suffix == ".safetensors":
                found_safetensors = True
                for name, value in _load_safetensors(path).items():
                    if name in tensors:
                        raise ValueError(f"duplicate adapter tensor key: {name}")
                    tensors[name] = value
        if not found_safetensors:
            raise ValueError("NativeAdapter1 requires a safetensors LoRA artifact")

        pairs: dict[str, dict[str, np.ndarray]] = {}
        lora_keys = 0
        for key, value in tensors.items():
            role = _lora_role(key)
            if role is None:
                continue
            lora_keys += 1
            prefix, side = role
            if not _is_output_target(prefix):
                raise ValueError(
                    "NativeAdapter1 refuses partial adapter application; unsupported LoRA target: " + prefix
                )
            slot = pairs.setdefault(prefix, {})
            if side in slot:
                raise ValueError(f"duplicate LoRA {side.upper()} tensor for {prefix}")
            slot[side] = value
        if lora_keys == 0 or not pairs:
            raise ValueError("no LM-head LoRA tensors found in safetensors artifact")
        if any(set(pair) != {"a", "b"} for pair in pairs.values()):
            raise ValueError("incomplete LoRA A/B pair")

        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("native adapter output directory must be absent or empty")
        out.mkdir(parents=True, exist_ok=True)

        native_tensors: list[NativeAdapterTensor] = []
        for index, (prefix, pair) in enumerate(sorted(pairs.items())):
            # Infer rank from the dimension shared by A and B after trying the
            # two common PEFT/MLX orientations.
            candidates = set(pair["a"].shape) & set(pair["b"].shape)
            candidates.discard(input_features)
            candidates.discard(output_features)
            if len(candidates) != 1:
                # More direct inference for degenerate/equal feature shapes.
                possible = [d for d in pair["a"].shape if d <= min(input_features, output_features)]
                possible = [d for d in possible if d in pair["b"].shape]
                if len(set(possible)) != 1:
                    raise ValueError(f"cannot infer a unique LoRA rank for {prefix}")
                rank = int(possible[0])
            else:
                rank = int(next(iter(candidates)))
            if rank <= 0:
                raise ValueError("invalid LoRA rank")
            a = _normalize_a(pair["a"], rank, input_features)
            b = _normalize_b(pair["b"], rank, output_features)
            a_bytes = a.astype("<f4", copy=False).tobytes(order="C")
            b_bytes = b.astype("<f4", copy=False).tobytes(order="C")
            ap = out / f"output.weight.{index}.A.f32"
            bp = out / f"output.weight.{index}.B.f32"
            ap.write_bytes(a_bytes)
            bp.write_bytes(b_bytes)
            native_tensors.append(NativeAdapterTensor(
                target="output.weight",
                rank=rank,
                in_features=input_features,
                out_features=output_features,
                scale=float(plan.scale),
                a=NativeAdapterFile(ap.name, _sha256_bytes(a_bytes), len(a_bytes)),
                b=NativeAdapterFile(bp.name, _sha256_bytes(b_bytes), len(b_bytes)),
            ))

        manifest = {
            "schema": "qw3-native-lora-bundle-v1",
            "adapter_set_root": qualified.adapter_set_root,
            "model_sha256": model_sha256,
            "foundation_model_digest": plan.foundation_model_digest,
            "source_adapter_manifest_digest": adapter.digest,
            "source_adapter_set_bundle_digest": adapter_set.digest,
            "qualified_candidate_digest": qualified.digest,
            "tensors": [asdict(t) for t in native_tensors],
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_path = out / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        bundle_root = _sha256_bytes(manifest_bytes)
        return NativeAdapterBundle(
            adapter_set_root=qualified.adapter_set_root,
            bundle_root=bundle_root,
            model_sha256=model_sha256,
            foundation_model_digest=plan.foundation_model_digest,
            source_adapter_manifest_digest=adapter.digest,
            source_adapter_set_bundle_digest=adapter_set.digest,
            qualified_candidate_digest=qualified.digest,
            tensors=tuple(native_tensors),
            directory=str(out.resolve()),
        )

    @staticmethod
    def bind_served_manifest(
        *,
        base: ServedArtifactManifest,
        qualified: QualifiedAdapterSetCandidate,
        native_bundle: NativeAdapterBundle,
    ) -> ServedArtifactManifest:
        if qualified.decision != "PASS":
            raise PermissionError("blocked adapter cannot alter served state")
        if native_bundle.adapter_set_root != qualified.adapter_set_root:
            raise ValueError("native bundle is not bound to qualified adapter set")
        if native_bundle.qualified_candidate_digest != qualified.digest:
            raise ValueError("native bundle is bound to a different qualified candidate")
        if base.foundation_digest != qualified.foundation_model_digest.split(":", 1)[1]:
            raise PermissionError("native adapter foundation does not match served foundation")
        return replace(
            base,
            adapter_set_root=qualified.adapter_set_root,
            native_adapter_bundle_root=native_bundle.bundle_root,
        )


# ---------------------------------------------------------------------------
# NativeAdapter2 — v2 bundles with per-layer attention LoRA (Route B).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeAdapterTensorV2:
    target: str            # "output.weight" or "self_attn.{q,k,v,o}_proj"
    layer: int             # kNoLayer equivalent for output.weight: -1
    rank: int
    in_features: int
    out_features: int
    scale: float
    a: NativeAdapterFile
    b: NativeAdapterFile


@dataclass(frozen=True)
class NativeAdapter2Bundle:
    adapter_set_root: str
    bundle_root: str
    model_sha256: str
    foundation_model_digest: str
    source_adapter_manifest_digest: str
    qualified_candidate_digest: str
    tensors: tuple[NativeAdapterTensorV2, ...]
    directory: str
    schema: str = "qw3-native-lora-bundle-v2"

    def __post_init__(self) -> None:
        _hex64(self.adapter_set_root, "adapter_set_root")
        _hex64(self.bundle_root, "bundle_root")
        _hex64(self.model_sha256, "model_sha256")
        for d in (self.foundation_model_digest,
                  self.source_adapter_manifest_digest,
                  self.qualified_candidate_digest):
            validate_digest(d)
        if not self.tensors:
            raise ValueError("native adapter v2 bundle requires tensors")


# Expected model geometry the bundle must match: hidden (n_embd), q_rows
# (q_proj output width), kv_rows (k/v output width), n_layers, plus the
# LM-head pair lm_head_in/lm_head_out when an output.weight entry exists.
@dataclass(frozen=True)
class NativeAttentionDims:
    n_layers: int
    hidden: int
    q_rows: int   # q_proj output width (includes gate on gated models)
    kv_rows: int  # k/v output width
    o_in: int     # o_proj input width — post-attention mid width


class NativeQW3Adapter2Compiler:
    """Compile a qualified PEFT adapter into a v2 native LoRA bundle.

    Every lora tensor must resolve to either the LM head (v1 target) or a
    `layers.<n>.self_attn.{q,k,v,o}_proj` module — anything else fails
    compilation rather than silently dropping weights. Dims are verified
    against declared model geometry; tensor contents are re-hashed after
    layout normalization.
    """

    SCHEMA = "qw3-native-lora-bundle-v2"

    def compile(
        self,
        *,
        adapter_dir: str | Path,
        output_dir: str | Path,
        adapter_set_root: str,
        model_sha256: str,
        foundation_model_digest: str,
        source_adapter_manifest_digest: str,
        qualified_candidate_digest: str,
        dims: NativeAttentionDims,
        lm_head_in: int | None = None,
        lm_head_out: int | None = None,
        lora_alpha: float,
    ) -> NativeAdapter2Bundle:
        _hex64(adapter_set_root, "adapter_set_root")
        _hex64(model_sha256, "model_sha256")
        for d in (foundation_model_digest, source_adapter_manifest_digest,
                  qualified_candidate_digest):
            validate_digest(d)

        root = Path(adapter_dir).resolve()
        tensors: dict[str, np.ndarray] = {}
        for path in sorted(root.iterdir()):
            if path.suffix == ".safetensors":
                for name, value in _load_safetensors(path).items():
                    if name in tensors:
                        raise ValueError(f"duplicate adapter tensor key: {name}")
                    tensors[name] = value
        if not tensors:
            raise ValueError("NativeAdapter2 requires a safetensors artifact")

        pairs: dict[str, dict[str, np.ndarray]] = {}
        for key, value in tensors.items():
            role = _lora_role(key)
            if role is None:
                continue  # non-LoRA tensors (e.g. modules_to_save) ignored
            prefix, side = role
            slot = pairs.setdefault(prefix, {})
            if side in slot:
                raise ValueError(f"duplicate LoRA {side.upper()} for {prefix}")
            slot[side] = value
        if not pairs:
            raise ValueError("no LoRA tensors found in safetensors artifact")
        if any(set(p) != {"a", "b"} for p in pairs.values()):
            raise ValueError("incomplete LoRA A/B pair")

        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("native adapter output dir must be absent/empty")
        out.mkdir(parents=True, exist_ok=True)

        entries: list[NativeAdapterTensorV2] = []
        for prefix, pair in sorted(pairs.items()):
            attn = _attention_target_kind(prefix)
            if attn is not None:
                layer, kind = attn
                if layer >= dims.n_layers:
                    raise ValueError(f"adapter layer {layer} >= n_layers")
                want_in = dims.o_in if kind == "o_proj" else dims.hidden
                want_out = dims.hidden if kind == "o_proj" else (
                    dims.q_rows if kind == "q_proj" else dims.kv_rows)
                target = f"self_attn.{kind}"
                layer_i = layer
            elif _is_output_target(prefix):
                if lm_head_in is None or lm_head_out is None:
                    raise ValueError("output.weight entry requires lm_head dims")
                want_in, want_out = lm_head_in, lm_head_out
                target = "output.weight"
                layer_i = -1
            else:
                raise ValueError(
                    "NativeAdapter2 refuses unsupported LoRA target: " + prefix)

            a_raw, b_raw = pair["a"], pair["b"]
            rank_candidates = set(a_raw.shape) & set(b_raw.shape)
            rank_candidates.discard(want_in)
            rank_candidates.discard(want_out)
            if len(rank_candidates) != 1:
                raise ValueError(f"cannot infer unique LoRA rank for {prefix}")
            rank = int(next(iter(rank_candidates)))
            a = _normalize_a(a_raw, rank, want_in)
            b = _normalize_b(b_raw, rank, want_out)
            scale = float(lora_alpha) / float(rank)
            stem = (f"L{layer_i}.{kind}" if attn is not None
                    else "output.weight")
            a_bytes = a.astype("<f4", copy=False).tobytes(order="C")
            b_bytes = b.astype("<f4", copy=False).tobytes(order="C")
            ap, bp = out / f"{stem}.A.f32", out / f"{stem}.B.f32"
            ap.write_bytes(a_bytes)
            bp.write_bytes(b_bytes)
            entries.append(NativeAdapterTensorV2(
                target=target, layer=layer_i, rank=rank,
                in_features=want_in, out_features=want_out, scale=scale,
                a=NativeAdapterFile(ap.name, _sha256_bytes(a_bytes), len(a_bytes)),
                b=NativeAdapterFile(bp.name, _sha256_bytes(b_bytes), len(b_bytes))))

        manifest = {
            "schema": self.SCHEMA,
            "adapter_set_root": adapter_set_root,
            "model_sha256": model_sha256,
            "foundation_model_digest": foundation_model_digest,
            "source_adapter_manifest_digest": source_adapter_manifest_digest,
            "qualified_candidate_digest": qualified_candidate_digest,
            "tensors": [
                ({**{k: getattr(t, k) for k in
                     ("target", "rank", "in_features", "out_features", "scale")},
                  **({"layer": t.layer} if t.layer >= 0 else {}),
                  "a": asdict(t.a), "b": asdict(t.b)})
                for t in entries],
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        (out / "manifest.json").write_bytes(manifest_bytes)
        return NativeAdapter2Bundle(
            adapter_set_root=adapter_set_root,
            bundle_root=_sha256_bytes(manifest_bytes),
            model_sha256=model_sha256,
            foundation_model_digest=foundation_model_digest,
            source_adapter_manifest_digest=source_adapter_manifest_digest,
            qualified_candidate_digest=qualified_candidate_digest,
            tensors=tuple(entries),
            directory=str(out.resolve()))
