"""v16.4.5 worker IPC protocol v2 — strict framed JSON (SEC-401).

v16.4.4's worker channel transported ``pickle`` frames in both
directions. That meant the supervisor executed ``pickle.load`` on
bytes fully controlled by the less-trusted worker — a compromised or
wedged worker could run arbitrary code inside the supervisor process,
defeating the entire isolation boundary RUN-401 was built for.

The v2 protocol replaces deserialization with a closed schema:

  * a frame is a 4-byte unsigned big-endian payload length followed
    by exactly that many bytes of UTF-8 JSON;
  * the length is checked BEFORE any payload bytes are read — an
    announced giant frame is refused without allocating for it;
  * the JSON parser is strict: duplicate object keys, NaN/Infinity
    literals, malformed UTF-8, and nesting deeper than ``MAX_DEPTH``
    are all refused;
  * every message must match the closed message schema — a fixed
    protocol tag, a bounded request id, a type drawn from a closed
    set, and a dict payload. Unknown types, missing fields, and
    unexpected shapes are protocol violations, not tolerated input;
  * responses carry only JSON primitives — never objects a remote
    side could materialize into live classes. The single typed-value
    escape hatch (``typed_value``) exists only for the boot kwargs
    the supervisor sends INTO the worker, is restricted to classes
    under the project's own modules, and requires an explicit
    ``from_doc`` reconstruction protocol. The supervisor never
    decodes typed values from the worker.

One codec serves both directions. There is no compatibility path and
no silent fallback to pickle — a worker that emits a pickle blob (or
anything else that is not a valid v2 frame) violates the protocol and
its channel is torn down.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

#: Protocol identifier required on every message.
PROTOCOL = "minagi-worker-v2"

#: Engineering limits (SEC-401). These are starting bounds, subject to
#: profiling — they exist so a hostile or wedged worker cannot turn the
#: frame channel into a memory or CPU exhaustion primitive.
MAX_FRAME_BYTES = 1 << 20            # 1 MiB control-frame ceiling
MAX_DEPTH = 16                       # JSON nesting bound
MAX_KEY_COUNT = 8192                 # total object keys per message
MAX_REQUEST_ID_LEN = 128
MAX_PENDING_REQUESTS = 16            # outstanding requests per worker
MAX_OVERFLOW_BYTES = 64 << 20        # content-addressed result-file bound
MAX_OVERFLOW_NAME = 128

#: Closed operation vocabulary — supervisor -> worker.
REQUEST_TYPES = frozenset(
    {"boot", "load", "probe", "infer", "cancel", "unload", "shutdown"})
#: Closed response vocabulary — worker -> supervisor.
RESPONSE_TYPES = frozenset({"result", "error"})

#: Modules a typed value may name. Boot kwargs are operator
#: configuration travelling supervisor -> worker; even so, the only
#: re-materialization allowed is an explicit ``from_doc`` on a class
#: inside the project's own packages — no arbitrary construction.
_TYPED_MODULE_PREFIXES = ("minagi.", "egai.", "kvcontinual.",
                          "dream_rsi_governed.")


class ProtocolRefused(PermissionError):
    """A message or frame violated the worker protocol — fail closed."""


class WorkerProtocolError(RuntimeError):
    """The channel itself failed (framing, encoding, transport)."""


@dataclass(frozen=True)
class OverflowRef:
    """A result payload too large for a control frame, stored as a
    content-addressed JSON file inside the parent-provided overflow
    directory. Only the file NAME travels in the message — the
    supervisor resolves it inside the directory it itself supplied."""
    name: str
    sha256: str
    size: int

    def to_doc(self) -> dict:
        return {"name": self.name, "sha256": self.sha256,
                "size": self.size}

    @classmethod
    def from_doc(cls, doc) -> "OverflowRef":
        if not isinstance(doc, dict):
            raise ProtocolRefused("overflow reference must be an object")
        name = str(doc.get("name") or "")
        if not name or len(name) > MAX_OVERFLOW_NAME or \
                name != Path(name).name or name.startswith("."):
            raise ProtocolRefused(
                f"overflow reference name {name!r} is not a plain file "
                "name inside the overflow directory")
        sha = str(doc.get("sha256") or "")
        if len(sha) != 64 or any(c not in "0123456789abcdef"
                                 for c in sha):
            raise ProtocolRefused("overflow reference sha256 malformed")
        size = doc.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or \
                size < 0 or size > MAX_OVERFLOW_BYTES:
            raise ProtocolRefused(
                "overflow reference size is missing or out of bounds")
        return cls(name=name, sha256=sha, size=size)


# --- strict JSON ---------------------------------------------------------

def _check_depth(value, depth: int = 0, _keys=None) -> None:
    if _keys is None:
        _keys = [0]
    if depth > MAX_DEPTH:
        raise ProtocolRefused(
            f"message nests deeper than {MAX_DEPTH} levels")
    if isinstance(value, dict):
        _keys[0] += len(value)
        if _keys[0] > MAX_KEY_COUNT:
            raise ProtocolRefused(
                f"message carries more than {MAX_KEY_COUNT} object "
                "keys — structure bound exceeded")
        for v in value.values():
            _check_depth(v, depth + 1, _keys)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check_depth(v, depth + 1, _keys)


def _no_duplicates(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ProtocolRefused(
                f"duplicate object key {k!r} — ambiguous input refused")
        out[k] = v
    return out


def _bad_constant(name: str):
    raise ProtocolRefused(
        f"non-standard JSON constant {name!r} — strict JSON only")


def strict_json_loads(raw: bytes):
    """Parse `raw` as strict JSON: well-formed UTF-8, no duplicate
    keys, no NaN/Infinity, bounded nesting, object-or-array root."""
    if len(raw) > MAX_FRAME_BYTES + MAX_OVERFLOW_BYTES:
        raise ProtocolRefused("payload exceeds every configured bound")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolRefused(f"payload is not valid UTF-8: {exc}") \
            from exc
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicates,
                           parse_constant=_bad_constant)
    except ProtocolRefused:
        raise
    except ValueError as exc:
        raise ProtocolRefused(f"payload is not valid JSON: {exc}") \
            from exc
    _check_depth(value)
    return value


def canonical_json(value) -> bytes:
    """Deterministic encoding used on the wire."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


# --- message schema --------------------------------------------------------

def _validate_request_id(rid) -> str:
    if not isinstance(rid, str) or not rid or \
            len(rid) > MAX_REQUEST_ID_LEN:
        raise ProtocolRefused(
            "request_id must be a non-empty bounded string")
    return rid


def validate_message(msg, *, types: frozenset) -> dict:
    """Envelope validation common to both directions."""
    if not isinstance(msg, dict):
        raise ProtocolRefused("message must be a JSON object")
    if msg.get("protocol") != PROTOCOL:
        raise ProtocolRefused(
            f"message protocol {msg.get('protocol')!r} is not "
            f"{PROTOCOL!r}")
    _validate_request_id(msg.get("request_id"))
    mtype = msg.get("type")
    if mtype not in types:
        raise ProtocolRefused(
            f"unknown message type {mtype!r} — the protocol vocabulary "
            "is closed")
    if not isinstance(msg.get("payload"), dict):
        raise ProtocolRefused("message payload must be an object")
    if "ok" in msg and not isinstance(msg["ok"], bool):
        raise ProtocolRefused("message 'ok' must be a boolean")
    allowed = {"protocol", "request_id", "type", "ok", "payload",
               "error"}
    extra = set(msg) - allowed
    if extra:
        raise ProtocolRefused(
            f"unexpected top-level fields {sorted(extra)}")
    return msg


def encode_request(request_id: str, op: str, payload: dict) -> bytes:
    if op not in REQUEST_TYPES:
        raise ProtocolRefused(f"cannot encode unknown op {op!r}")
    return encode_frame({"protocol": PROTOCOL,
                         "request_id": request_id,
                         "type": op,
                         "payload": dict(payload or {})})


def encode_response(request_id: str, ok: bool, payload: dict) -> bytes:
    msg = {"protocol": PROTOCOL, "request_id": request_id,
           "type": "result" if ok else "error",
           "ok": bool(ok), "payload": dict(payload or {})}
    return encode_frame(msg)


def encode_frame(msg: dict) -> bytes:
    """Schema-validate and serialize one message into a frame."""
    types = REQUEST_TYPES | RESPONSE_TYPES
    validate_message(msg, types=types)
    blob = canonical_json(msg)
    if len(blob) > MAX_FRAME_BYTES:
        raise WorkerProtocolError(
            f"message encodes to {len(blob)} bytes > the "
            f"{MAX_FRAME_BYTES}-byte control bound")
    return struct.pack(">I", len(blob)) + blob


def _read_exactly(stream, n: int) -> bytes | None:
    """Read exactly `n` bytes; None on clean EOF before the first
    byte, ProtocolRefused on truncation mid-read."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise WorkerProtocolError(
                "channel ended mid-frame — a truncated message is not "
                "a partial message, it is a dead channel")
        buf += chunk
    return bytes(buf)


def read_frame(stream, *, types: frozenset) -> dict | None:
    """Read one validated message from `stream`; None on clean EOF at
    a frame boundary. Raises ProtocolRefused on any violation — the
    announced length is checked BEFORE the payload is read, so an
    oversized announcement is refused without allocation."""
    header = _read_exactly(stream, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES or length == 0:
        raise ProtocolRefused(
            f"declared frame length {length} violates the "
            f"{MAX_FRAME_BYTES}-byte bound — refused before read")
    raw = _read_exactly(stream, length)
    if raw is None:
        raise WorkerProtocolError("channel ended before payload")
    msg = strict_json_loads(raw)
    return validate_message(msg, types=types)


# --- typed values (boot kwargs only: supervisor -> worker) ---------------

def encode_typed(value):
    """Encode a config value for transport. JSON primitives pass
    through; an object may travel only as ``{__typed__:
    module:QualName, doc: obj.to_doc()}`` when it implements
    ``to_doc`` — an explicit, project-internal reconstruction
    protocol, never an object graph."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): encode_typed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_typed(v) for v in value]
    to_doc = getattr(value, "to_doc", None)
    if callable(to_doc):
        cls = type(value)
        spec = f"{cls.__module__}:{cls.__qualname__}"
        return {"__typed__": spec, "doc": encode_typed(to_doc())}
    raise WorkerProtocolError(
        f"value of type {type(value).__name__} cannot cross the "
        "worker boundary — no to_doc protocol")


def decode_typed(value):
    """Re-materialize a ``encode_typed`` value — used ONLY inside the
    worker for boot kwargs (the supervisor's own configuration). The
    class must live under a project module and implement
    ``from_doc``."""
    if isinstance(value, list):
        return [decode_typed(v) for v in value]
    if isinstance(value, dict):
        spec = value.get("__typed__")
        if spec is None:
            return {str(k): decode_typed(v) for k, v in value.items()}
        if not isinstance(spec, str) or ":" not in spec:
            raise ProtocolRefused("typed value spec malformed")
        module_name, qualname = spec.split(":", 1)
        if not module_name.startswith(_TYPED_MODULE_PREFIXES):
            raise ProtocolRefused(
                f"typed value module {module_name!r} is outside the "
                "project packages")
        try:
            obj = importlib.import_module(module_name)
            for part in qualname.split("."):
                obj = getattr(obj, part)
        except Exception as exc:  # noqa: BLE001 - spec errors refuse
            raise ProtocolRefused(
                f"typed value {spec!r} does not resolve: {exc}") from exc
        from_doc = getattr(obj, "from_doc", None)
        if not callable(from_doc):
            raise ProtocolRefused(
                f"typed value {spec!r} names a class without from_doc")
        return from_doc(decode_typed(value["doc"]))
    return value


# --- content-addressed overflow ------------------------------------------

def write_overflow(directory, name: str, value) -> OverflowRef:
    """Store a JSON value as `<directory>/<name>` and return its
    content reference. Atomic: written to a sibling temp file then
    renamed, so a partial write can never masquerade as a result."""
    directory = Path(directory)
    raw = canonical_json(value)
    if len(raw) > MAX_OVERFLOW_BYTES:
        raise WorkerProtocolError(
            f"result payload {len(raw)} bytes exceeds the "
            f"{MAX_OVERFLOW_BYTES}-byte overflow bound")
    target = directory / name
    tmp = directory / f".{name}.tmp"
    tmp.write_bytes(raw)
    tmp.replace(target)
    return OverflowRef(
        name=name, sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))


def read_overflow(directory, ref: OverflowRef):
    """Read + verify an overflow result the worker stored. Only the
    caller-supplied directory and the message-carried file name are
    combined — a worker cannot redirect the read elsewhere — and the
    bytes must match the announced digest before they are parsed."""
    target = Path(directory) / ref.name
    try:
        if target.is_symlink() or not target.is_file():
            raise WorkerProtocolError(
                "overflow result path is not a regular file")
        size = target.stat().st_size
        if size != ref.size or size > MAX_OVERFLOW_BYTES:
            raise WorkerProtocolError(
                f"overflow result size {size} does not match the "
                f"announced {ref.size} bytes")
        raw = target.read_bytes()
    except WorkerProtocolError:
        raise
    except OSError as exc:
        raise WorkerProtocolError(
            f"overflow result unreadable: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise WorkerProtocolError(
            "overflow result digest mismatch — the stored bytes are "
            "not the bytes the reply announced")
    return strict_json_loads(raw)


__all__ = ["MAX_DEPTH", "MAX_FRAME_BYTES", "MAX_KEY_COUNT",
           "MAX_OVERFLOW_BYTES",
           "MAX_PENDING_REQUESTS", "MAX_REQUEST_ID_LEN", "OverflowRef",
           "PROTOCOL", "ProtocolRefused", "REQUEST_TYPES",
           "RESPONSE_TYPES", "WorkerProtocolError", "canonical_json",
           "decode_typed", "encode_frame", "encode_request",
           "encode_response", "encode_typed", "read_frame",
           "read_overflow", "strict_json_loads", "validate_message",
           "write_overflow"]
