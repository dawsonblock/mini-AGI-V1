"""v16.4.5 worker IPC protocol tests (SEC-401).

The framed JSON codec is the ONLY thing between the supervisor and
worker-controlled bytes. Every structural violation must refuse
before allocation/parsing — never negotiate, never execute.
"""
import io
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.worker_protocol import (  # noqa: E402
    MAX_DEPTH, MAX_FRAME_BYTES, MAX_KEY_COUNT, PROTOCOL, OverflowRef,
    ProtocolRefused, WorkerProtocolError, decode_typed, encode_request,
    encode_response, encode_typed, read_frame, read_overflow,
    strict_json_loads, write_overflow)


def _stream(*blobs: bytes) -> io.BytesIO:
    return io.BytesIO(b"".join(blobs))


def _frame(obj) -> bytes:
    body = json.dumps(obj).encode()
    return struct.pack(">I", len(body)) + body


def _msg(**kw):
    return dict({"protocol": PROTOCOL, "request_id": "r1",
                 "type": "result", "payload": {}}, **kw)


# ---------- happy path -----------------------------------------------------

def test_request_response_roundtrip():
    blob = encode_request("r7", "infer", {"request": {"prompt": "hi"}})
    msg = read_frame(_stream(blob), types={"infer"})
    assert msg["request_id"] == "r7"
    assert msg["type"] == "infer"
    assert msg["payload"]["request"]["prompt"] == "hi"

    blob = encode_response("r7", True, {"result": {"text": "ok"}})
    msg = read_frame(_stream(blob), types={"result"})
    assert msg["ok"] is True
    assert msg["payload"]["result"]["text"] == "ok"


def test_eof_is_clean_close_not_an_error():
    assert read_frame(_stream(), types={"result"}) is None


def test_multiple_frames_read_in_order():
    s = _stream(encode_request("a", "probe", {}),
                encode_request("b", "probe", {}))
    assert read_frame(s, types={"probe"})["request_id"] == "a"
    assert read_frame(s, types={"probe"})["request_id"] == "b"


# ---------- frame-level attacks ---------------------------------------------

def test_oversized_frame_refused_before_reading():
    # announce a >1 MiB payload; the reader refuses at the header,
    # before allocating or reading any payload bytes
    s = _stream(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(ProtocolRefused, match="bound"):
        read_frame(s, types={"result"})


def test_malformed_header_refused():
    with pytest.raises((ProtocolRefused, WorkerProtocolError)):
        read_frame(_stream(b"\x00\x01"), types={"result"})


def test_truncated_payload_is_eof_mid_frame():
    blob = encode_request("r", "probe", {"x": "y" * 100})
    with pytest.raises(WorkerProtocolError, match="mid-frame"):
        read_frame(_stream(blob[:-50]), types={"probe"})


def test_non_json_payload_refused():
    with pytest.raises(ProtocolRefused):
        read_frame(_stream(struct.pack(">I", 7) + b"\xff" * 7),
                   types={"result"})


def test_pickle_payload_is_not_a_message():
    """A pickled Python object is just bytes that fail JSON/schema —
    nothing in the supervisor ever calls a deserializer on it."""
    import pickle
    evil = pickle.dumps({"__reduce__": "os.system"})
    with pytest.raises(ProtocolRefused):
        read_frame(_stream(struct.pack(">I", len(evil)) + evil),
                   types={"result"})


def test_wrong_protocol_field_refused():
    with pytest.raises(ProtocolRefused, match="protocol"):
        read_frame(_stream(_frame(_msg(protocol="pickle-v1"))),
                   types={"result"})


def test_unknown_message_type_refused():
    from minagi.runtime.worker_protocol import RESPONSE_TYPES
    with pytest.raises(ProtocolRefused, match="message type"):
        read_frame(_stream(_frame(_msg(type="exec", payload={
            "code": "import os;os.system('id')"}))),
            types=RESPONSE_TYPES)


def test_message_type_filter_enforced():
    blob = encode_request("r", "infer", {})
    with pytest.raises(ProtocolRefused):
        read_frame(_stream(blob), types={"probe"})


def test_missing_request_id_refused():
    with pytest.raises(ProtocolRefused):
        read_frame(_stream(_frame(_msg(request_id=None))),
                   types={"result"})


def test_ok_flag_must_be_boolean():
    with pytest.raises(ProtocolRefused):
        read_frame(_stream(_frame(_msg(ok="yes"))), types={"result"})


def test_unexpected_top_level_fields_refused():
    with pytest.raises(ProtocolRefused, match="fields"):
        read_frame(_stream(_frame(_msg(eval="1+1"))), types={"result"})


# ---------- JSON body attacks -----------------------------------------------

def test_deep_nesting_refused():
    deep = cur = {}
    for _ in range(MAX_DEPTH + 5):
        cur["a"] = {}
        cur = cur["a"]
    with pytest.raises(ProtocolRefused, match="nests"):
        read_frame(_stream(_frame(_msg(payload=deep))),
                   types={"result"})


def test_key_count_bound_refused():
    wide = {f"k{i}": i for i in range(MAX_KEY_COUNT + 1)}
    with pytest.raises(ProtocolRefused, match="keys"):
        strict_json_loads(json.dumps(_msg(payload=wide)).encode())


def test_duplicate_keys_refused():
    raw = (b'{"protocol":"' + PROTOCOL.encode() +
           b'","request_id":"r","type":"result","ok":true,'
           b'"payload":{"x":1},"payload":{"x":2}}')
    with pytest.raises(ProtocolRefused, match="duplicate"):
        strict_json_loads(raw)


def test_non_utf8_payload_refused():
    with pytest.raises(ProtocolRefused):
        strict_json_loads(b'{"a": "\xff\xfe"}')


def test_nan_inf_refused():
    with pytest.raises(ProtocolRefused):
        strict_json_loads(b'{"x": NaN}')


# ---------- typed values / overflow ------------------------------------------

def test_typed_values_roundtrip():
    """Boot kwargs may carry config objects that implement to_doc —
    reconstruction is explicit (from_doc) inside project modules."""
    from minagi.runtime.inference_policy import InferenceBudgetPolicyV1
    enc = encode_typed({"budget": InferenceBudgetPolicyV1(
        max_prompt_tokens=64, max_new_tokens=16)})
    assert enc["budget"]["__typed__"].startswith("minagi.")
    out = decode_typed(enc)
    assert isinstance(out["budget"], InferenceBudgetPolicyV1)
    assert out["budget"].max_prompt_tokens == 64


def test_decode_typed_foreign_module_refused():
    with pytest.raises(ProtocolRefused, match="project packages"):
        decode_typed({"__typed__": "os:system", "doc": {}})


def test_decode_typed_malformed_spec_refused():
    with pytest.raises(ProtocolRefused):
        decode_typed({"__typed__": 123, "doc": {}})


def test_encode_typed_unknown_object_refused():
    with pytest.raises(WorkerProtocolError):
        encode_typed(object())


def test_overflow_file_is_content_addressed(tmp_path):
    ref = write_overflow(tmp_path, "big.result.json",
                         {"rows": list(range(100))})
    assert (tmp_path / ref.name).is_file()
    out = read_overflow(tmp_path, ref)
    assert out["rows"][99] == 99


def test_overflow_name_traversal_refused():
    with pytest.raises(ProtocolRefused):
        OverflowRef.from_doc({"name": "../x.json",
                              "sha256": "0" * 64, "size": 2})


def test_overflow_digest_mismatch_refused(tmp_path):
    ref = write_overflow(tmp_path, "f.json", {"a": 1})
    (tmp_path / "f.json").write_text('{"a":9}')    # same size, new bytes
    with pytest.raises(WorkerProtocolError, match="digest"):
        read_overflow(tmp_path, ref)


def test_frame_encode_bounds_the_request():
    with pytest.raises(WorkerProtocolError):
        encode_request("r", "infer",
                       {"blob": "x" * (MAX_FRAME_BYTES + 1)})
