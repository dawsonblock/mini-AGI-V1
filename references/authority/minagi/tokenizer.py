"""Tokenization for mini-AGI.

v4 keeps the original byte tokenizer for checkpoint compatibility and adds a
first-class ByteLevel-BPE path for new model families.  A ByteLevel BPE model
still has complete byte fallback, but common words/code fragments collapse to
far fewer positions.  The fast trainer/runtime are provided by Hugging Face
``tokenizers`` when that optional dependency is installed.

A tokenizer is part of the model's architecture.  New checkpoints copy their
``tokenizer.json`` into the weights root and record its SHA-256 in the model
config; serving never guesses a tokenizer from the current checkout.
"""

from __future__ import annotations

import json
import os
import re as _re
from pathlib import Path


class _Enc:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


SPECIALS = ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>",
            "<g>", "</g>", "<|endoftext|>"]
SPECIAL_ID = {s: 256 + i for i, s in enumerate(SPECIALS)}
ID_SPECIAL = {v: k for k, v in SPECIAL_ID.items()}
_SPECIAL_RE = _re.compile("|".join(_re.escape(s) for s in SPECIALS))


class ByteTokenizer:
    """Raw UTF-8 bytes plus structural markers; every input is representable."""

    size = 256 + len(SPECIALS)
    kind = "byte"

    def encode(self, text):
        ids, pos = [], 0
        for m in _SPECIAL_RE.finditer(text):
            ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.append(SPECIAL_ID[m.group(0)])
            pos = m.end()
        ids.extend(text[pos:].encode("utf-8"))
        return _Enc(ids)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i in ID_SPECIAL:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(ID_SPECIAL[i])
            elif 0 <= i < 256:
                buf.append(i)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def get_vocab_size(self):
        return self.size

    def token_to_id(self, token):
        if token in SPECIAL_ID:
            return SPECIAL_ID[token]
        b = token.encode("utf-8")
        return b[0] if len(b) == 1 else None


def _hf_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise RuntimeError(
            "this model uses a ByteLevel-BPE tokenizer; install the optional "
            "dependency with `pip install -r requirements-v4.txt`") from e
    return Tokenizer.from_file(str(path))


def tokenizer_info(tok, path=None):
    """Small serializable architecture description for manifests."""
    kind = getattr(tok, "kind", None)
    if kind == "byte" or isinstance(tok, ByteTokenizer):
        return {"kind": "byte", "vocab_size": ByteTokenizer.size}
    n = int(tok.get_vocab_size())
    out = {"kind": "bytelevel_bpe", "vocab_size": n}
    if path:
        from .integrity import sha256_file
        out["file"] = "tokenizer.json"
        out["sha256"] = sha256_file(path)
    return out


def load_tokenizer(data_dir):
    """Load the tokenizer declared by a packed corpus directory."""
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        kind = meta.get("tokenizer")
        if kind == "byte":
            return ByteTokenizer()
        tok_file = meta.get("tokenizer_file", "tokenizer.json")
        path = os.path.join(data_dir, tok_file)
        if kind in ("bytelevel_bpe", "bpe") or os.path.exists(path):
            return _hf_tokenizer(path)
    path = os.path.join(data_dir, "tokenizer.json")
    if os.path.exists(path):
        return _hf_tokenizer(path)
    return ByteTokenizer()


def load_model_tokenizer(weights_dir, manifest=None):
    """Load exactly the tokenizer bound to a weights directory.

    Legacy/v3 manifests without tokenizer metadata are byte models.  For v4
    BPE models, the tokenizer is copied into the weights root and its digest is
    verified before use so an accidental tokenizer swap cannot silently change
    every token ID.
    """
    manifest = manifest or {}
    cfg = manifest.get("cfg") or {}
    spec = cfg.get("tokenizer") or manifest.get("tokenizer") or {}
    if not spec or spec.get("kind", "byte") == "byte":
        return ByteTokenizer()
    path = Path(weights_dir) / spec.get("file", "tokenizer.json")
    if not path.exists():
        raise RuntimeError(f"checkpoint requires tokenizer but it is missing: {path}")
    expected = spec.get("sha256")
    if expected:
        from .integrity import sha256_file
        got = sha256_file(path)
        if got != expected:
            raise RuntimeError(
                f"tokenizer digest mismatch: expected {expected}, got {got}")
    tok = _hf_tokenizer(path)
    want = int(spec.get("vocab_size", tok.get_vocab_size()))
    if tok.get_vocab_size() != want:
        raise RuntimeError(
            f"tokenizer vocabulary mismatch: manifest {want}, file {tok.get_vocab_size()}")
    return tok


def copy_tokenizer_into_weights(tokenizer_path, weights_dir):
    """Copy and bind a tokenizer file to a fresh model directory."""
    from shutil import copy2
    from .integrity import sha256_file
    src = Path(tokenizer_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = Path(weights_dir) / "tokenizer.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(src, dst)
    tok = _hf_tokenizer(dst)
    return {"kind": "bytelevel_bpe", "file": "tokenizer.json",
            "vocab_size": int(tok.get_vocab_size()), "sha256": sha256_file(dst)}
