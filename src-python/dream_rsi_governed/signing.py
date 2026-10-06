from __future__ import annotations

import base64
import hashlib
from dataclasses import replace

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

from .canonical import canonical_json_bytes
from .models import PromotionManifest


def public_key_id(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return "ed25519:" + hashlib.sha256(raw).hexdigest()[:24]


def generate_private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def private_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())


def private_key_from_bytes(data: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(data)


def _unsigned(manifest: PromotionManifest):
    return replace(manifest, signature_b64="")


def sign_manifest(manifest: PromotionManifest, key: Ed25519PrivateKey) -> PromotionManifest:
    kid = public_key_id(key.public_key())
    if manifest.signer_key_id != kid:
        raise ValueError("manifest signer key id does not match private key")
    sig = key.sign(canonical_json_bytes(_unsigned(manifest)))
    return replace(manifest, signature_b64=base64.b64encode(sig).decode("ascii"))


def verify_manifest(manifest: PromotionManifest, key: Ed25519PublicKey) -> bool:
    if manifest.signer_key_id != public_key_id(key):
        return False
    try:
        key.verify(base64.b64decode(manifest.signature_b64), canonical_json_bytes(_unsigned(manifest)))
        return True
    except Exception:
        return False
