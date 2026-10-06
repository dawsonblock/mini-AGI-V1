from __future__ import annotations
import base64
import json
import os
from dataclasses import asdict, dataclass
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from egai.common.canonical import canonical_bytes, digest, sha256_bytes
from egai.replay.policy import GroundedReplayEpisode, GroundedOutcome, ReplayRound


@dataclass(frozen=True)
class SealedReplaySuite:
    suite_id: str
    generation: int
    algorithm: str
    nonce_b64: str
    ciphertext_b64: str
    aad_digest: str
    episode_count: int

    @property
    def sealed_digest(self) -> str:
        return digest(asdict(self))


class ReplayHistoryVault:
    ALGORITHM = "AES-256-GCM"

    @staticmethod
    def generate_key() -> bytes:
        return AESGCM.generate_key(bit_length=256)

    @staticmethod
    def _encode(episodes) -> bytes:
        return canonical_bytes([asdict(e) for e in episodes])

    @staticmethod
    def _decode(raw: bytes) -> tuple[GroundedReplayEpisode, ...]:
        wrapped = json.loads(raw)
        data = wrapped.get("value", wrapped)
        out = []
        for e in data:
            rounds = []
            for r in e["rounds"]:
                outcomes = tuple(GroundedOutcome(**o) for o in r["outcomes"])
                rounds.append(ReplayRound(r["t"], r["visible_prefix_digest"], tuple(r["legal_actions"]), outcomes))
            out.append(GroundedReplayEpisode(e["episode_id"], tuple(rounds), float(e["budget"]), e.get("episode_digest", "")))
        return tuple(out)

    @classmethod
    def seal(cls, suite_id: str, generation: int, episodes, key: bytes) -> SealedReplaySuite:
        eps = tuple(episodes)
        if not eps:
            raise ValueError("replay suite cannot be empty")
        if len(key) != 32:
            raise ValueError("AES-256-GCM requires 32-byte key")
        aad = canonical_bytes({
            "suite_id": suite_id,
            "generation": generation,
            "algorithm": cls.ALGORITHM,
            "episode_count": len(eps),
        })
        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(nonce, cls._encode(eps), aad)
        return SealedReplaySuite(
            suite_id,
            generation,
            cls.ALGORITHM,
            base64.b64encode(nonce).decode(),
            base64.b64encode(ct).decode(),
            sha256_bytes(aad),
            len(eps),
        )

    @classmethod
    def open(cls, sealed: SealedReplaySuite, key: bytes) -> tuple[GroundedReplayEpisode, ...]:
        if sealed.algorithm != cls.ALGORITHM:
            raise ValueError("unsupported replay-vault algorithm")
        aad = canonical_bytes({
            "suite_id": sealed.suite_id,
            "generation": sealed.generation,
            "algorithm": sealed.algorithm,
            "episode_count": sealed.episode_count,
        })
        if sha256_bytes(aad) != sealed.aad_digest:
            raise ValueError("replay-suite AAD binding mismatch")
        pt = AESGCM(key).decrypt(base64.b64decode(sealed.nonce_b64), base64.b64decode(sealed.ciphertext_b64), aad)
        eps = cls._decode(pt)
        if len(eps) != sealed.episode_count:
            raise ValueError("replay episode-count mismatch")
        return eps
