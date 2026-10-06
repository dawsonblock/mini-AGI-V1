from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping
from .util import canonical_json, sha256_bytes

@dataclass(frozen=True)
class EffectiveModelGeneration:
    base_weights: str
    adapters: tuple[tuple[str,str,float], ...]
    tokenizer: str
    chat_template: str
    rope_config: str
    layer_layout: str
    execution_config: str
    cache_abi_version: int = 1

    @classmethod
    def build(cls, *, base_weights: str, adapters: Iterable[tuple[str,str,float]]=(),
              tokenizer: str, chat_template: str='', rope_config: str='',
              layer_layout: str='', execution_config: str='', cache_abi_version: int=1):
        aa=tuple(sorted((str(a),str(h),float(s)) for a,h,s in adapters))
        return cls(str(base_weights), aa, str(tokenizer), str(chat_template),
                   str(rope_config), str(layer_layout), str(execution_config), int(cache_abi_version))

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode())

@dataclass(frozen=True)
class CacheCompatibility:
    generation_digest: str
    cache_abi_version: int
    storage_dtype: str
    transition_orientation: str

    def assert_compatible(self, generation: EffectiveModelGeneration, *,
                          storage_dtype: str | None=None, orientation: str | None=None) -> None:
        if self.generation_digest != generation.digest:
            raise ValueError('neural cache compiled for a different effective model generation')
        if self.cache_abi_version != generation.cache_abi_version:
            raise ValueError('cache ABI mismatch')
        if storage_dtype is not None and self.storage_dtype != storage_dtype:
            raise ValueError('cache dtype mismatch')
        if orientation is not None and self.transition_orientation != orientation:
            raise ValueError('transition orientation mismatch')
