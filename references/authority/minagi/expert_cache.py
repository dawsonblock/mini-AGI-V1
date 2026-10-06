"""Two-level cache for logical experts.

v5.1 keeps expert identity in :class:`ExpertStore` and treats every faster tier
as disposable cache state:

    NVMe/files -> pinned host RAM -> device cache

Prefetch is digest-bound: a future is keyed by ``(uid, sha256)`` and therefore
cannot silently serve a newer expert version into an older graph epoch.  The
host tier is useful even on CPU-only machines because it eliminates repeated
NPZ decode/hash work during backward recomputation; on CUDA it optionally pins
tensors so non-blocking H2D copies can overlap with compute.
"""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import torch


class ExpertCache:
    def __init__(self, store, capacity: int = 16, device="cpu", dtype=None,
                 host_capacity: int = 256, prefetch_workers: int = 4,
                 pin_host: bool = True):
        self.store = store
        self.capacity = max(1, int(capacity))
        self.device = torch.device(device)
        self.dtype = dtype
        self.host_capacity = max(self.capacity, int(host_capacity or self.capacity))
        self.prefetch_workers = max(0, int(prefetch_workers or 0))
        self.pin_host = bool(pin_host) and self.device.type == "cuda" and torch.cuda.is_available()

        # Device entries and host entries are both keyed by immutable logical
        # version, never by a mutable cache slot.
        self.cache = OrderedDict()       # (uid,digest) -> device weight dict
        self.host = OrderedDict()        # (uid,digest) -> CPU weight dict
        self._futures: dict[tuple[int, str], Future] = {}
        self._executor = (ThreadPoolExecutor(max_workers=self.prefetch_workers,
                                             thread_name_prefix="minagi-expert-prefetch")
                          if self.prefetch_workers else None)
        self._lock = threading.Lock()

        self.hits = 0
        self.loads = 0
        self.evictions = 0
        self.host_hits = 0
        self.host_loads = 0
        self.host_evictions = 0
        self.prefetch_submitted = 0
        self.prefetch_hits = 0

    @staticmethod
    def _key(uid: int, digest: str):
        return (int(uid), str(digest))

    def _read_host(self, uid: int, digest: str):
        raw = self.store.load(uid, expected_sha256=digest)
        out = {}
        for k in ("w1", "w3", "w2"):
            t = raw[k].detach().cpu().contiguous()
            if self.pin_host:
                try:
                    t = t.pin_memory()
                except RuntimeError:
                    # Pinned memory can fail under constrained runtimes.  It is
                    # a performance feature, never a correctness requirement.
                    pass
            out[k] = t
        return out

    def _host_insert(self, key, weights):
        self.host[key] = weights
        self.host.move_to_end(key)
        self.host_loads += 1
        while len(self.host) > self.host_capacity:
            self.host.popitem(last=False)
            self.host_evictions += 1
        return weights

    def prefetch(self, items):
        """Begin loading ``(uid,digest)`` pairs into host RAM.

        Duplicate, already-resident and already-in-flight requests are cheap.
        No mutation is performed and every read verifies the caller-supplied
        SHA-256 before decoding the expert file.
        """
        if self._executor is None:
            return 0
        submitted = 0
        with self._lock:
            for uid, digest in items:
                key = self._key(uid, digest)
                if key in self.cache or key in self.host or key in self._futures:
                    self.prefetch_hits += 1
                    continue
                self._futures[key] = self._executor.submit(self._read_host, key[0], key[1])
                self.prefetch_submitted += 1
                submitted += 1
        return submitted

    def _get_host(self, uid: int, digest: str):
        key = self._key(uid, digest)
        if key in self.host:
            self.host_hits += 1
            self.host.move_to_end(key)
            return self.host[key]

        fut = None
        with self._lock:
            fut = self._futures.pop(key, None)
        if fut is not None:
            weights = fut.result()  # exceptions propagate; fail closed
            return self._host_insert(key, weights)

        weights = self._read_host(uid, digest)
        return self._host_insert(key, weights)

    def get(self, uid: int, digest: str):
        key = self._key(uid, digest)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

        raw = self._get_host(uid, digest)
        dt = self.dtype
        weights = {
            k: raw[k].to(device=self.device, dtype=(dt or raw[k].dtype),
                         non_blocking=self.pin_host)
            for k in ("w1", "w3", "w2")
        }
        self.cache[key] = weights
        self.cache.move_to_end(key)
        self.loads += 1
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1
        return weights

    def invalidate_uid(self, uid: int):
        uid = int(uid)
        for key in [k for k in self.cache if k[0] == uid]:
            self.cache.pop(key, None)
        for key in [k for k in self.host if k[0] == uid]:
            self.host.pop(key, None)
        with self._lock:
            for key in [k for k in self._futures if k[0] == uid]:
                fut = self._futures.pop(key)
                fut.cancel()

    def clear(self):
        self.cache.clear()
        self.host.clear()
        with self._lock:
            futures = list(self._futures.values())
            self._futures.clear()
        for fut in futures:
            fut.cancel()

    def close(self):
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def report(self):
        total = self.loads + self.hits
        htotal = self.host_loads + self.host_hits
        return {
            "held": len(self.cache), "capacity": self.capacity,
            "loads": self.loads, "hits": self.hits,
            "hit_rate": self.hits / max(total, 1),
            "evictions": self.evictions,
            "host_held": len(self.host), "host_capacity": self.host_capacity,
            "host_loads": self.host_loads, "host_hits": self.host_hits,
            "host_hit_rate": self.host_hits / max(htotal, 1),
            "host_evictions": self.host_evictions,
            "prefetch_workers": self.prefetch_workers,
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_hits": self.prefetch_hits,
            "pin_host": self.pin_host,
        }
