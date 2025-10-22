from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class _CacheEntry(Generic[V]):
    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """
    Simple in-memory TTL cache with LRU eviction.

    - O(1) get/set
    - Evicts least-recently-used when capacity exceeded
    - Skips expired entries on access
    """

    def __init__(self, max_size: int, default_ttl_seconds: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")

        self._store: OrderedDict[K, _CacheEntry[V]] = OrderedDict()
        self._max_size = max_size
        self._ttl = float(default_ttl_seconds)

    def _now(self) -> float:
        return time.monotonic()

    def _purge_expired(self) -> None:
        now = self._now()
        keys_to_delete: list[K] = []
        for key, entry in self._store.items():
            if entry.expires_at <= now:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            self._store.pop(key, None)

    def get(self, key: K) -> V | None:
        self._purge_expired()
        entry = self._store.get(key)
        if not entry:
            return None
        # move to end to mark as recently used
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: K, value: V, ttl_seconds: int | None = None) -> None:
        self._purge_expired()
        ttl = float(ttl_seconds) if ttl_seconds else self._ttl
        self._store[key] = _CacheEntry(value=value, expires_at=self._now() + ttl)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            # popitem(last=False) pops the least-recently-used
            self._store.popitem(last=False)

    def get_or_set(self, key: K, factory: Callable[[], V], ttl_seconds: int | None = None) -> V:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl_seconds=ttl_seconds)
        return value
