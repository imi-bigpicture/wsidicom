#    Copyright 2024 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from threading import Condition, Lock
from typing import Generic, TypeVar

import numpy as np
from cachetools import LRUCache, cachedmethod


def lru_cached_method(
    maxsize: int | Callable[[], int | None] | None = 128,
):
    """Cache a method's return values in a per-instance LRU cache.

    Like ``functools.lru_cache`` but bound to the instance rather than the class,
    so the cache is released with the instance instead of being pinned on the
    class for the lifetime of the process. A per-instance ``threading.Condition``
    makes concurrent calls for the same key wait for the first to finish instead
    of recomputing, so the wrapped method runs at most once per key while calls
    for different keys still run concurrently.

    Parameters
    ----------
    maxsize:
        Maximum number of entries before least-recently-used eviction, or
        ``None`` for an unbounded cache. May be a callable returning the size,
        evaluated when an instance's cache is first created, so a size taken from
        runtime configuration is honoured.
    """

    def decorator(method: Callable):
        cache_attr = f"_{method.__name__}_cache"
        condition_attr = f"_{method.__name__}_condition"

        def get_cache(self):
            cache = self.__dict__.get(cache_attr)
            if cache is None:
                size = maxsize() if callable(maxsize) else maxsize
                default = {} if size is None else LRUCache(maxsize=size)
                cache = self.__dict__.setdefault(cache_attr, default)
            return cache

        def get_condition(self):
            condition = self.__dict__.get(condition_attr)
            if condition is None:
                condition = self.__dict__.setdefault(condition_attr, Condition())
            return condition

        return cachedmethod(get_cache, condition=get_condition)(method)

    return decorator


CacheKeyType = TypeVar("CacheKeyType")
CacheItemType = TypeVar("CacheItemType")


@dataclass
class CacheItem(Generic[CacheItemType]):
    value: CacheItemType
    size: int


class LRU(Generic[CacheKeyType, CacheItemType]):
    def __init__(self, maxsize: int):
        self._lock = Lock()
        self._cache: dict[CacheKeyType, CacheItem[CacheItemType]] = {}
        self._maxsize = maxsize
        self._size = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def get(self, key: CacheKeyType) -> CacheItemType | None:
        with self._lock:
            item = self._cache.pop(key, None)
            if item is not None:
                self._cache[key] = item
            return item.value if item is not None else None

    def put(self, key: CacheKeyType, item: CacheItem[CacheItemType]) -> None:
        with self._lock:
            self._cache[key] = item
            self._size += item.size
            while self._size > self._maxsize:
                self._size -= self._cache.pop(next(iter(self._cache))).size

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def resize(self, maxsize: int) -> None:
        with self._lock:
            self._maxsize = maxsize
            while self._size > maxsize:
                self._size -= self._cache.pop(next(iter(self._cache))).size


class FrameCache(Generic[CacheItemType]):
    """Cache of frames, keyed by image data and frame index."""

    def __init__(self, size: int):
        self._size = size
        self._lru_cache = LRU[tuple[int, int], CacheItemType](size)

    def get_tile_frame(
        self,
        image_data_id: int,
        frame_index: int,
        frame_getter: Callable[[int], CacheItemType],
    ) -> CacheItemType:
        if self._lru_cache.maxsize < 1:
            return frame_getter(frame_index)
        key = (image_data_id, frame_index)
        frame = self._lru_cache.get(key)
        if frame is None:
            frame = frame_getter(frame_index)
            self._lru_cache.put(key, self._to_cache_item(frame))
        return frame

    def get_tile_frames(
        self,
        image_data_id: int,
        frame_indices: Sequence[int],
        frames_getter: Callable[[Iterable[int]], Iterator[CacheItemType]],
    ) -> Iterator[CacheItemType]:
        if self._lru_cache.maxsize < 1:
            yield from frames_getter(frame_indices)
            return
        cached_frames = {
            frame_index: frame
            for (frame_index, frame) in (
                (frame_index, self._lru_cache.get((image_data_id, frame_index)))
                for frame_index in frame_indices
            )
            if frame is not None
        }
        fetched_frames = frames_getter(
            frame_index
            for frame_index in frame_indices
            if frame_index not in cached_frames
        )
        for frame_index in frame_indices:
            frame = cached_frames.get(frame_index)
            if frame is None:
                frame = next(fetched_frames)
                self._lru_cache.put(
                    (image_data_id, frame_index), self._to_cache_item(frame)
                )
            yield frame

    def clear(self) -> None:
        self._lru_cache.clear()

    def resize(self, size: int) -> None:
        self._lru_cache.resize(size)

    @classmethod
    @abstractmethod
    def _to_cache_item(cls, value: CacheItemType) -> CacheItem[CacheItemType]:
        pass


class EncodedFrameCache(FrameCache[bytes]):
    @classmethod
    def _to_cache_item(cls, value: bytes) -> CacheItem[bytes]:
        return CacheItem(value, len(value))


class DecodedFrameCache(FrameCache[np.ndarray]):
    """Caches decoded tiles as numpy arrays."""

    @classmethod
    def _to_cache_item(cls, value: np.ndarray) -> CacheItem[np.ndarray]:
        return CacheItem(value, value.nbytes)
