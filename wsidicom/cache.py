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
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Lock
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from PIL.Image import Image

CacheKeyType = TypeVar("CacheKeyType")
CacheItemType = TypeVar("CacheItemType")


@dataclass
class CacheItem(Generic[CacheItemType]):
    value: CacheItemType
    size: int


class LRU(Generic[CacheKeyType, CacheItemType]):
    def __init__(self, maxsize: int):
        self._lock = Lock()
        self._cache: Dict[CacheKeyType, CacheItem[CacheItemType]] = {}
        self._maxsize = maxsize
        self._size = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def get(self, key: CacheKeyType) -> Optional[CacheItemType]:
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
    def __init__(self, size: int):
        self._size = size
        self._lru_cache = LRU[Tuple[int, int], CacheItemType](size)

    def get_tile_frame(
        self,
        image_data_id: int,
        frame_index: int,
        frame_getter: Callable[[int], CacheItemType],
    ) -> CacheItemType:
        if self._lru_cache.maxsize < 1:
            return frame_getter(frame_index)
        frame = self._lru_cache.get((image_data_id, frame_index))
        if frame is None:
            frame = frame_getter(frame_index)
            self._lru_cache.put(
                (image_data_id, frame_index), self._to_cache_item(frame)
            )
        return frame

    def get_tile_frames(
        self,
        image_data_id: int,
        frame_indices: Sequence[int],
        frames_getter: Callable[[Iterable[int]], Iterator[CacheItemType]],
    ) -> Iterator[CacheItemType]:
        if self._lru_cache.maxsize < 1:
            return frames_getter(frame_indices)
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


class DecodedFrameCache(FrameCache[Image]):
    _mode_to_bytes = {
        "L": 1,
        "RGB": 3,
        "I": 4,
    }

    @classmethod
    def _to_cache_item(cls, value: Image) -> CacheItem[Image]:
        return CacheItem(
            value, value.size[0] * value.size[1] * cls._mode_to_bytes[value.mode]
        )
