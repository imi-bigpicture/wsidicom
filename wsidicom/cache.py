from collections.abc import Iterator
from threading import Lock
from typing import Callable, Dict, Generic, Iterable, Optional, Sequence, Tuple, TypeVar

from PIL.Image import Image

CacheKeyType = TypeVar("CacheKeyType")
CacheItemType = TypeVar("CacheItemType")


class LRU(Generic[CacheKeyType, CacheItemType]):
    def __init__(self, maxsize: int):
        self._lock = Lock()
        self._cache: Dict[CacheKeyType, CacheItemType] = {}
        self._maxsize = maxsize

    def get(self, key: CacheKeyType) -> Optional[CacheItemType]:
        with self._lock:
            item = self._cache.pop(key, None)
            if item is not None:
                self._cache[key] = item
            return item

    def put(self, key: CacheKeyType, value: CacheItemType) -> None:
        with self._lock:
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.pop(next(iter(self._cache)))

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class FrameCache(Generic[CacheItemType]):
    def __init__(self, size: int):
        self._lru_cache = LRU[Tuple[int, int], CacheItemType](size)

    def get_tile_frame(
        self,
        image_data_id: int,
        frame_index: int,
        frame_getter: Callable[[int], CacheItemType],
    ) -> CacheItemType:
        frame = self._lru_cache.get((image_data_id, frame_index))
        if frame is None:
            frame = frame_getter(frame_index)
            self._lru_cache.put((image_data_id, frame_index), frame)
        return frame

    def get_tile_frames(
        self,
        image_data_id: int,
        frame_indices: Sequence[int],
        frames_getter: Callable[[Iterable[int]], Iterator[CacheItemType]],
    ) -> Iterator[CacheItemType]:
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
                self._lru_cache.put((image_data_id, frame_index), frame)
            yield frame

    def clear(self) -> None:
        self._lru_cache.clear()


EncodedFrameCache = FrameCache[bytes]
DecodedFrameCache = FrameCache[Image]
