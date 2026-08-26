#    Copyright 2026 SECTRA AB
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

import pytest

from wsidicom.cache import LRU, CacheItem, EncodedFrameCache

ITEM_SIZE = 50


def item(value: bytes) -> CacheItem[bytes]:
    return CacheItem(value, len(value))


@pytest.mark.unittest
class TestLRU:
    def test_evicts_least_recently_used_when_over_maxsize(self):
        # Arrange
        cache = LRU[str, bytes](2 * ITEM_SIZE)
        cache.put("a", item(b"a" * ITEM_SIZE))
        cache.put("b", item(b"b" * ITEM_SIZE))

        # Act
        cache.put("c", item(b"c" * ITEM_SIZE))

        # Assert
        assert cache.get("a") is None
        assert cache.get("b") == b"b" * ITEM_SIZE
        assert cache.get("c") == b"c" * ITEM_SIZE

    def test_caches_again_after_clear(self):
        # Arrange
        cache = LRU[str, bytes](ITEM_SIZE)
        cache.put("a", item(b"a" * ITEM_SIZE))
        cache.clear()

        # Act
        cache.put("b", item(b"b" * ITEM_SIZE))

        # Assert
        assert cache.get("b") == b"b" * ITEM_SIZE

    def test_replacing_an_item_does_not_consume_extra_capacity(self):
        # Arrange
        cache = LRU[str, bytes](2 * ITEM_SIZE)
        cache.put("a", item(b"a" * ITEM_SIZE))
        cache.put("a", item(b"A" * ITEM_SIZE))

        # Act
        cache.put("b", item(b"b" * ITEM_SIZE))

        # Assert
        assert cache.get("a") == b"A" * ITEM_SIZE
        assert cache.get("b") == b"b" * ITEM_SIZE

    def test_replacing_an_item_makes_it_most_recently_used(self):
        # Arrange
        cache = LRU[str, bytes](2 * ITEM_SIZE)
        cache.put("a", item(b"a" * ITEM_SIZE))
        cache.put("b", item(b"b" * ITEM_SIZE))
        cache.put("a", item(b"A" * ITEM_SIZE))

        # Act
        cache.put("c", item(b"c" * ITEM_SIZE))

        # Assert
        assert cache.get("a") == b"A" * ITEM_SIZE
        assert cache.get("b") is None


@pytest.mark.unittest
class TestFrameCache:
    def test_caches_frames_again_after_clear(self):
        # Arrange
        FRAME = b"z" * 100
        FRAME_COUNT = 10
        cache = EncodedFrameCache(FRAME_COUNT * len(FRAME))
        fetched: list[int] = []

        def frame_getter(frame_index: int) -> bytes:
            fetched.append(frame_index)
            return FRAME

        for frame_index in range(FRAME_COUNT):
            cache.get_tile_frame(1, frame_index, frame_getter)
        cache.clear()
        fetched.clear()

        # Act
        for _ in range(3):
            for frame_index in range(FRAME_COUNT):
                cache.get_tile_frame(1, frame_index, frame_getter)

        # Assert
        assert len(fetched) == FRAME_COUNT
