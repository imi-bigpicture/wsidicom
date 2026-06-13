#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Tests for tile cache implementations."""

import tempfile
from collections.abc import Generator

import pytest
from upath import UPath

from wsidicom.writing.tile_cache import (
    ByteBudgetMemoryTileStore,
    ByteBudgetTileCache,
    DictTileCache,
    FileTileStore,
)


@pytest.fixture
def temp_cache_dir() -> Generator[UPath, None, None]:
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield UPath(tmpdir)


@pytest.fixture
def test_tile() -> bytes:
    """Create a test tile as encoded bytes."""
    return b"test_tile_data"


LEVEL = 0


@pytest.mark.unittest
class Test_FileTileStore:
    def test_save_and_load(self, temp_cache_dir, test_tile):
        """Test saving and loading from file cache."""
        # Arrange
        cache = FileTileStore(temp_cache_dir)
        index = 0

        # Act
        cache.save(LEVEL, index, test_tile)
        has_before_load = cache.has(LEVEL, index)
        loaded_tile = cache.load(LEVEL, index)
        has_after_load = cache.has(LEVEL, index)

        # Assert
        assert has_before_load
        assert loaded_tile is not None
        assert loaded_tile == test_tile
        assert not has_after_load

    def test_different_indices(self, temp_cache_dir, test_tile):
        """Test that file cache creates different files for different indices."""
        # Arrange
        cache = FileTileStore(temp_cache_dir)

        # Act
        cache.save(LEVEL, 0, test_tile)
        cache.save(LEVEL, 100, test_tile)
        cache.save(LEVEL, 200, test_tile)

        # Assert
        assert cache.has(LEVEL, 0)
        assert cache.has(LEVEL, 100)
        assert cache.has(LEVEL, 200)

    def test_load_nonexistent(self, temp_cache_dir):
        """Test loading a tile that doesn't exist."""
        # Arrange
        cache = FileTileStore(temp_cache_dir)

        # Act
        tile = cache.load(LEVEL, 99)

        # Assert
        assert tile is None

    def test_clear(self, temp_cache_dir, test_tile):
        """Test clearing all cached files."""
        # Arrange
        cache = FileTileStore(temp_cache_dir)
        cache.save(LEVEL, 0, test_tile)
        cache.save(LEVEL, 100, test_tile)

        # Act
        cache.clear()

        # Assert
        assert not cache.has(LEVEL, 0)
        assert not cache.has(LEVEL, 100)


@pytest.mark.unittest
class TestDictTileCache:
    def test_save_and_load_sequential(self, test_tile):
        """Test saving and loading tiles sequentially."""
        # Arrange
        cache = DictTileCache()
        tiles = [b"tile_0", b"tile_1", b"tile_2"]

        # Act
        cache.save_sequential(LEVEL, 0, tiles)
        loaded = list(cache.load_sequential(LEVEL, 0))

        # Assert
        assert loaded == tiles

    def test_out_of_order(self, test_tile):
        """Test saving out of order, load_sequential returns consecutive run."""
        # Arrange
        cache = DictTileCache()

        # Act - save tiles 2, 0, 1
        cache.save_sequential(LEVEL, 2, [b"tile_2"])
        cache.save_sequential(LEVEL, 0, [b"tile_0"])
        cache.save_sequential(LEVEL, 1, [b"tile_1"])
        loaded = list(cache.load_sequential(LEVEL, 0))

        # Assert - should get 0, 1, 2 consecutively
        assert loaded == [b"tile_0", b"tile_1", b"tile_2"]

    def test_load_stops_at_gap(self):
        """Test that load_sequential stops when next index is missing."""
        # Arrange
        cache = DictTileCache()
        cache.save_sequential(LEVEL, 0, [b"tile_0", b"tile_1"])
        cache.save_sequential(LEVEL, 3, [b"tile_3"])  # Gap at index 2

        # Act
        loaded = list(cache.load_sequential(LEVEL, 0))

        # Assert
        assert len(loaded) == 2
        assert loaded == [b"tile_0", b"tile_1"]

    def test_clear(self, test_tile):
        """Test that clear removes all tiles for a level."""
        # Arrange
        cache = DictTileCache()
        cache.save_sequential(LEVEL, 0, [test_tile, test_tile])
        assert (LEVEL, 0) in cache._cache  # positive control: the tiles are present

        # Act
        cache.clear(LEVEL)

        # Assert
        assert list(cache.load_sequential(LEVEL, 0)) == []

    def test_levels_isolated(self, test_tile):
        """Test that different levels don't interfere."""
        # Arrange
        cache = DictTileCache()
        cache.save_sequential(0, 0, [b"level0_tile0"])
        cache.save_sequential(1, 0, [b"level1_tile0"])

        # Act
        loaded_0 = list(cache.load_sequential(0, 0))
        loaded_1 = list(cache.load_sequential(1, 0))

        # Assert
        assert loaded_0 == [b"level0_tile0"]
        assert loaded_1 == [b"level1_tile0"]

    def test_clear_one_level(self, test_tile):
        """Test that clearing one level doesn't affect another."""
        # Arrange
        cache = DictTileCache()
        cache.save_sequential(0, 0, [b"level0"])
        cache.save_sequential(1, 0, [b"level1"])

        # Act
        cache.clear(0)

        # Assert
        assert list(cache.load_sequential(0, 0)) == []
        assert list(cache.load_sequential(1, 0)) == [b"level1"]


@pytest.mark.unittest
class Test_ByteBudgetMemoryTileStore:
    def test_save_and_load(self, test_tile):
        """Test saving and loading from byte-budget memory cache."""
        # Arrange
        cache = ByteBudgetMemoryTileStore(budget_bytes=1024)

        # Act
        saved = cache.try_save(LEVEL, 0, test_tile)
        has_before = cache.has(LEVEL, 0)
        loaded = cache.load(LEVEL, 0)
        has_after = cache.has(LEVEL, 0)

        # Assert
        assert saved
        assert has_before
        assert loaded == test_tile
        assert not has_after

    def test_budget_enforcement(self, test_tile):
        """Test that cache rejects tiles when budget would be exceeded."""
        # Arrange - budget for exactly one tile
        cache = ByteBudgetMemoryTileStore(budget_bytes=len(test_tile))

        # Act
        first_saved = cache.try_save(LEVEL, 0, test_tile)
        second_saved = cache.try_save(LEVEL, 1, test_tile)

        # Assert
        assert first_saved
        assert not second_saved
        assert cache.has(LEVEL, 0)
        assert not cache.has(LEVEL, 1)

    def test_budget_after_load_frees_space(self, test_tile):
        """Test that loading a tile frees budget for new tiles."""
        # Arrange - budget for exactly one tile
        cache = ByteBudgetMemoryTileStore(budget_bytes=len(test_tile))
        cache.try_save(LEVEL, 0, test_tile)

        # Act - load tile 0 (frees space), then save tile 1
        cache.load(LEVEL, 0)
        saved = cache.try_save(LEVEL, 1, test_tile)

        # Assert
        assert saved
        assert cache.has(LEVEL, 1)

    def test_clear(self, test_tile):
        """Test clearing the cache."""
        # Arrange
        cache = ByteBudgetMemoryTileStore(budget_bytes=1024)
        cache.try_save(LEVEL, 0, test_tile)
        cache.try_save(LEVEL, 1, test_tile)

        # Act
        cache.clear()

        # Assert
        assert not cache.has(LEVEL, 0)
        assert not cache.has(LEVEL, 1)
        assert not cache.has_tiles

    def test_empty_budget_rejects_all(self, test_tile):
        """Test that zero budget rejects all tiles."""
        # Arrange
        cache = ByteBudgetMemoryTileStore(budget_bytes=0)

        # Act
        saved = cache.try_save(LEVEL, 0, test_tile)

        # Assert
        assert not saved

    def test_has_tiles(self, test_tile):
        """Test has_tiles property."""
        # Arrange
        cache = ByteBudgetMemoryTileStore(budget_bytes=1024)

        # Assert - empty
        assert not cache.has_tiles

        # Act
        cache.try_save(LEVEL, 0, test_tile)

        # Assert - has tile
        assert cache.has_tiles

    def test_shared_budget_across_levels(self, test_tile):
        """Test that budget is shared across levels."""
        # Arrange - budget for exactly one tile
        cache = ByteBudgetMemoryTileStore(budget_bytes=len(test_tile))

        # Act - save on level 0 fills budget
        cache.try_save(0, 0, test_tile)
        saved = cache.try_save(1, 0, test_tile)

        # Assert - level 1 can't save
        assert not saved


@pytest.mark.unittest
class TestByteBudgetTileCache:
    def test_save_to_memory_within_budget(self, temp_cache_dir, test_tile):
        """Test that tiles within budget go to memory."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=1024,
        )

        # Act
        cache.save_sequential(LEVEL, 0, [test_tile])

        # Assert
        assert cache._memory_store.has(LEVEL, 0)
        assert not cache._file_store.has(LEVEL, 0)

    def test_overflow_to_disk(self, temp_cache_dir, test_tile):
        """Test that tiles exceeding budget overflow to disk."""
        # Arrange - budget for exactly one tile
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=len(test_tile),
        )

        # Act - first tile fits, second overflows
        cache.save_sequential(LEVEL, 0, [test_tile, test_tile])

        # Assert
        assert cache._memory_store.has(LEVEL, 0)
        assert cache._file_store.has(LEVEL, 1)

    def test_load_sequential(self, temp_cache_dir, test_tile):
        """Test loading sequential tiles from mixed cache."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=len(test_tile),
        )
        # Tile 0 in memory, tile 1 on disk
        cache.save_sequential(LEVEL, 0, [test_tile, test_tile])

        # Act
        tiles = list(cache.load_sequential(LEVEL, 0))

        # Assert
        assert len(tiles) == 2
        assert tiles[0] == test_tile
        assert tiles[1] == test_tile

    def test_load_sequential_stops_at_gap(self, temp_cache_dir, test_tile):
        """Test that load_sequential stops at a gap."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=1024,
        )
        cache.save_sequential(LEVEL, 0, [b"tile_0", b"tile_1"])
        cache.save_sequential(LEVEL, 3, [b"tile_3"])  # Gap at index 2

        # Act
        tiles = list(cache.load_sequential(LEVEL, 0))

        # Assert
        assert len(tiles) == 2

    def test_clear(self, temp_cache_dir, test_tile):
        """Test clearing a level's cache."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=len(test_tile),
        )
        cache.save_sequential(LEVEL, 0, [test_tile, test_tile])
        # Positive control: tile 0 in memory, tile 1 overflowed to disk
        assert cache._memory_store.has(LEVEL, 0)
        assert cache._file_store.has(LEVEL, 1)

        # Act
        cache.clear(LEVEL)

        # Assert
        assert not cache._memory_store.has(LEVEL, 0)
        assert not cache._file_store.has(LEVEL, 1)
        assert list(cache.load_sequential(LEVEL, 0)) == []

    def test_zero_budget_all_to_disk(self, temp_cache_dir, test_tile):
        """Test that zero budget sends everything to disk."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=0,
        )

        # Act
        cache.save_sequential(LEVEL, 0, [test_tile])

        # Assert
        assert not cache._memory_store.has(LEVEL, 0)
        assert cache._file_store.has(LEVEL, 0)

    def test_large_budget_all_in_memory(self, temp_cache_dir, test_tile):
        """Test that large budget keeps everything in memory."""
        # Arrange
        cache = ByteBudgetTileCache(
            cache_dir=temp_cache_dir,
            memory_budget_bytes=1024 * 1024,
        )
        tiles = [test_tile] * 10

        # Act
        cache.save_sequential(LEVEL, 0, tiles)

        # Assert
        for i in range(10):
            assert cache._memory_store.has(LEVEL, i)
            assert not cache._file_store.has(LEVEL, i)
