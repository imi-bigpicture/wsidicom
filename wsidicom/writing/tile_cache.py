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

"""Two-tier cache for out-of-order tiles during sequential writing."""

import threading
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable

from upath import UPath


class TileCache(metaclass=ABCMeta):
    """Tile-byte cache keyed by pyramid level and linear tile index.

    Supports storing and draining runs of consecutive tiles for use
    when output tiles arrive out of order and need to be held until
    their position is ready to write.
    """

    @abstractmethod
    def save_sequential(
        self, level: int, start_index: int, tiles: Iterable[bytes]
    ) -> None:
        """Save consecutive tiles starting at `start_index` for `level`.

        Parameters
        ----------
        level: int
            Pyramid level index.
        start_index: int
            Linear index of the first tile in the run.
        tiles: Iterable[bytes]
            Encoded tiles, in order from `start_index` onwards.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_sequential(self, level: int, index: int) -> Iterable[bytes]:
        """Load and remove consecutive tiles starting at `index` for `level`.

        Stops at the first missing index — the returned iterable contains
        only the contiguous run found in the cache.

        Parameters
        ----------
        level: int
            Pyramid level index.
        index: int
            Linear index to start loading from.

        Returns
        -------
        Iterable[bytes]
            The contiguous run of cached tiles starting at `index`.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self, level: int) -> None:
        """Remove all cached tiles for `level`."""
        raise NotImplementedError()


class FileTileStore:
    """Disk-based cache for tiles."""

    def __init__(self, cache_dir: UPath):
        """Create a file cache for tiles.

        The cache directory is created lazily on first save.

        Parameters
        ----------
        cache_dir: UPath
            Directory to store cached tiles.
        """
        self._cache_dir = cache_dir
        self._dir_created = False

        # Maps (level, index) to the file path of the cached tile.
        self._paths: dict[tuple[int, int], UPath] = {}

    def _ensure_dir(self) -> None:
        """Create the cache directory if it doesn't exist yet."""
        if self._dir_created:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dir_created = True

    def save(self, level: int, index: int, tile: bytes) -> None:
        """Save tile to disk cache as raw bytes."""
        self._ensure_dir()
        level_dir = self._cache_dir / str(level)
        level_dir.mkdir(exist_ok=True)
        filepath = level_dir / f"{index}.bin"
        filepath.write_bytes(tile)
        self._paths[(level, index)] = filepath

    def load(self, level: int, index: int) -> bytes | None:
        """Load and delete tile from disk cache."""
        filepath = self._paths.pop((level, index), None)
        if filepath is None:
            return None
        tile = filepath.read_bytes()
        filepath.unlink()
        return tile

    def has(self, level: int, index: int) -> bool:
        """Check if tile exists in disk cache."""
        return (level, index) in self._paths

    def clear_level(self, level: int) -> None:
        """Delete all cached tile files for a level."""
        level_file_keys = [key for key in self._paths if key[0] == level]
        for key in level_file_keys:
            filepath = self._paths.pop(key)
            filepath.unlink(missing_ok=True)

    def clear(self) -> None:
        """Delete all cached tile files."""
        for filepath in list(self._paths.values()):
            filepath.unlink(missing_ok=True)
        self._paths.clear()


class ByteBudgetMemoryTileStore:
    """In-memory cache for tiles with byte-budget capacity.

    Thread-safe: the shared budget is protected by a lock.
    """

    def __init__(self, budget_bytes: int):
        """Create a byte-budget memory cache.

        Parameters
        ----------
        budget_bytes: int
            Maximum number of bytes to store in memory.
        """
        self._budget_bytes = budget_bytes
        self._current_bytes = 0
        self._lock = threading.Lock()

        # Maps (level, index) to tile bytes.
        self._cache: dict[tuple[int, int], bytes] = {}

    def try_save(self, level: int, index: int, tile: bytes) -> bool:
        """Try to save tile to memory cache.

        Returns False if adding the tile would exceed the budget.
        """
        with self._lock:
            if self._current_bytes + len(tile) > self._budget_bytes:
                return False
            self._cache[(level, index)] = tile
            self._current_bytes += len(tile)
            return True

    def load(self, level: int, index: int) -> bytes | None:
        """Load and remove tile from memory cache."""
        with self._lock:
            tile = self._cache.pop((level, index), None)
            if tile is not None:
                self._current_bytes -= len(tile)
            return tile

    def has(self, level: int, index: int) -> bool:
        """Check if tile exists in memory cache."""
        with self._lock:
            return (level, index) in self._cache

    def clear_level(self, level: int) -> None:
        """Clear all tiles for a level from memory cache."""
        with self._lock:
            keys = [k for k in self._cache if k[0] == level]
            for key in keys:
                tile = self._cache.pop(key)
                self._current_bytes -= len(tile)

    @property
    def has_tiles(self) -> bool:
        """Whether the cache contains any tiles."""
        return len(self._cache) > 0

    def clear(self) -> None:
        """Clear all tiles from memory cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0


class DictTileCache(TileCache):
    """Simple dict-based tile cache for near-raster-order tile arrival.

    Uses a plain dict keyed by (level, index). Thread-safe via lock.
    """

    def __init__(self) -> None:
        # Maps (level, index) to tile bytes.
        self._cache: dict[tuple[int, int], bytes] = {}
        self._lock = threading.Lock()

    def save_sequential(
        self, level: int, start_index: int, tiles: Iterable[bytes]
    ) -> None:
        with self._lock:
            count = 0
            for i, tile in enumerate(tiles):
                self._cache[(level, start_index + i)] = tile
                count += 1

    def load_sequential(self, level: int, index: int) -> Iterable[bytes]:
        result: list[bytes] = []
        with self._lock:
            while (level, index) in self._cache:
                result.append(self._cache.pop((level, index)))
                index += 1
        return result

    def clear(self, level: int) -> None:
        with self._lock:
            keys = [k for k in self._cache if k[0] == level]
            for key in keys:
                del self._cache[key]


class ByteBudgetTileCache(TileCache):
    """Two-tier cache with byte-budget memory and disk overflow.

    Tiles are stored in memory until the budget is exceeded, then overflow
    to disk. The memory budget is shared across all levels.
    """

    def __init__(
        self,
        cache_dir: UPath,
        memory_budget_bytes: int,
    ):
        """Create a byte-budget two-tier tile cache.

        Parameters
        ----------
        cache_dir: UPath
            Directory for disk cache.
        memory_budget_bytes: int
            Maximum bytes to keep in memory. Tiles that would exceed this
            budget are stored on disk.
        """
        self._memory_store = ByteBudgetMemoryTileStore(budget_bytes=memory_budget_bytes)
        self._file_store = FileTileStore(cache_dir)
        self._lock = threading.Lock()

    def save_sequential(
        self, level: int, start_index: int, tiles: Iterable[bytes]
    ) -> None:
        count = 0
        for i, tile in enumerate(tiles):
            tile_index = start_index + i
            if not self._memory_store.try_save(level, tile_index, tile):
                self._file_store.save(level, tile_index, tile)
            count += 1

    def load_sequential(self, level: int, index: int) -> Iterable[bytes]:
        tile = self._load(level, index)
        while tile is not None:
            yield tile
            index += 1
            tile = self._load(level, index)

    def clear(self, level: int) -> None:
        self._memory_store.clear_level(level)
        self._file_store.clear_level(level)

    def _load(self, level: int, index: int) -> bytes | None:
        """Load and remove tile from cache, checking memory first then disk."""
        tile = self._memory_store.load(level, index)
        if tile is not None:
            return tile
        return self._file_store.load(level, index)
