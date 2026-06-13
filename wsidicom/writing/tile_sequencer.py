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

"""Tile sequencer that reorders tiles for sequential writing."""

from collections.abc import Iterable, Sequence
from threading import Thread
from typing import Protocol

from wsidicom.geometry import Size
from wsidicom.thread import (
    CancellationToken,
    Cancelled,
    ReadOnlyQueue,
    ShutdownSentinel,
)
from wsidicom.writing.models import (
    EncodingTaskResult,
    PyramidTilePosition,
)
from wsidicom.writing.tile_cache import TileCache


class TileWriter(Protocol):
    """Sink that accepts batches of encoded tile bytes."""

    def write_tiles(self, tiles: Iterable[bytes]) -> int:
        """Write a batch of encoded tiles; return the count written."""
        ...


class TileSequencer:
    """Reorders tiles for sequential writing.

    Reads tiles from an injected queue and forwards them left-to-right,
    top-to-bottom to a `TileWriter`, using a cache for early arrivals.
    Stops on a `ShutdownSentinel` read from the queue. Runs in a separate
    thread.

    Writing order (from slowest to fastest changing index):
    optical path -> focal plane (z) -> row (y) -> column (x)
    """

    def __init__(
        self,
        tile_writer: TileWriter,
        tile_cache: TileCache,
        level_index: int,
        tiled_size: Size,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        input_queue: ReadOnlyQueue[EncodingTaskResult | ShutdownSentinel],
        token: CancellationToken,
    ):
        """Create a tile sequencer.

        Parameters
        ----------
        tile_writer: TileWriter
            Underlying writer for actual tile writing.
        tile_cache: TileCache
            Cache for out-of-order tiles (injected dependency).
        level_index: int
            Pyramid level index, used as cache key.
        tiled_size: Size
            Size of tiled image (width and height in tiles).
        focal_planes: Sequence[float]
            Ordered list of focal plane z-coordinates.
        optical_paths: Sequence[str]
            Ordered list of optical path identifiers.
        input_queue: ReadOnlyQueue[Union[EncodingTaskResult, ShutdownSentinel]]
            Queue from which encoded tile batches are consumed. Reading
            stops when a `ShutdownSentinel` is read.
        token: CancellationToken
            Shared cancellation token for fail-fast teardown. The worker cancels
            it if it fails, and its read observes it so the worker stops instead
            of blocking when an upstream producer has died.
        """
        self._tile_writer = tile_writer
        self._tile_cache = tile_cache
        self._level_index = level_index
        self._tiled_size = tiled_size
        self._focal_planes = focal_planes
        self._optical_paths = optical_paths
        self._total_tiles = tiled_size.area * len(focal_planes) * len(optical_paths)
        self._written_tiles = 0
        self._worker_exception: BaseException | None = None
        self._token = token
        self._queue = input_queue
        self._worker_thread = Thread(target=self._writer_loop, daemon=False)

    def start(self) -> None:
        """Start the sequencer thread."""
        if self._worker_thread.is_alive():
            raise RuntimeError("Sequencer thread is already running")
        self._worker_thread.start()

    def finalize(self) -> None:
        """Signal that all tiles have been submitted and wait for completion.

        Raises
        ------
        RuntimeError
            If the worker thread failed or not all expected tiles were written.
        """
        self.shutdown(timeout=None)

        if self._worker_exception is not None:
            raise RuntimeError("Sequencer thread failed") from self._worker_exception

        # Another stage may have cancelled the shared token while this worker
        # exited cleanly; surface that cause rather than a misleading
        # tile-count mismatch below.
        self._token.raise_if_cancelled()

        if self._written_tiles != self._total_tiles:
            raise RuntimeError(
                f"Expected {self._total_tiles} tiles, but only wrote "
                f"{self._written_tiles} tiles"
            )

    def shutdown(self, timeout: float | None = 10.0) -> None:
        """Join the worker thread (if alive) and clear the level's cache.

        Does not verify tile count or emit a sentinel; the worker stops
        when it reads a `ShutdownSentinel` from the input queue.

        Parameters
        ----------
        timeout: Optional[float]
            Maximum time in seconds to wait for the sequencer thread.
            None waits indefinitely.
        """
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        self._tile_cache.clear(self._level_index)

    def _writer_loop(self) -> None:
        """Main loop running in worker thread.

        Stops on a `ShutdownSentinel` (graceful), when the token is cancelled
        by another stage (`Cancelled`), or on its own failure — which it
        records and uses to cancel the shared token.
        """
        try:
            written_tiles = 0
            batch = self._queue.get(self._token)
            while not isinstance(batch, ShutdownSentinel):
                written_tiles = self._process_batch(batch, written_tiles)
                batch = self._queue.get(self._token)
            self._written_tiles = written_tiles
        except Cancelled:
            return
        except Exception as e:
            self._worker_exception = e
            self._token.cancel(e)

    def _process_batch(self, batch: EncodingTaskResult, written_tiles: int) -> int:
        """Process a batch of tiles.

        If the batch's first tile is the next one due in write order, writes the
        batch directly and then flushes any cached tiles that now follow on
        sequentially. Otherwise the batch has arrived early and is cached until
        the preceding tiles have been written.

        Parameters
        ----------
        batch: EncodingTaskResult
            Batch of encoded sequential tiles to process.
        written_tiles: int
            Number of tiles written so far.

        Returns
        -------
        int
            Updated number of tiles written.
        """
        batch_start_index = self._to_linear_index(batch.coordinates)
        if batch_start_index == written_tiles:
            written_tiles += self._tile_writer.write_tiles(batch.tiles)
            if written_tiles < self._total_tiles:
                consecutive_tiles = self._tile_cache.load_sequential(
                    self._level_index, written_tiles
                )
                written_tiles += self._tile_writer.write_tiles(consecutive_tiles)
        else:
            self._tile_cache.save_sequential(
                self._level_index, batch_start_index, batch.tiles
            )
        return written_tiles

    def _to_linear_index(self, coordinates: PyramidTilePosition) -> int:
        """Convert tile coordinates to a linear index within this level."""
        tiles_per_plane = self._tiled_size.area
        tiles_per_path = tiles_per_plane * len(self._focal_planes)
        return (
            coordinates.optical_path_index * tiles_per_path
            + coordinates.z_index * tiles_per_plane
            + coordinates.y_index * self._tiled_size.width
            + coordinates.x_index
        )
