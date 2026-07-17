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

"""Stitcher for assembling a grid of tile pixels into one array."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import as_completed
from threading import Event

import numpy as np
from PIL import Image

from wsidicom.geometry import Point, Region, Size
from wsidicom.thread import ReadExecutor


class Stitcher(ABC):
    """Abstract interface for assembling a row-major grid of tile pixels
    into one array."""

    @abstractmethod
    def stitch_grid(self, tiles: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """Compose a row-major 2D grid of tile pixels into one array.

        Placement only (no resampling): tiles within a row are laid out along
        the width axis and rows along the height axis. Tile sizes may vary as
        long as the grid stays rectangular (equal row lengths, each column's
        tiles sharing a width and each row's a height), so both full and smaller
        edge tiles are supported.

        Parameters
        ----------
        tiles: Sequence[Sequence[np.ndarray]]
            Row-major grid of tile pixels. Outer sequence is rows, inner is columns.

        Returns
        -------
        np.ndarray
            The composed array.
        """
        raise NotImplementedError()

    @abstractmethod
    def stitch_parallel(
        self,
        region: Region,
        tile_region: Region,
        fetch: Callable[[Sequence[Point]], Iterator[np.ndarray]],
        tile_size: Size,
        dtype: np.dtype,
        samples_per_pixel: int,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Assemble the pixels of ``region`` from its covering tiles, with
        internal parallelism.

        Each tile is clipped to ``region`` and written into a canvas sized to
        ``region``, so tile pixels falling outside it are never copied and the
        result needs no further crop. ``tile_region`` must be the minimal set of
        whole tiles covering ``region``; every fetched tile therefore overlaps
        it. Dispatch is split into chunks across worker threads.

        Parameters
        ----------
        region: Region
            Pixel region to return, in the same coordinate space as the tile
            grid (a tile at point ``p`` occupies pixels ``p * tile_size``).
        tile_region: Region
            Tile coordinates to fetch, covering ``region``. Partitioned with
            ``chunked_iterate_all``.
        fetch: Callable[[Sequence[Point]], Iterator[np.ndarray]]
            Called once per chunk, yielding one full ``tile_size`` array per
            input point, in order. A mismatched count raises ``ValueError``.
        tile_size: Size
            Pixel size per tile.
        dtype: np.dtype
            Pixel dtype of the output canvas, matching the fetched tiles.
        samples_per_pixel: int
            Samples per pixel, matching the fetched tiles. ``1`` gives a 2D
            canvas; more gives a ``(..., samples_per_pixel)`` canvas.
        executor: ReadExecutor
            Executor the fetch-and-write chunks are submitted to; its
            ``workers`` count sets how many chunks the region is split into.

        Returns
        -------
        np.ndarray
            Contiguous array of the region, shape
            ``(region.size.height, region.size.width[, samples])``.
        """
        raise NotImplementedError()

    @staticmethod
    def _sample_shape(samples_per_pixel: int) -> tuple[int, ...]:
        """Trailing per-pixel canvas dimensions: ``()`` for a single sample,
        ``(samples_per_pixel,)`` for more."""
        return () if samples_per_pixel == 1 else (samples_per_pixel,)

    @staticmethod
    def _validate_grid(tiles: Sequence[Sequence[np.ndarray]]) -> int:
        """Validate a row-major grid and return its column count."""
        if not tiles or not tiles[0]:
            raise ValueError("Cannot stitch an empty tile grid")
        columns = len(tiles[0])
        if any(len(row) != columns for row in tiles):
            raise ValueError("All rows must have the same length")
        return columns

    @staticmethod
    def _run_workers(
        tile_region: Region,
        worker: Callable[[Sequence[Point]], None],
        cancel: Event,
        executor: ReadExecutor,
    ) -> None:
        """Dispatch ``worker`` over the region's chunks. On the first worker
        error, set ``cancel`` (so still-running workers can stop early) and
        re-raise once all have finished."""
        chunks = tile_region.chunked_iterate_all(executor.workers)
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        first_error: BaseException | None = None
        for future in as_completed(futures):
            exception = future.exception()
            if exception is not None and first_error is None:
                first_error = exception
                cancel.set()
        if first_error is not None:
            raise first_error


class NumpyStitcher(Stitcher):
    """Stitches tile pixels into one array by slice-write.

    The production stitcher: the slice-writes release the GIL, so workers
    compose concurrently.
    """

    def stitch_grid(self, tiles: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        self._validate_grid(tiles)
        rows = [np.concatenate(list(row), axis=1) for row in tiles]
        return np.concatenate(rows, axis=0)

    def stitch_parallel(
        self,
        region: Region,
        tile_region: Region,
        fetch: Callable[[Sequence[Point]], Iterator[np.ndarray]],
        tile_size: Size,
        dtype: np.dtype,
        samples_per_pixel: int,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        tile_height, tile_width = tile_size.height, tile_size.width
        canvas_shape = (
            region.size.height,
            region.size.width,
        ) + self._sample_shape(samples_per_pixel)
        canvas = np.empty(canvas_shape, dtype=dtype)
        crop_y, crop_x = region.start.y, region.start.x
        cancel = Event()

        def worker(chunk_points: Sequence[Point]) -> None:
            for point, tile in zip(chunk_points, fetch(chunk_points), strict=True):
                if cancel.is_set():
                    return
                # Tile's top-left in canvas coordinates; may be negative when the
                # tile starts left/above the region and is clipped.
                destination_y = point.y * tile_height - crop_y
                destination_x = point.x * tile_width - crop_x
                source_y = max(0, -destination_y)
                source_x = max(0, -destination_x)
                row_start = max(0, destination_y)
                column_start = max(0, destination_x)
                row_end = min(region.size.height, destination_y + tile_height)
                column_end = min(region.size.width, destination_x + tile_width)
                canvas[row_start:row_end, column_start:column_end] = tile[
                    source_y : source_y + (row_end - row_start),
                    source_x : source_x + (column_end - column_start),
                ]

        self._run_workers(tile_region, worker, cancel, executor)
        return canvas


class PillowStitcher(Stitcher):
    """Reference stitcher that composes tiles with Pillow paste.

    Produces the same array as the slice-write stitcher for non-overlapping
    tiles, but paste holds the GIL so it does not scale. Kept as an independent
    cross-check for tests, not for production use.
    """

    def stitch_grid(self, tiles: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        columns = self._validate_grid(tiles)
        column_widths = [tiles[0][column].shape[1] for column in range(columns)]
        row_heights = [row[0].shape[0] for row in tiles]
        canvas = self._new_canvas(tiles[0][0], sum(column_widths), sum(row_heights))
        y = 0
        for row, height in zip(tiles, row_heights, strict=True):
            x = 0
            for tile, width in zip(row, column_widths, strict=True):
                canvas.paste(Image.fromarray(tile), (x, y))
                x += width
            y += height
        return np.asarray(canvas)

    def stitch_parallel(
        self,
        region: Region,
        tile_region: Region,
        fetch: Callable[[Sequence[Point]], Iterator[np.ndarray]],
        tile_size: Size,
        dtype: np.dtype,
        samples_per_pixel: int,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        tile_height, tile_width = tile_size.height, tile_size.width
        sample = np.empty((1, 1) + self._sample_shape(samples_per_pixel), dtype=dtype)
        canvas = self._new_canvas(sample, region.size.width, region.size.height)
        crop_y, crop_x = region.start.y, region.start.x
        cancel = Event()

        def worker(chunk_points: Sequence[Point]) -> None:
            for point, tile in zip(chunk_points, fetch(chunk_points), strict=True):
                if cancel.is_set():
                    return
                destination_y = point.y * tile_height - crop_y
                destination_x = point.x * tile_width - crop_x
                source_y = max(0, -destination_y)
                source_x = max(0, -destination_x)
                row_start = max(0, destination_y)
                column_start = max(0, destination_x)
                row_end = min(region.size.height, destination_y + tile_height)
                column_end = min(region.size.width, destination_x + tile_width)
                clipped = tile[
                    source_y : source_y + (row_end - row_start),
                    source_x : source_x + (column_end - column_start),
                ]
                canvas.paste(Image.fromarray(clipped), (column_start, row_start))

        self._run_workers(tile_region, worker, cancel, executor)
        return np.asarray(canvas)

    @staticmethod
    def _new_canvas(sample: np.ndarray, width: int, height: int) -> Image.Image:
        return Image.new(Image.fromarray(sample).mode, (width, height))
