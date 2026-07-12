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

"""Stitcher for assembling a grid of tiles into one image."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import as_completed
from itertools import chain
from threading import Event
from typing import (
    Literal,
)

from PIL import Image

from wsidicom.geometry import Point, Region, Size
from wsidicom.thread import ReadExecutor


class Stitcher(ABC):
    """Abstract interface for stitching a row-major grid of tiles."""

    @abstractmethod
    def stitch(
        self,
        tiles: Iterable[Image.Image],
        grid_size: Size,
        fill: int | tuple[int, int, int] | None = None,
    ) -> Image.Image:
        """Stitch tiles laid out as a row-major grid into one image.

        The canvas dimensions are taken from the first tile's size and mode,
        and equal ``grid_size * tile_size`` pixels. Tiles are pasted in
        row-major order: first row left-to-right, then second row, and so on.
        A tile smaller than the first tile is pasted at its slot's top-left
        and the remainder of the slot keeps the canvas fill.

        Parameters
        ----------
        tiles: Iterable[Image.Image]
            Row-major iterable of tiles. May be a generator; only consumed
            once.
        grid_size: Size
            Grid dimensions in tile units (columns, rows).
        fill: Optional[Union[int, Tuple[int, int, int]]]
            Optional background colour for the canvas, used where tiles
            don't fully cover the grid. ``None`` leaves the canvas
            uninitialised.

        Returns
        -------
        Image.Image
            Stitched image.
        """
        raise NotImplementedError()

    @abstractmethod
    def stitch_parallel(
        self,
        tile_region: Region,
        fetch: Callable[[Sequence[Point]], Iterator[Image.Image]],
        tile_size: Size,
        mode: Literal["L", "I", "RGB"],
        canvas_grid_size: Size | None = None,
        fill: int | tuple[int, int, int] | None = None,
        *,
        executor: ReadExecutor,
    ) -> Image.Image:
        """Stitch with internal parallelism; workers fetch and paste.

        The tile region is split into chunks and dispatched across worker
        threads. Each worker calls `fetch` once for its chunk and pastes
        each returned tile directly into the canvas — paste runs on worker
        threads and overlaps with other workers' fetch/decode work.

        Tile size and mode are explicit parameters (not inferred from the
        first tile as in `stitch`) because the canvas must be allocated
        before any worker runs.

        Parameters
        ----------
        tile_region: Region
            Tile coordinates to fetch. `tile_region.start` is the tile
            coordinate that maps to canvas position (0, 0); the region is
            partitioned with `tile_region.chunked_iterate_all(threads)`.
        fetch: Callable[[Sequence[Point]], Iterator[Image.Image]]
            Called once per chunk. Must yield exactly one decoded tile per
            input point in the same order; a mismatched count raises
            ``ValueError``. Returned tiles may be smaller than `tile_size`;
            smaller tiles are pasted at the slot's top-left and the
            remainder of the slot keeps the canvas fill.
        tile_size: Size
            Pixel size per tile.
        mode: Literal["L", "I", "RGB"]
            Pillow image mode for the canvas.
        canvas_grid_size: Optional[Size]
            Canvas size in tile units. Defaults to `tile_region.size`.
            Pass a larger value to embed the fetched region inside a
            bigger canvas; uncovered area shows the fill colour. Must be
            at least as large as `tile_region.size` in both dimensions;
            otherwise raises ``ValueError``.
        fill: Optional[Union[int, Tuple[int, int, int]]]
            Optional background colour. ``None`` leaves the canvas
            uninitialised.
        executor: ReadExecutor
            Executor the fetch-and-paste chunks are submitted to; its
            ``workers`` count sets how many chunks the region is split into.

        Returns
        -------
        Image.Image
            Stitched image.
        """
        raise NotImplementedError()

    def stitch_grid(
        self,
        tiles: Sequence[Sequence[Image.Image]],
        fill: int | tuple[int, int, int] | None = None,
    ) -> Image.Image:
        """Stitch tiles given as a row-major 2D sequence.

        Convenience over `stitch` for callers that already hold a
        rectangular grid in memory; the grid size is inferred from the
        outer and inner sequence lengths. All rows must have the same
        length.

        Parameters
        ----------
        tiles: Sequence[Sequence[Image.Image]]
            Row-major grid of tiles. Outer sequence is rows, inner is
            columns within a row.
        fill: Optional[Union[int, Tuple[int, int, int]]]
            See `stitch`.

        Returns
        -------
        Image.Image
            Stitched image.
        """
        if not tiles or not tiles[0]:
            raise ValueError("Cannot stitch an empty tile grid")
        columns = len(tiles[0])
        if any(len(row) != columns for row in tiles):
            raise ValueError("All rows must have the same length")
        flat = (tile for row in tiles for tile in row)
        return self.stitch(flat, Size(columns, len(tiles)), fill)


class PillowStitcher(Stitcher):
    """Stitcher using Pillow's Image.new and paste."""

    def stitch(
        self,
        tiles: Iterable[Image.Image],
        grid_size: Size,
        fill: int | tuple[int, int, int] | None = None,
    ) -> Image.Image:
        tile_iterator = iter(tiles)
        try:
            first = next(tile_iterator)
        except StopIteration:
            raise ValueError("Cannot stitch an empty tile iterable") from None

        tile_size = Size(first.width, first.height)
        canvas = self._create_canvas(first.mode, grid_size, tile_size, fill)

        for index, tile in enumerate(chain([first], tile_iterator)):
            row, column = divmod(index, grid_size.width)
            canvas.paste(tile, (column * tile_size.width, row * tile_size.height))
        return canvas

    def stitch_parallel(
        self,
        tile_region: Region,
        fetch: Callable[[Sequence[Point]], Iterator[Image.Image]],
        tile_size: Size,
        mode: Literal["L", "I", "RGB"],
        canvas_grid_size: Size | None = None,
        fill: int | tuple[int, int, int] | None = None,
        *,
        executor: ReadExecutor,
    ) -> Image.Image:
        if canvas_grid_size is None:
            grid_size = tile_region.size
        else:
            if (
                canvas_grid_size.width < tile_region.size.width
                or canvas_grid_size.height < tile_region.size.height
            ):
                raise ValueError(
                    f"canvas_grid_size {canvas_grid_size} must be at least "
                    f"tile_region.size {tile_region.size}"
                )
            grid_size = canvas_grid_size

        canvas = self._create_canvas(mode, grid_size, tile_size, fill)
        grid_origin = tile_region.start

        # Workers set this to signal an error and stop other workers early.
        cancel = Event()

        def worker(chunk_points: Sequence[Point]) -> None:
            for point, tile in zip(chunk_points, fetch(chunk_points), strict=True):
                if cancel.is_set():
                    return
                canvas_x = (point.x - grid_origin.x) * tile_size.width
                canvas_y = (point.y - grid_origin.y) * tile_size.height
                canvas.paste(tile, (canvas_x, canvas_y))

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
        return canvas

    def _create_canvas(
        self,
        mode: str,
        grid_size: Size,
        tile_size: Size,
        fill: int | tuple[int, int, int] | None,
    ) -> Image.Image:
        canvas_size = Size(
            grid_size.width * tile_size.width,
            grid_size.height * tile_size.height,
        )
        return Image.new(mode, canvas_size.to_tuple(), color=fill)
