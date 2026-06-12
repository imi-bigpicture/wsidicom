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

"""Instance writers: orchestrators that produce tiles for one DICOM instance
(a pyramid level or an ancillary group) and hand them to an injected `TileWriter`.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Semaphore
from typing import Iterable, List, Optional, Sequence, Union

from wsidicom.codec.encoder import Encoder
from wsidicom.geometry import Point, Size
from wsidicom.group import Instances
from wsidicom.instance import ImageData
from wsidicom.instance.dataset import WsiDataset
from wsidicom.thread import (
    CancellationToken,
    Cancelled,
    PriorityCancelableQueue,
    ReadOnlyQueue,
    ShutdownSentinel,
)
from wsidicom.writing.models import EncodingTaskResult
from wsidicom.writing.pyramid_tile_accumulator import PyramidTileAccumulator
from wsidicom.writing.tile_cache import TileCache
from wsidicom.writing.tile_readers import TileReader
from wsidicom.writing.tile_sequencer import TileSequencer, TileWriter


class GroupInstanceWriter:
    """Writes one group instance (label, overview, thumbnail) sequentially."""

    def __init__(
        self,
        dataset: WsiDataset,
        instances: Instances,
        encoder: Encoder,
        transcode: bool,
    ):
        self._dataset = dataset
        self._instances = instances
        self._encoder = encoder
        self._transcode = transcode

    @property
    def dataset(self) -> WsiDataset:
        return self._dataset

    def write(self, tile_writer: TileWriter) -> None:
        """Write all tiles to the given writer. Does not finalize the writer."""
        if self._transcode:
            tiles: Iterable[bytes] = (
                self._encoder.encode(decoded)
                for decoded in self._instances.iter_decoded_tiles()
            )
        else:
            tiles = self._instances.iter_encoded_tiles()
        tile_writer.write_tiles(tiles)


class PyramidLevelWriter:
    """Base orchestrator for one pyramid level."""

    def __init__(
        self,
        level_index: int,
        dataset: WsiDataset,
        tile_cache: TileCache,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        tiled_size: Size,
        tile_queue: ReadOnlyQueue[Union[EncodingTaskResult, ShutdownSentinel]],
        token: CancellationToken,
    ):
        """Create a pyramid-level orchestrator.

        Parameters
        ----------
        level_index: int
            Pyramid level index this writer handles.
        dataset: WsiDataset
            DICOM dataset describing this level's instance.
        tile_cache: TileCache
            Cache for out-of-order tile reordering inside the sequencer.
        focal_planes: Sequence[float]
            Ordered focal-plane z-coordinates for this level.
        optical_paths: Sequence[str]
            Ordered optical-path identifiers for this level.
        tiled_size: Size
            Tiled size (columns, rows) of this level.
        tile_queue: ReadOnlyQueue[Union[EncodingTaskResult, ShutdownSentinel]]
            Queue feeding the `TileSequencer`. The queue owner is
            responsible for emitting the `ShutdownSentinel`.
        token: CancellationToken
            Shared cancellation token for fail-fast teardown, passed to the
            `TileSequencer`.
        """
        self._level_index = level_index
        self._dataset = dataset
        self._tile_cache = tile_cache
        self._focal_planes = focal_planes
        self._optical_paths = optical_paths
        self._tiled_size = tiled_size
        self._tile_queue = tile_queue
        self._token = token
        self._tile_sequencer: Optional[TileSequencer] = None

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def dataset(self) -> WsiDataset:
        return self._dataset

    def start(self, tile_writer: TileWriter) -> None:
        """Create the tile sequencer with the given writer and start its thread."""
        self._tile_sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=self._tile_cache,
            level_index=self._level_index,
            tiled_size=self._tiled_size,
            focal_planes=self._focal_planes,
            optical_paths=self._optical_paths,
            input_queue=self._tile_queue,
            token=self._token,
        )
        self._tile_sequencer.start()

    def finalize_writers(self) -> None:
        """Finalize the tile sequencer pipeline."""
        assert self._tile_sequencer is not None
        self._tile_sequencer.finalize()

    def cleanup(self) -> None:
        """Best-effort pipeline shutdown on error."""
        if self._tile_sequencer is not None:
            try:
                self._tile_sequencer.shutdown()
            except Exception:
                pass


class SourcePyramidLevelWriter(PyramidLevelWriter):
    """Source level: reads tiles from source and dispatches them.

    Owns the tile queue feeding the sequencer and emits its shutdown
    sentinel after source reading completes.
    """

    def __init__(
        self,
        level_index: int,
        dataset: WsiDataset,
        tile_cache: TileCache,
        source_group: Instances,
        tiled_size: Size,
        tile_reader: TileReader,
        queue_maxsize: int = 100,
        chunk_size: Optional[int] = None,
        *,
        token: CancellationToken,
    ):
        self._owned_tile_queue: PriorityCancelableQueue[
            Union[EncodingTaskResult, ShutdownSentinel]
        ] = PriorityCancelableQueue(maxsize=queue_maxsize)
        super().__init__(
            level_index=level_index,
            dataset=dataset,
            tile_cache=tile_cache,
            focal_planes=source_group.focal_planes,
            optical_paths=source_group.optical_paths,
            tiled_size=tiled_size,
            tile_queue=self._owned_tile_queue,
            token=token,
        )
        self._source_group = source_group
        self._tile_reader = tile_reader
        self._chunk_size = chunk_size

    def finalize_writers(self) -> None:
        """Emit the shutdown sentinel, then finalize the sequencer.

        The emission is cancel-aware so it cannot block if the token cancels
        concurrently; the sequencer's `finalize` then surfaces the cause.
        """
        try:
            self._owned_tile_queue.put(ShutdownSentinel(), self._token)
        except Cancelled:
            pass
        super().finalize_writers()

    def run(self, pool: ThreadPoolExecutor, max_inflight: int) -> None:
        """Produce tiles by reading from source.

        Stops early if the token is cancelled (another stage failed); a batch
        that observes the cancelled token raises `Cancelled`, which unwinds the
        per-batch future and ends the run.
        """
        inflight = Semaphore(max_inflight)

        futures: List[Future[None]] = []
        for path_index, path in enumerate(self._source_group.optical_paths):
            for z_index, z in enumerate(self._source_group.focal_planes):
                image_data = self._source_group.image_data_map[(path, z)]
                for positions in self._iter_batches(
                    image_data,
                    self._tiled_size,
                ):
                    if self._token.is_cancelled():
                        return
                    inflight.acquire()
                    futures.append(
                        pool.submit(
                            self._process_batch,
                            image_data,
                            positions,
                            z,
                            z_index,
                            path,
                            path_index,
                            inflight,
                        )
                    )

        for future in futures:
            try:
                future.result()
            except Cancelled:
                return

    def _process_batch(
        self,
        image_data: ImageData,
        positions: List[Point],
        z: float,
        z_index: int,
        path: str,
        path_index: int,
        inflight: Semaphore,
    ) -> None:
        try:
            self._tile_reader.read_and_submit(
                image_data,
                positions,
                z,
                path,
                z_index,
                path_index,
                self._owned_tile_queue,
            )
        finally:
            inflight.release()

    def _iter_batches(
        self,
        image_data: ImageData,
        tiled_size: Size,
    ) -> Iterable[List[Point]]:
        """Generate tile position batches in Z-order or raster order."""
        height = tiled_size.height
        width = tiled_size.width
        chunk_width = max(
            (
                self._chunk_size
                if self._chunk_size is not None
                else image_data.suggested_minimum_chunk_size
            ),
            2,
        )
        if self._tile_reader.accumulator_chain_depth > 0:
            yield from self._iter_block_recursive(
                0,
                0,
                width,
                height,
                chunk_width,
                self._tile_reader.accumulator_chain_depth,
            )
        else:
            for y in range(height):
                for x in range(0, width, chunk_width):
                    end_x = min(x + chunk_width, width)
                    yield [Point(xi, y) for xi in range(x, end_x)]

    def _iter_block_recursive(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        chunk_width: int,
        depth: int,
    ) -> Iterable[List[Point]]:
        """Recursively yield batches in Z-order."""
        if start_x >= end_x or start_y >= end_y:
            return

        if depth <= 1:
            for y in range(start_y, end_y, 2):
                has_second_row = y + 1 < end_y
                for x in range(start_x, end_x, chunk_width):
                    batch_end_x = min(x + chunk_width, end_x)
                    positions: List[Point] = []
                    for xi in range(x, batch_end_x, 2):
                        xi_end = min(xi + 2, batch_end_x)
                        for yi in [y, y + 1] if has_second_row else [y]:
                            for xj in range(xi, xi_end):
                                positions.append(Point(xj, yi))
                    yield positions
        else:
            half = 2 ** (depth - 1)
            mid_x = min(start_x + half, end_x)
            mid_y = min(start_y + half, end_y)
            yield from self._iter_block_recursive(
                start_x, start_y, mid_x, mid_y, chunk_width, depth - 1
            )
            yield from self._iter_block_recursive(
                mid_x, start_y, end_x, mid_y, chunk_width, depth - 1
            )
            yield from self._iter_block_recursive(
                start_x, mid_y, mid_x, end_y, chunk_width, depth - 1
            )
            yield from self._iter_block_recursive(
                mid_x, mid_y, end_x, end_y, chunk_width, depth - 1
            )


class GeneratedPyramidLevelWriter(PyramidLevelWriter):
    """Pipeline orchestrator for a generated pyramid level.

    Composes an injected `PyramidTileAccumulator` and shares its output
    queue with the inherited `TileSequencer`.
    """

    def __init__(
        self,
        level_index: int,
        dataset: WsiDataset,
        tile_cache: TileCache,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        tiled_size: Size,
        accumulator: PyramidTileAccumulator,
        token: CancellationToken,
    ):
        super().__init__(
            level_index=level_index,
            dataset=dataset,
            tile_cache=tile_cache,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            tiled_size=tiled_size,
            tile_queue=accumulator.output_queue,
            token=token,
        )
        self._accumulator = accumulator

    @property
    def accumulator(self) -> PyramidTileAccumulator:
        return self._accumulator

    def start(self, tile_writer: TileWriter) -> None:
        super().start(tile_writer)
        self._accumulator.start()

    def shutdown(self) -> None:
        """Shut down the accumulator."""
        self._accumulator.shutdown()

    def cleanup(self) -> None:
        """Best-effort shutdown on error."""
        self._accumulator.cleanup()
        super().cleanup()
