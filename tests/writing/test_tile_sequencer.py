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

"""Tests for TileSequencer."""

from collections.abc import Iterable, Sequence

import pytest
from decoy import Decoy
from decoy.matchers import Anything
from upath import UPath

from wsidicom.file.io.wsidicom_writer import WsiDicomWriter
from wsidicom.geometry import Size
from wsidicom.thread import (
    CancellationToken,
    FifoCancelableQueue,
    PriorityCancelableQueue,
    ShutdownSentinel,
)
from wsidicom.writing.models import EncodingTaskResult, PyramidTilePosition
from wsidicom.writing.tile_cache import ByteBudgetTileCache, DictTileCache
from wsidicom.writing.tile_sequencer import TileSequencer


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.fixture
def input_queue() -> "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]":
    """Create the priority queue feeding the sequencer."""
    return PriorityCancelableQueue()


@pytest.fixture
def token() -> CancellationToken:
    """Create a cancellation token for queue operations."""
    return CancellationToken()


@pytest.fixture
def tile_writer(decoy: Decoy) -> WsiDicomWriter:
    """Create a mock tile writer."""
    return decoy.mock(cls=WsiDicomWriter)


@pytest.fixture
def tile_cache(decoy: Decoy) -> DictTileCache:
    """Create a mock tile cache."""
    cache = decoy.mock(cls=DictTileCache)
    return cache


def make_batch(
    x: int,
    y: int,
    z: int = 0,
    path: int = 0,
    level: int = 0,
    count: int = 1,
) -> EncodingTaskResult:
    """Helper to create EncodingTaskResult."""
    return EncodingTaskResult(
        coordinates=PyramidTilePosition(
            level=level,
            x_index=x,
            y_index=y,
            z_index=z,
            optical_path_index=path,
        ),
        tiles=[
            f"tile_data {x,y, z, path, level, index}".encode() for index in range(count)
        ],
    )


@pytest.mark.unittest
class TestTileSequencer:
    def test_sequential_writing(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test writing tiles in sequential order."""
        # Arrange
        written_data: list[bytes] = []

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        tiled_size = Size(2, 2)
        focal_planes = [0.0]
        optical_paths = ["0"]
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        for index in batches:
            decoy.when(tile_cache.load_sequential(Anything(), index)).then_return([])

        # Act
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]
        decoy.verify(
            tile_cache.save_sequential(Anything(), Anything(), Anything()), times=0
        )

    def test_out_of_order_writing(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test writing tiles out of order."""
        # Arrange
        written_data: list[bytes] = []
        cached_data: dict[int, bytes] = {}

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        def save_to_cache(level: int, index: int, tiles: list[bytes]) -> None:
            for i, tile in enumerate(tiles):
                cached_data[index + i] = tile

        def load_from_cache(level: int, index: int) -> list[bytes]:
            result = []
            while index in cached_data:
                result.append(cached_data.pop(index))
                index += 1
            return result

        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        decoy.when(
            tile_cache.save_sequential(Anything(), Anything(), Anything())
        ).then_do(save_to_cache)
        decoy.when(tile_cache.load_sequential(Anything(), Anything())).then_do(
            load_from_cache
        )

        tiled_size = Size(3, 3)
        focal_planes = [0.0]
        optical_paths = ["0"]
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for x in range(tiled_size.width - 1, -1, -1)
            for y in range(tiled_size.height - 1, -1, -1)
        }

        # Act
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]

    def test_incomplete_tiles_error(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test that finalizing with missing tiles raises error."""
        # Arrange
        written_data: list[bytes] = []

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        tiled_size = Size(2, 2)
        focal_planes = [0.0]
        optical_paths = ["0"]
        batches = {
            0 + 0 * tiled_size.width: make_batch(x=0, y=0),
            1 + 0 * tiled_size.width: make_batch(x=1, y=0),
            0 + 1 * tiled_size.width: make_batch(x=0, y=1),
            # Missing batch for (1, 1)
        }
        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        for index in batches:
            decoy.when(tile_cache.load_sequential(Anything(), index)).then_return([])
        decoy.when(
            tile_cache.load_sequential(Anything(), 1 + 1 * tiled_size.width)
        ).then_return([])

        # Act & Assert
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        with pytest.raises(RuntimeError, match="Expected 4 tiles, but only wrote 3"):
            sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]

    def test_multiple_z_planes(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test writing tiles with multiple z-planes."""
        # Arrange
        written_data: list[bytes] = []

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        tiled_size = Size(2, 2)
        focal_planes = [0.0, 5.0]
        optical_paths = ["0"]
        cached_data: dict[int, bytes] = {}

        def save_to_cache(level: int, index: int, tiles: list[bytes]) -> None:
            for i, tile in enumerate(tiles):
                cached_data[index + i] = tile

        def load_from_cache(level: int, index: int) -> list[bytes]:
            result = []
            while index in cached_data:
                result.append(cached_data.pop(index))
                index += 1
            return result

        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        decoy.when(
            tile_cache.save_sequential(Anything(), Anything(), Anything())
        ).then_do(save_to_cache)
        decoy.when(tile_cache.load_sequential(Anything(), Anything())).then_do(
            load_from_cache
        )

        batches = {
            (x + y * tiled_size.width + z_index * tiled_size.area): make_batch(
                x=x, y=y, z=z_index
            )
            for x in range(tiled_size.width)
            for y in range(tiled_size.height)
            for z_index in range(len(focal_planes))
        }

        # Act
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]

    def test_multiple_optical_paths(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test writing tiles with multiple optical paths."""
        # Arrange
        written_data: list[bytes] = []

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        cached_data: dict[int, bytes] = {}

        def save_to_cache(level: int, index: int, tiles: list[bytes]) -> None:
            for i, tile in enumerate(tiles):
                cached_data[index + i] = tile

        def load_from_cache(level: int, index: int) -> list[bytes]:
            result = []
            while index in cached_data:
                result.append(cached_data.pop(index))
                index += 1
            return result

        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        decoy.when(
            tile_cache.save_sequential(Anything(), Anything(), Anything())
        ).then_do(save_to_cache)
        decoy.when(tile_cache.load_sequential(Anything(), Anything())).then_do(
            load_from_cache
        )

        tiled_size = Size(2, 2)
        focal_planes = [0.0]
        optical_paths = ["0", "1"]
        batches = {
            x
            + y * tiled_size.width
            + path_index * tiled_size.area: make_batch(x=x, y=y, path=path_index)
            for x in range(tiled_size.width)
            for y in range(tiled_size.height)
            for path_index in range(len(optical_paths))
        }

        # Act
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]

    def test_submit_tiles_batch(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test submitting multiple tiles in a batch."""
        # Arrange
        written_data: list[bytes] = []

        def capture_write(tiles: Iterable[bytes]) -> int:
            tiles_list = list(tiles)
            written_data.extend(tiles_list)
            return len(tiles_list)

        tiled_size = Size(4, 4)
        focal_planes = [0.0]
        optical_paths = ["0"]
        batches = {
            y * tiled_size.width: make_batch(x=0, y=y, count=4)
            for y in range(0, tiled_size.height)
        }
        decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)
        for index in batches:
            decoy.when(tile_cache.load_sequential(Anything(), index)).then_return([])

        # Act
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        for batch in batches.values():
            input_queue.put(batch, token)
        input_queue.put(ShutdownSentinel(), token)
        sequencer.finalize()

        # Assert
        assert written_data == [
            tile for batch in sorted(batches.items()) for tile in batch[1].tiles
        ]

    def test_shutdown_clears_cache(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test that shutdown clears the level's cache."""
        # Arrange
        tiled_size = Size(2, 2)
        focal_planes = [0.0]
        optical_paths = ["0"]
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()

        # Act
        input_queue.put(ShutdownSentinel(), token)
        sequencer.shutdown()

        # Assert
        decoy.verify(tile_cache.clear(Anything()), times=1)

    def test_tile_writer_failure_propagated_and_cancels(
        self,
        decoy: Decoy,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """A tile_writer failure (the worker's own work) surfaces on finalize and
        cancels the shared token."""
        # Arrange
        tiled_size = Size(1, 1)
        focal_planes = [0.0]
        optical_paths = ["0"]
        decoy.when(tile_writer.write_tiles(Anything())).then_raise(
            RuntimeError("disk write failed")
        )

        # Act & Assert
        sequencer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        sequencer.start()
        input_queue.put(make_batch(x=0, y=0), token)
        input_queue.put(ShutdownSentinel(), token)
        with pytest.raises(RuntimeError, match="Sequencer thread failed"):
            sequencer.finalize()
        assert isinstance(token.exception, RuntimeError)

    def test_start_twice_error(
        self,
        tile_writer: WsiDicomWriter,
        tile_cache: DictTileCache,
        input_queue: "PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel]",
        token: CancellationToken,
    ):
        """Test that starting twice raises error."""
        # Arrange
        tiled_size = Size(0, 0)
        focal_planes = [0.0]
        optical_paths = ["0"]
        writer = TileSequencer(
            tile_writer=tile_writer,
            tile_cache=tile_cache,
            level_index=0,
            tiled_size=tiled_size,
            focal_planes=focal_planes,
            optical_paths=optical_paths,
            input_queue=input_queue,
            token=token,
        )
        writer.start()

        # Act & Assert
        with pytest.raises(RuntimeError, match="already running"):
            writer.start()

        # Cleanup
        input_queue.put(ShutdownSentinel(), token)
        writer.finalize()


def _run_with_real_cache(
    decoy: Decoy,
    tiled_size: Size,
    focal_planes: Sequence[float],
    optical_paths: Sequence[str],
    batches: dict[int, EncodingTaskResult],
    submission_order: list[int],
) -> list[bytes]:
    """Helper to run TileSequencer with a real DictTileCache.

    Parameters
    ----------
    decoy: Decoy
        Mock framework instance.
    tiled_size: Size
        Grid size in tiles.
    focal_planes: Sequence[float]
        Focal plane values.
    optical_paths: Sequence[str]
        Optical path identifiers.
    batches: Dict[int, EncodingTaskResult]
        Map of tile index to batch (all must be present).
    submission_order: List[int]
        Order in which tile indices are submitted.

    Returns
    -------
    List[bytes]
        Tiles in the order they were written.
    """
    written_data: list[bytes] = []
    tile_writer = decoy.mock(cls=WsiDicomWriter)

    def capture_write(tiles: Iterable[bytes]) -> int:
        tiles_list = list(tiles)
        written_data.extend(tiles_list)
        return len(tiles_list)

    decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)

    real_cache = DictTileCache()
    input_queue: PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel] = PriorityCancelableQueue()
    token = CancellationToken()

    sequencer = TileSequencer(
        tile_writer=tile_writer,
        tile_cache=real_cache,
        level_index=0,
        tiled_size=tiled_size,
        focal_planes=focal_planes,
        optical_paths=optical_paths,
        input_queue=input_queue,
        token=token,
    )
    sequencer.start()
    for index in submission_order:
        input_queue.put(batches[index], token)
    input_queue.put(ShutdownSentinel(), token)
    sequencer.finalize()

    return written_data


@pytest.mark.unittest
class TestTileSequencerWithRealCache:
    """Integration tests using real DictTileCache.

    These exercise the generator returned by DictTileCache.load_sequential,
    which is always truthy. The ``if tiles:`` guard in
    ``_flush_consecutive_tiles`` must correctly drain the generator once
    and then stop (not spin forever as ``while tiles:`` would).
    """

    def test_all_in_order(self, decoy: Decoy):
        """Sequential submission — no caching needed."""
        tiled_size = Size(3, 2)
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        order = list(range(tiled_size.area))

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in order for t in batches[i].tiles]
        assert written == expected

    def test_fully_reversed(self, decoy: Decoy):
        """All tiles arrive in reverse — everything cached, single flush."""
        tiled_size = Size(4, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(4)}
        order = [3, 2, 1, 0]

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in range(4) for t in batches[i].tiles]
        assert written == expected

    def test_reverse_3x3_grid(self, decoy: Decoy):
        """3x3 grid submitted in reverse row-major order."""
        tiled_size = Size(3, 3)
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        order = list(range(tiled_size.area - 1, -1, -1))

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in range(tiled_size.area) for t in batches[i].tiles]
        assert written == expected

    def test_interleaved_arrivals(self, decoy: Decoy):
        """Even indices first, then odd — triggers multiple partial flushes."""
        tiled_size = Size(6, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(6)}
        # Submit: 0, 2, 4, 1, 3, 5
        order = [0, 2, 4, 1, 3, 5]

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in range(6) for t in batches[i].tiles]
        assert written == expected

    def test_gap_then_fill(self, decoy: Decoy):
        """Tiles arrive with gaps, then gaps are filled in.

        Order: 0, 3, 1, 4, 2, 5. After tile 0 is written directly, tiles
        3 and 1 are cached. When tile 1 arrives it would not trigger a flush
        because 1 != next_expected (which is 1 after writing 0). Actually 1
        _is_ the next expected so it writes 1, then flushes 2 if present, etc.
        This tests the cascade where filling a gap triggers draining.
        """
        tiled_size = Size(6, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(6)}
        order = [0, 5, 4, 3, 2, 1]

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in range(6) for t in batches[i].tiles]
        assert written == expected

    def test_single_tile(self, decoy: Decoy):
        """Edge case: 1x1 grid, single tile."""
        tiled_size = Size(1, 1)
        batches = {0: make_batch(x=0, y=0)}

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, [0])

        assert written == list(batches[0].tiles)

    def test_batch_tiles_with_real_cache(self, decoy: Decoy):
        """Multi-tile batches (count > 1) submitted out of order."""
        tiled_size = Size(4, 2)
        # Each batch covers a full row (4 tiles)
        batches = {
            y * tiled_size.width: make_batch(x=0, y=y, count=tiled_size.width)
            for y in range(tiled_size.height)
        }
        # Submit row 1 before row 0
        order = [tiled_size.width, 0]

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in sorted(batches) for t in batches[i].tiles]
        assert written == expected

    def test_multiple_z_planes_with_real_cache(self, decoy: Decoy):
        """Multiple focal planes submitted out of order."""
        tiled_size = Size(2, 2)
        focal_planes = [0.0, 5.0]
        batches = {
            x + y * tiled_size.width + z * tiled_size.area: make_batch(x=x, y=y, z=z)
            for z in range(len(focal_planes))
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        total = tiled_size.area * len(focal_planes)
        # Submit in reverse
        order = list(range(total - 1, -1, -1))

        written = _run_with_real_cache(
            decoy, tiled_size, focal_planes, ["0"], batches, order
        )

        expected = [t for i in range(total) for t in batches[i].tiles]
        assert written == expected

    def test_multiple_optical_paths_with_real_cache(self, decoy: Decoy):
        """Multiple optical paths submitted out of order."""
        tiled_size = Size(2, 2)
        optical_paths = ["0", "1"]
        batches = {
            x + y * tiled_size.width + p * tiled_size.area: make_batch(x=x, y=y, path=p)
            for p in range(len(optical_paths))
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        total = tiled_size.area * len(optical_paths)
        order = list(range(total - 1, -1, -1))

        written = _run_with_real_cache(
            decoy, tiled_size, [0.0], optical_paths, batches, order
        )

        expected = [t for i in range(total) for t in batches[i].tiles]
        assert written == expected

    def test_large_grid_reverse(self, decoy: Decoy):
        """Larger 5x5 grid submitted fully reversed."""
        tiled_size = Size(5, 5)
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        order = list(range(tiled_size.area - 1, -1, -1))

        written = _run_with_real_cache(decoy, tiled_size, [0.0], ["0"], batches, order)

        expected = [t for i in range(tiled_size.area) for t in batches[i].tiles]
        assert written == expected

    def test_multiple_z_and_paths_interleaved(self, decoy: Decoy):
        """2x2 grid with 2 z-planes and 2 optical paths, interleaved order."""
        tiled_size = Size(2, 2)
        focal_planes = [0.0, 5.0]
        optical_paths = ["0", "1"]
        num_z = len(focal_planes)
        num_p = len(optical_paths)
        batches = {
            x
            + y * tiled_size.width
            + z * tiled_size.area
            + p * tiled_size.area * num_z: make_batch(x=x, y=y, z=z, path=p)
            for p in range(num_p)
            for z in range(num_z)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        total = tiled_size.area * num_z * num_p
        # Interleave: even indices, then odd indices
        order = list(range(0, total, 2)) + list(range(1, total, 2))

        written = _run_with_real_cache(
            decoy, tiled_size, focal_planes, optical_paths, batches, order
        )

        expected = [t for i in range(total) for t in batches[i].tiles]
        assert written == expected


def _run_with_byte_budget_cache(
    decoy: Decoy,
    tmp_path: UPath,
    tiled_size: Size,
    focal_planes: Sequence[float],
    optical_paths: Sequence[str],
    batches: dict[int, EncodingTaskResult],
    submission_order: list[int],
    memory_budget_bytes: int = 1024 * 1024,
) -> list[bytes]:
    """Helper to run TileSequencer with a ByteBudgetTileCache.

    Parameters
    ----------
    decoy: Decoy
        Mock framework instance.
    tmp_path: UPath
        Temp directory for disk cache.
    tiled_size: Size
        Grid size in tiles.
    focal_planes: Sequence[float]
        Focal plane values.
    optical_paths: Sequence[str]
        Optical path identifiers.
    batches: Dict[int, EncodingTaskResult]
        Map of tile index to batch (all must be present).
    submission_order: List[int]
        Order in which tile indices are submitted.
    memory_budget_bytes: int
        Byte budget for memory cache.

    Returns
    -------
    List[bytes]
        Tiles in the order they were written.
    """
    written_data: list[bytes] = []
    tile_writer = decoy.mock(cls=WsiDicomWriter)

    def capture_write(tiles: Iterable[bytes]) -> int:
        tiles_list = list(tiles)
        written_data.extend(tiles_list)
        return len(tiles_list)

    decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)

    real_cache = ByteBudgetTileCache(
        cache_dir=UPath(tmp_path / "byte_budget_cache"),
        memory_budget_bytes=memory_budget_bytes,
    )
    input_queue: PriorityCancelableQueue[EncodingTaskResult | ShutdownSentinel] = PriorityCancelableQueue()
    token = CancellationToken()

    sequencer = TileSequencer(
        tile_writer=tile_writer,
        tile_cache=real_cache,
        level_index=0,
        tiled_size=tiled_size,
        focal_planes=focal_planes,
        optical_paths=optical_paths,
        input_queue=input_queue,
        token=token,
    )
    sequencer.start()
    for index in submission_order:
        input_queue.put(batches[index], token)
    input_queue.put(ShutdownSentinel(), token)
    sequencer.finalize()

    return written_data


@pytest.mark.unittest
class TestTileSequencerWithByteBudgetCache:
    """Integration tests using ByteBudgetTileCache with TileSequencer."""

    def test_all_in_order(self, decoy: Decoy, tmp_path: UPath):
        """Sequential submission — no caching needed."""
        tiled_size = Size(3, 2)
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        order = list(range(tiled_size.area))

        written = _run_with_byte_budget_cache(
            decoy, tmp_path, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in order for t in batches[i].tiles]
        assert written == expected

    def test_fully_reversed(self, decoy: Decoy, tmp_path: UPath):
        """All tiles arrive in reverse — everything cached, single flush."""
        tiled_size = Size(4, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(4)}
        order = [3, 2, 1, 0]

        written = _run_with_byte_budget_cache(
            decoy, tmp_path, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in range(4) for t in batches[i].tiles]
        assert written == expected

    def test_disk_overflow(self, decoy: Decoy, tmp_path: UPath):
        """Tiles overflow to disk when memory budget is exceeded."""
        tiled_size = Size(8, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(8)}
        order = list(range(7, -1, -1))

        # Each tile is ~30 bytes, set budget to hold ~2 tiles
        written = _run_with_byte_budget_cache(
            decoy,
            tmp_path,
            tiled_size,
            [0.0],
            ["0"],
            batches,
            order,
            memory_budget_bytes=60,
        )

        expected = [t for i in range(8) for t in batches[i].tiles]
        assert written == expected

    def test_interleaved_arrivals(self, decoy: Decoy, tmp_path: UPath):
        """Even indices first, then odd — triggers multiple partial flushes."""
        tiled_size = Size(6, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(6)}
        order = [0, 2, 4, 1, 3, 5]

        written = _run_with_byte_budget_cache(
            decoy, tmp_path, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in range(6) for t in batches[i].tiles]
        assert written == expected

    def test_zero_budget_all_to_disk(self, decoy: Decoy, tmp_path: UPath):
        """Zero budget forces all tiles to disk — still works correctly."""
        tiled_size = Size(4, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(4)}
        order = [3, 2, 1, 0]

        written = _run_with_byte_budget_cache(
            decoy,
            tmp_path,
            tiled_size,
            [0.0],
            ["0"],
            batches,
            order,
            memory_budget_bytes=0,
        )

        expected = [t for i in range(4) for t in batches[i].tiles]
        assert written == expected

    def test_multiple_z_and_paths(self, decoy: Decoy, tmp_path: UPath):
        """Multiple z-planes and optical paths with byte-budget cache."""
        tiled_size = Size(2, 2)
        focal_planes = [0.0, 5.0]
        optical_paths = ["0", "1"]
        num_z = len(focal_planes)
        num_p = len(optical_paths)
        batches = {
            x
            + y * tiled_size.width
            + z * tiled_size.area
            + p * tiled_size.area * num_z: make_batch(x=x, y=y, z=z, path=p)
            for p in range(num_p)
            for z in range(num_z)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        total = tiled_size.area * num_z * num_p
        order = list(range(total - 1, -1, -1))

        written = _run_with_byte_budget_cache(
            decoy, tmp_path, tiled_size, focal_planes, optical_paths, batches, order
        )

        expected = [t for i in range(total) for t in batches[i].tiles]
        assert written == expected


def _run_with_dict_cache_fifo(
    decoy: Decoy,
    tiled_size: Size,
    focal_planes: Sequence[float],
    optical_paths: Sequence[str],
    batches: dict[int, EncodingTaskResult],
    submission_order: list[int],
) -> list[bytes]:
    """Helper to run TileSequencer with DictTileCache + FIFO Queue.

    Parameters
    ----------
    decoy: Decoy
        Mock framework instance.
    tiled_size: Size
        Grid size in tiles.
    focal_planes: Sequence[float]
        Focal plane values.
    optical_paths: Sequence[str]
        Optical path identifiers.
    batches: Dict[int, EncodingTaskResult]
        Map of tile index to batch (all must be present).
    submission_order: List[int]
        Order in which tile indices are submitted.

    Returns
    -------
    List[bytes]
        Tiles in the order they were written.
    """
    written_data: list[bytes] = []
    tile_writer = decoy.mock(cls=WsiDicomWriter)

    def capture_write(tiles: Iterable[bytes]) -> int:
        tiles_list = list(tiles)
        written_data.extend(tiles_list)
        return len(tiles_list)

    decoy.when(tile_writer.write_tiles(Anything())).then_do(capture_write)

    dict_cache = DictTileCache()
    input_queue: FifoCancelableQueue[EncodingTaskResult | ShutdownSentinel] = FifoCancelableQueue()
    token = CancellationToken()

    sequencer = TileSequencer(
        tile_writer=tile_writer,
        tile_cache=dict_cache,
        level_index=0,
        tiled_size=tiled_size,
        focal_planes=focal_planes,
        optical_paths=optical_paths,
        input_queue=input_queue,
        token=token,
    )
    sequencer.start()
    for index in submission_order:
        input_queue.put(batches[index], token)
    input_queue.put(ShutdownSentinel(), token)
    sequencer.finalize()

    return written_data


@pytest.mark.unittest
class TestTileSequencerWithDictCacheFifo:
    """Integration tests using DictTileCache + FIFO Queue with TileSequencer."""

    def test_fifo_queue_with_dict_cache_sequential(self, decoy: Decoy):
        """Submit tiles in order via FIFO queue with DictTileCache."""
        tiled_size = Size(3, 2)
        batches = {
            x + y * tiled_size.width: make_batch(x=x, y=y)
            for y in range(tiled_size.height)
            for x in range(tiled_size.width)
        }
        order = list(range(tiled_size.area))

        written = _run_with_dict_cache_fifo(
            decoy, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in order for t in batches[i].tiles]
        assert written == expected

    def test_fifo_queue_with_dict_cache_reversed(self, decoy: Decoy):
        """Submit tiles in reverse via FIFO queue with DictTileCache."""
        tiled_size = Size(4, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(4)}
        order = [3, 2, 1, 0]

        written = _run_with_dict_cache_fifo(
            decoy, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in range(4) for t in batches[i].tiles]
        assert written == expected

    def test_fifo_queue_with_dict_cache_interleaved(self, decoy: Decoy):
        """Submit tiles interleaved via FIFO queue with DictTileCache."""
        tiled_size = Size(6, 1)
        batches = {x: make_batch(x=x, y=0) for x in range(6)}
        order = [0, 2, 4, 1, 3, 5]

        written = _run_with_dict_cache_fifo(
            decoy, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in range(6) for t in batches[i].tiles]
        assert written == expected

    def test_fifo_queue_with_dict_cache_batch_tiles(self, decoy: Decoy):
        """Multi-tile batches submitted out of order via FIFO queue."""
        tiled_size = Size(4, 2)
        batches = {
            y * tiled_size.width: make_batch(x=0, y=y, count=tiled_size.width)
            for y in range(tiled_size.height)
        }
        # Submit row 1 before row 0
        order = [tiled_size.width, 0]

        written = _run_with_dict_cache_fifo(
            decoy, tiled_size, [0.0], ["0"], batches, order
        )

        expected = [t for i in sorted(batches) for t in batches[i].tiles]
        assert written == expected
