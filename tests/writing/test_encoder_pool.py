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

"""Tests for EncoderPool."""

from typing import cast

import numpy as np
import pytest
from decoy import Decoy, matchers

from wsidicom.codec import Encoder
from wsidicom.downsampler import Downsampler, PillowDownsampler
from wsidicom.geometry import Size
from wsidicom.stitcher import NumpyStitcher
from wsidicom.thread import (
    CancellationToken,
    CompletionTracker,
    FifoCancelableQueue,
    PriorityCancelableQueue,
)
from wsidicom.writing.encoder_pool import EncoderPool
from wsidicom.writing.models import (
    DownsampleEncodeTask,
    EncodeTask,
    EncodingTaskResult,
    PyramidTilePosition,
)


class ArrayEq:
    """Argument matcher for a numpy array.

    Decoy matches recorded call arguments by equality, but `==` on a numpy array
    is elementwise and yields an array, which is ambiguous as a bool. Setting
    `__array_ufunc__` to None makes numpy defer the comparison to this class,
    which then compares by value.
    """

    __array_ufunc__ = None

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __eq__(self, other: object) -> bool:
        return isinstance(other, np.ndarray) and np.array_equal(other, self._array)

    def __hash__(self) -> int:
        return hash(self._array.tobytes())

    def __repr__(self) -> str:
        return f"ArrayEq(shape={self._array.shape})"


def array_eq(array: np.ndarray) -> np.ndarray:
    """An ``ArrayEq`` matcher typed as ``np.ndarray`` so decoy call sites, which
    pass a matcher where the real argument type is expected, type-check."""
    return cast(np.ndarray, ArrayEq(array))


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.fixture
def dtype() -> np.dtype:
    """Pixel dtype for building test arrays."""
    return np.dtype(np.uint8)


@pytest.fixture
def test_tile(dtype: np.dtype) -> np.ndarray:
    """Create a test tile array."""
    return np.full((256, 256, 3), (255, 0, 0), dtype=dtype)


@pytest.fixture
def downsampler(decoy: Decoy) -> Downsampler:
    """Create a mock downsampler."""
    return decoy.mock(cls=Downsampler)


@pytest.fixture
def stitcher(decoy: Decoy) -> NumpyStitcher:
    """Mock NumpyStitcher; downsample tests configure return values."""
    return decoy.mock(cls=NumpyStitcher)


@pytest.fixture
def tile_size() -> Size:
    """Create a default tile size."""
    return Size(256, 256)


@pytest.fixture
def blank_tile(dtype: np.dtype) -> np.ndarray:
    """Background tile used to pad downsampled edge blocks to a full tile."""
    return np.full((256, 256, 3), 255, dtype=dtype)


@pytest.fixture
def token() -> CancellationToken:
    """Create a cancellation token for queue operations."""
    return CancellationToken()


@pytest.mark.unittest
class TestPyramidTilePosition:
    """Tests for PyramidTilePosition comparison and indexing."""

    def test_comparison_by_level(self):
        """Test that lower levels are less than higher levels."""
        # Arrange
        coord_low = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        coord_high = PyramidTilePosition(
            level=2, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )

        # Act
        result = coord_low < coord_high

        # Assert
        assert result

    def test_comparison_by_optical_path(self):
        """Test that lower optical paths are less than higher ones."""
        # Arrange
        coord1 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        coord2 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=1
        )

        # Act
        result = coord1 < coord2

        # Assert
        assert result

    def test_comparison_by_z_index(self):
        """Test that lower z indices are less than higher ones."""
        # Arrange
        coord1 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        coord2 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=1, optical_path_index=0
        )

        # Act
        result = coord1 < coord2

        # Assert
        assert result

    def test_comparison_by_y_index(self):
        """Test that lower y indices are less than higher ones (top before bottom)."""
        # Arrange
        coord1 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        coord2 = PyramidTilePosition(
            level=1, x_index=0, y_index=1, z_index=0, optical_path_index=0
        )

        # Act
        result = coord1 < coord2

        # Assert
        assert result

    def test_comparison_by_x_index(self):
        """Test that lower x indices are less than higher ones (left before right)."""
        # Arrange
        coord1 = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        coord2 = PyramidTilePosition(
            level=1, x_index=1, y_index=0, z_index=0, optical_path_index=0
        )

        # Act
        result = coord1 < coord2

        # Assert
        assert result


@pytest.mark.unittest
class TestMixedTaskPriorityQueue:
    """Regression: PriorityQueue must accept both task types without TypeError."""

    def test_encode_and_downsample_tasks_compare(self, token: CancellationToken):
        """EncodeTask and DownsampleEncodeTask with different coordinates compare."""
        # Arrange
        encode_task = EncodeTask(
            coordinates=PyramidTilePosition(0, 0, 0, 0, 0),
            tiles=[],
            output_queue=PriorityCancelableQueue(),
        )
        downsample_task = DownsampleEncodeTask(
            coordinates=PyramidTilePosition(0, 1, 0, 0, 0),
            tiles=[],
            output_queue=PriorityCancelableQueue(),
            cascade_tracker=CompletionTracker(),
        )

        # Act
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        queue.put(encode_task, token)
        queue.put(downsample_task, token)
        first = queue.get(token)
        second = queue.get(token)

        # Assert — coordinate ordering preserved across types
        assert first is encode_task
        assert second is downsample_task

    def test_tasks_with_equal_coordinates_compare(self, token: CancellationToken):
        """Tasks with identical coordinates compare without TypeError."""
        # Arrange
        coords = PyramidTilePosition(0, 0, 0, 0, 0)
        encode_task = EncodeTask(
            coordinates=coords, tiles=[], output_queue=PriorityCancelableQueue()
        )
        downsample_task = DownsampleEncodeTask(
            coordinates=coords,
            tiles=[],
            output_queue=PriorityCancelableQueue(),
            cascade_tracker=CompletionTracker(),
        )

        # Act
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        queue.put(encode_task, token)
        queue.put(downsample_task, token)

        # Assert — both come out; order between equal-coord tasks is unspecified
        out = [queue.get(token), queue.get(token)]
        assert encode_task in out and downsample_task in out


@pytest.mark.unittest
class TestEncoderPool:
    """Tests for EncoderPool."""

    def test_encode_one_task(
        self,
        decoy: Decoy,
        test_tile: np.ndarray,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """Test that encoding works via the queue property."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        expected_bytes = b"encoded_data"
        decoy.when(encoder.encode(test_tile)).then_return(expected_bytes)
        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        coordinates = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        task = EncodeTask(
            coordinates=coordinates,
            tiles=[test_tile],
            output_queue=output_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert
        assert output_queue.qsize() == 1
        result = output_queue.get(token)
        assert result.tiles[0] == expected_bytes

    def test_encode_two_tasks(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """Test that encoding works via the queue property."""
        # Arrange
        test_tile_1 = np.full((256, 256, 3), (255, 0, 0), dtype=dtype)
        test_tile_2 = np.full((256, 256, 3), (0, 255, 0), dtype=dtype)
        expected_bytes_1 = b"encoded_data_1"
        expected_bytes_2 = b"encoded_data_2"
        encoder = decoy.mock(cls=Encoder)

        decoy.when(encoder.encode(array_eq(test_tile_1))).then_return(expected_bytes_1)
        decoy.when(encoder.encode(array_eq(test_tile_2))).then_return(expected_bytes_2)
        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        coordinates = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        task_1 = EncodeTask(
            coordinates=coordinates,
            tiles=[test_tile_1],
            output_queue=output_queue,
        )
        task_2 = EncodeTask(
            coordinates=coordinates,
            tiles=[test_tile_2],
            output_queue=output_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task_1, token)
            pool.queue.put(task_2, token)

        # Assert
        assert output_queue.qsize() == 2
        first_result = output_queue.get(token)
        assert first_result.tiles[0] == expected_bytes_1
        second_result = output_queue.get(token)
        assert second_result.tiles[0] == expected_bytes_2

    def test_encode_batch_with_multiple_tiles(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """Test encoding a batch with multiple tiles."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        tiles = [np.full((256, 256, 3), (i * 50, 0, 0), dtype=dtype) for i in range(4)]
        expected_results = [f"encoded_{i}".encode() for i in range(4)]
        for _, (tile, expected_result) in enumerate(
            zip(tiles, expected_results, strict=True)
        ):
            decoy.when(encoder.encode(array_eq(tile))).then_return(expected_result)
        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        coordinates = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        task = EncodeTask(
            coordinates=coordinates,
            tiles=tiles,
            output_queue=output_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert
        assert output_queue.qsize() == 1
        result = output_queue.get(token)
        assert len(result.tiles) == 4
        for i, expected_result in enumerate(expected_results):
            assert result.tiles[i] == expected_result

    def test_start_twice_raises_error(
        self,
        decoy: Decoy,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """Test that starting twice raises an error."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        pool = EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        )
        pool.start()

        # Act & Assert
        with pytest.raises(RuntimeError, match="already running"):
            pool.start()

        pool.shutdown()

    def test_encoding_failure_cancels_token(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """An encoding failure cancels the shared token instead of deadlocking."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        test_tile = np.full((256, 256, 3), (255, 0, 0), dtype=dtype)
        decoy.when(encoder.encode(test_tile)).then_raise(ValueError("Encoding failed"))
        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        coordinates = PyramidTilePosition(
            level=0, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        task = EncodeTask(
            coordinates=coordinates,
            tiles=[test_tile],
            output_queue=output_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - token cancelled with the encoding error, no result emitted
        assert token.is_cancelled() is True
        assert isinstance(token.exception, ValueError)
        assert output_queue.qsize() == 0

    def test_shutdown_stops_workers(
        self,
        decoy: Decoy,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """Test that shutdown properly stops dispatcher and executor."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        pool = EncoderPool(
            encoder,
            num_workers=4,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        )
        pool.start()

        # Act
        pool.shutdown(wait=True)

        # Assert
        assert not pool._dispatcher_thread.is_alive()

    def test_downsample_and_encode(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """EncoderPool stitches input tiles, downsamples, encodes, and cascades."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        decoy.when(downsampler.commutes_with_stitch).then_return(False)

        composite = np.full((512, 512, 3), (10, 20, 30), dtype=dtype)
        downsampled = np.full((256, 256, 3), (50, 50, 50), dtype=dtype)
        decoy.when(stitcher.stitch_grid(matchers.Anything())).then_return(composite)
        decoy.when(downsampler.downsample(composite, Size(256, 256))).then_return(
            downsampled
        )
        decoy.when(encoder.encode(downsampled)).then_return(b"encoded")

        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        cascade_queue: FifoCancelableQueue = FifoCancelableQueue()
        tracker = CompletionTracker()

        coordinates = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        tiles = [
            [
                np.full((256, 256, 3), (0, 0, 0), dtype=dtype),
                np.full((256, 256, 3), (50, 0, 0), dtype=dtype),
            ],
            [
                np.full((256, 256, 3), (100, 0, 0), dtype=dtype),
                np.full((256, 256, 3), (150, 0, 0), dtype=dtype),
            ],
        ]
        tracker.increment()
        task = DownsampleEncodeTask(
            coordinates=coordinates,
            tiles=tiles,
            output_queue=output_queue,
            cascade_tracker=tracker,
            cascade_queue=cascade_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=Size(256, 256),
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - encoded result on output queue
        assert output_queue.qsize() == 1
        result = output_queue.get(token)
        assert result.tiles == [b"encoded"]

        # Assert - downsampled tile cascaded
        assert cascade_queue.qsize() == 1
        cascaded = cascade_queue.get(token)
        assert cascaded.tile is downsampled
        assert cascaded.x_index == 0
        assert cascaded.y_index == 0
        assert cascaded.z_index == 0
        assert cascaded.optical_path_index == 0

        # Assert - tracker decremented to zero
        tracker.wait_for_zero()

    def test_downsample_uses_reduce_fast_path(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """When downsampler commutes with stitch, no downsample call is made."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        decoy.when(downsampler.commutes_with_stitch).then_return(True)

        stitched = np.full((256, 256, 3), (80, 80, 80), dtype=dtype)
        decoy.when(stitcher.stitch_grid(matchers.Anything())).then_return(stitched)
        decoy.when(encoder.encode(stitched)).then_return(b"enc")

        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        tracker = CompletionTracker()

        coordinates = PyramidTilePosition(
            level=2, x_index=1, y_index=0, z_index=0, optical_path_index=0
        )
        # 1x2 edge block (1 column, 2 rows)
        tiles = [
            [np.full((256, 256, 3), (100, 0, 0), dtype=dtype)],
            [np.full((256, 256, 3), (200, 0, 0), dtype=dtype)],
        ]
        tracker.increment()
        task = DownsampleEncodeTask(
            coordinates=coordinates,
            tiles=tiles,
            output_queue=output_queue,
            cascade_tracker=tracker,
            cascade_queue=None,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=Size(256, 256),
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - stitched result is what got encoded (no downsample call)
        assert output_queue.qsize() == 1
        result = output_queue.get(token)
        assert result.tiles == [b"enc"]
        decoy.verify(
            downsampler.downsample(matchers.Anything(), matchers.Anything()), times=0
        )

        # Assert - tracker decremented
        tracker.wait_for_zero()

    def test_downsample_without_output_queue_cascades_without_encoding(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """An intermediate level is downsampled and cascaded, but not encoded.

        Such a level only exists to bridge a gap in a sparse pyramid, so nothing
        is written for it and the encode would be wasted work.
        """
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        decoy.when(downsampler.commutes_with_stitch).then_return(True)
        stitched = np.full((256, 256, 3), (80, 80, 80), dtype=dtype)
        decoy.when(stitcher.stitch_grid(matchers.Anything())).then_return(stitched)

        cascade_queue: FifoCancelableQueue = FifoCancelableQueue()
        tracker = CompletionTracker()
        tracker.increment()
        task = DownsampleEncodeTask(
            coordinates=PyramidTilePosition(
                level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
            ),
            tiles=[
                [np.zeros((256, 256, 3), dtype=dtype) for _ in range(2)]
                for _ in range(2)
            ],
            output_queue=None,
            cascade_tracker=tracker,
            cascade_queue=cascade_queue,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - the downsampled tile is cascaded, and never encoded
        assert cascade_queue.qsize() == 1
        assert cascade_queue.get(token).tile is stitched
        decoy.verify(encoder.encode(matchers.Anything()), times=0)
        tracker.wait_for_zero()

    def test_downsample_edge_block_is_padded_to_tile_size(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        tile_size: Size,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """A partial edge block is halved and padded back to a full tile.

        Regression: the reduce-then-stitch fast path returned the stitched block
        as-is, so an edge block (the level below has an odd number of tiles in a
        dimension) produced a frame smaller than the Rows/Columns declared in the
        dataset.
        """
        # Arrange - real downsampler and stitcher, so the sizes are real
        encoder = decoy.mock(cls=Encoder)
        encoded: list[np.ndarray] = []
        decoy.when(encoder.encode(matchers.Anything())).then_do(
            lambda tile: encoded.append(tile) or b"encoded"
        )
        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        tracker = CompletionTracker()
        # 1x2 edge block: one row of two tiles, i.e. no tile row below it.
        tiles = [[np.full((256, 256, 3), (10, 20, 30), dtype=dtype) for _ in range(2)]]
        tracker.increment()
        task = DownsampleEncodeTask(
            coordinates=PyramidTilePosition(
                level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
            ),
            tiles=tiles,
            output_queue=output_queue,
            cascade_tracker=tracker,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=PillowDownsampler(),
            stitcher=NumpyStitcher(),
            tile_size=tile_size,
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - the encoded tile is a full tile: halved block on top, and the
        # part below (outside the image) filled with the background.
        assert len(encoded) == 1
        padded = encoded[0]
        assert padded.shape == (tile_size.height, tile_size.width, 3)
        assert np.array_equal(padded[0, 0], np.array([10, 20, 30], dtype))
        assert np.array_equal(padded[200, 0], np.array([255, 255, 255], dtype))

        tracker.wait_for_zero()

    def test_downsample_failure_cancels_token_and_decrements(
        self,
        decoy: Decoy,
        dtype: np.dtype,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        blank_tile: np.ndarray,
        token: CancellationToken,
    ):
        """A downsampler exception cancels the token and still decrements the tracker."""
        # Arrange
        encoder = decoy.mock(cls=Encoder)
        decoy.when(downsampler.commutes_with_stitch).then_return(False)

        composite = np.zeros((512, 512, 3), dtype=dtype)
        decoy.when(stitcher.stitch_grid(matchers.Anything())).then_return(composite)
        decoy.when(
            downsampler.downsample(matchers.Anything(), matchers.Anything())
        ).then_raise(RuntimeError("downsample failed"))

        output_queue: PriorityCancelableQueue[EncodingTaskResult] = (
            PriorityCancelableQueue()
        )
        tracker = CompletionTracker()

        coordinates = PyramidTilePosition(
            level=1, x_index=0, y_index=0, z_index=0, optical_path_index=0
        )
        tiles = [
            [np.zeros((256, 256, 3), dtype=dtype) for _ in range(2)] for _ in range(2)
        ]
        tracker.increment()
        task = DownsampleEncodeTask(
            coordinates=coordinates,
            tiles=tiles,
            output_queue=output_queue,
            cascade_tracker=tracker,
            cascade_queue=None,
        )

        # Act
        with EncoderPool(
            encoder,
            num_workers=1,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=Size(256, 256),
            blank_tile=blank_tile,
            token=token,
        ) as pool:
            pool.queue.put(task, token)

        # Assert - token cancelled with the downsample error, no result emitted
        assert token.is_cancelled() is True
        assert isinstance(token.exception, RuntimeError)
        assert output_queue.qsize() == 0

        # Assert - tracker still decremented (no leak)
        tracker.wait_for_zero()
