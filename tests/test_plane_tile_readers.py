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

import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pytest

from wsidicom.geometry import Size
from wsidicom.instance import TileByteRanges
from wsidicom.plane_tile_readers import PlaneTileReader, PlaneTileReaders
from wsidicom.series import Pyramids

WAIT_SECONDS = 10
"""The longest a test waits for another thread. A fault then fails the test instead of
hanging it."""

NOT_RETURNED_SECONDS = 0.2
"""How long a read that should be waiting is given to return. A read that is slow to
start also passes, as it has not returned either."""

IMAGE_SIZE = Size(25, 15)
"""Three by two tiles of 10 x 10 pixels. The tiles of the last column and row are
cropped."""

TILE_SIZE = Size(10, 10)


class InMemoryFrameReader:
    """Reads frames from bytes held in memory."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, position: int, length: int, /) -> bytes:
        return self._data[position : position + length]


class BlockingPlaneTileReaders(PlaneTileReaders):
    """Readers whose reader of a plane is made by waiting for the test to release it,
    and then returning None or raising."""

    def __init__(self, blocked_level: int | None, error: Exception | None = None):
        super().__init__(Pyramids([]))
        self._blocked_level = blocked_level
        self._error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.made: list[tuple[int, int, float | None, str | None]] = []

    def _create_reader(
        self, pyramid: int, level: int, z: float | None, path: str | None
    ) -> PlaneTileReader | None:
        self.made.append((pyramid, level, z, path))
        if self._blocked_level is None or level == self._blocked_level:
            self.started.set()
            self.release.wait(WAIT_SECONDS)
        if self._error is not None:
            raise self._error
        return None


def plane_tile_reader(
    lengths: list[int], file_indices: list[int] | None = None
) -> PlaneTileReader:
    """Reader of the 3 x 2 tiles, tile i stored as the bytes from 10 * i, in one file
    or, with file indices, in two files whose bytes differ."""
    byte_ranges = TileByteRanges.from_arrays(
        IMAGE_SIZE,
        TILE_SIZE,
        np.arange(0, 60, 10, dtype=np.int64),
        np.array(lengths, dtype=np.int64),
        None if file_indices is None else np.array(file_indices, dtype=np.int64),
    )
    readers = [
        InMemoryFrameReader(bytes(range(60))),
        InMemoryFrameReader(bytes(range(100, 160))),
    ]
    return PlaneTileReader(byte_ranges, readers)


@pytest.mark.unittest
class TestPlaneTileReader:
    def test_read_returns_the_bytes_at_the_byte_range_of_the_tile(self):
        # Arrange
        reader = plane_tile_reader([4] * 6)

        # Act
        tile = reader.read(1, 0, crop_to_image_boundary=True)

        # Assert
        assert tile == bytes(range(10, 14))

    def test_read_reads_from_the_file_of_the_tile(self):
        # Arrange
        reader = plane_tile_reader([4] * 6, file_indices=[0, 1, 0, 1, 0, 1])

        # Act
        tile = reader.read(1, 0, crop_to_image_boundary=True)

        # Assert
        assert tile == bytes(range(110, 114))

    def test_read_returns_none_for_a_tile_to_read_through_the_image_data(self):
        # Arrange
        read_through = TileByteRanges.READ_THROUGH_IMAGE_DATA
        reader = plane_tile_reader([4, read_through, 4, 4, 4, 4])

        # Act
        tile = reader.read(1, 0, crop_to_image_boundary=True)

        # Assert
        assert tile is None

    @pytest.mark.parametrize(
        ["x", "y", "crop_to_image_boundary"],
        [(2, 0, True), (0, 1, True), (3, 0, False), (0, 2, False), (-1, 0, False)],
    )
    def test_read_returns_none_for_a_tile_not_to_read_at_its_byte_range(
        self, x: int, y: int, crop_to_image_boundary: bool
    ):
        """A tile outside the tiled size, or a cropped tile when cropping, is read
        through the level."""
        # Arrange
        reader = plane_tile_reader([4] * 6)

        # Act
        tile = reader.read(x, y, crop_to_image_boundary)

        # Assert
        assert tile is None

    def test_read_reads_a_cropped_tile_when_not_cropping(self):
        # Arrange
        reader = plane_tile_reader([4] * 6)

        # Act
        tile = reader.read(2, 1, crop_to_image_boundary=False)

        # Assert
        assert tile == bytes(range(50, 54))


@pytest.mark.unittest
class TestPlaneTileReaders:
    """The reader of a plane is made when a tile of the plane is first read. A read of a
    plane whose reader is being made waits for it. Reads of other planes do not wait.
    """

    def test_read_returns_none_for_a_plane_not_in_the_pyramids(self):
        # Arrange
        readers = PlaneTileReaders(Pyramids([]))

        # Act
        tile = readers.read(0, 0, (0, 0), None, None, True)

        # Assert
        assert tile is None

    def test_other_plane_is_made_while_a_plane_is_being_made(self):
        # Arrange
        readers = BlockingPlaneTileReaders(blocked_level=0)
        with ThreadPoolExecutor(1) as executor:
            being_made = executor.submit(readers.read, 0, 0, (0, 0), None, None, True)
            assert readers.started.wait(WAIT_SECONDS)

            # Act
            other = readers.read(0, 1, (0, 0), None, None, True)
            done_before_release = being_made.done()
            readers.release.set()

        # Assert
        assert other is None
        assert not done_before_release
        assert being_made.result(WAIT_SECONDS) is None

    def test_reads_of_a_plane_at_once_make_it_once(self):
        # Arrange
        readers = BlockingPlaneTileReaders(blocked_level=None)
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(readers.read, 0, 0, (0, 0), None, None, True)
            assert readers.started.wait(WAIT_SECONDS)

            # Act
            second = executor.submit(readers.read, 0, 0, (1, 0), None, None, True)
            wait([second], NOT_RETURNED_SECONDS)
            second_done_before_release = second.done()
            readers.release.set()

        # Assert
        assert not second_done_before_release
        assert first.result(WAIT_SECONDS) is None
        assert second.result(WAIT_SECONDS) is None
        assert readers.made == [(0, 0, None, None)]

    def test_failure_to_make_a_plane_reaches_reads_waiting_and_is_tried_again(self):
        # Arrange
        readers = BlockingPlaneTileReaders(
            blocked_level=None, error=ValueError("Failed to make the reader.")
        )
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(readers.read, 0, 0, (0, 0), None, None, True)
            assert readers.started.wait(WAIT_SECONDS)
            second = executor.submit(readers.read, 0, 0, (0, 0), None, None, True)
            wait([second], NOT_RETURNED_SECONDS)
            readers.release.set()
            first_exception = first.exception(WAIT_SECONDS)
            second_exception = second.exception(WAIT_SECONDS)

        # Act & Assert
        with pytest.raises(ValueError):
            readers.read(0, 0, (0, 0), None, None, True)
        assert isinstance(first_exception, ValueError)
        assert isinstance(second_exception, ValueError)
        assert readers.made == [(0, 0, None, None), (0, 0, None, None)]
