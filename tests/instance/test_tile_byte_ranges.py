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

import numpy as np
import pytest

from wsidicom.geometry import Size
from wsidicom.instance import TileByteRanges

IMAGE_SIZE = Size(25, 15)
"""Tiled 3 x 2 by tiles of 10 x 10, of which the last column and row are cropped."""
TILE_SIZE = Size(10, 10)


@pytest.mark.unittest
class TestTileByteRanges:
    def test_from_arrays_holds_byte_ranges_of_each_tile(self):
        # Arrange

        # Act
        byte_ranges = TileByteRanges.from_arrays(
            IMAGE_SIZE,
            TILE_SIZE,
            np.arange(0, 60, 10, dtype=np.int64),
            np.array([4, TileByteRanges.READ_THROUGH_IMAGE_DATA, 4, 4, 4, 4]),
        )

        # Assert
        assert byte_ranges.tiled_size == Size(3, 2)
        assert byte_ranges.whole_tiled_size == Size(2, 1)
        assert byte_ranges.positions.tolist() == [0, 10, 20, 30, 40, 50]
        assert byte_ranges.lengths.tolist() == [
            4,
            TileByteRanges.READ_THROUGH_IMAGE_DATA,
            4,
            4,
            4,
            4,
        ]
        assert byte_ranges.file_indices is None

    def test_from_arrays_holds_file_indices_of_each_tile(self):
        # Arrange

        # Act
        byte_ranges = TileByteRanges.from_arrays(
            IMAGE_SIZE,
            TILE_SIZE,
            np.zeros(6, dtype=np.int64),
            np.zeros(6, dtype=np.int64),
            file_indices=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        )

        # Assert
        assert byte_ranges.file_indices is not None
        assert byte_ranges.file_indices.tolist() == [0, 1, 0, 1, 0, 1]

    def test_from_arrays_shares_int64_arrays_read_only(self):
        """The byte ranges are views of arrays made by the image data, not copies,
        and cannot be changed through them."""
        # Arrange
        positions = np.arange(0, 60, 10, dtype=np.int64)

        # Act
        byte_ranges = TileByteRanges.from_arrays(
            IMAGE_SIZE, TILE_SIZE, positions, np.zeros(6, dtype=np.int64)
        )

        # Assert
        assert np.shares_memory(np.asarray(byte_ranges.positions), positions)
        assert byte_ranges.positions.readonly
        assert positions.flags.writeable

    def test_item_of_byte_ranges_is_int(self):
        """An item is a plain int, as a numpy integer costs more to index with and
        to read at."""
        # Arrange
        byte_ranges = TileByteRanges.from_arrays(
            IMAGE_SIZE,
            TILE_SIZE,
            np.arange(0, 60, 10, dtype=np.int64),
            np.zeros(6, dtype=np.int64),
        )

        # Act
        position = byte_ranges.positions[1]

        # Assert
        assert type(position) is int
        assert position == 10

    def test_create_with_positions_not_one_per_tile_raises(self):
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            TileByteRanges.from_arrays(
                IMAGE_SIZE,
                TILE_SIZE,
                np.zeros(5, dtype=np.int64),
                np.zeros(5, dtype=np.int64),
            )

    def test_create_with_file_indices_not_one_per_tile_raises(self):
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            TileByteRanges.from_arrays(
                IMAGE_SIZE,
                TILE_SIZE,
                np.zeros(6, dtype=np.int64),
                np.zeros(6, dtype=np.int64),
                file_indices=np.zeros(5, dtype=np.int64),
            )
