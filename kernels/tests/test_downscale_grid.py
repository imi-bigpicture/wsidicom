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

"""Tests for downscale_grid: a pyramid step reducing a grid of tiles to one image.

The image is optionally placed in a filled frame.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, downscale_grid

RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]

DTYPES = [np.uint8, np.uint16, np.float32]
CHANNELS = [1, 3, 4]


@pytest.fixture
def tiles(
    random_image: RandomImage,
    tile_size: tuple[int, int],
    edge_size: tuple[int, int],
    grid_width: int,
    grid_height: int,
    dtype: type[np.generic],
    channels: int,
) -> list[np.ndarray]:
    """Make a row-major grid of random tiles from indirect params.

    ``tile_size`` tiles, with ``edge_size`` ones in the last column and row.
    """

    def tile_shape(row: int, column: int) -> tuple[int, ...]:
        height = edge_size[0] if row == grid_height - 1 else tile_size[0]
        width = edge_size[1] if column == grid_width - 1 else tile_size[1]
        return (height, width) if channels == 1 else (height, width, channels)

    return [
        random_image(tile_shape(row, column), dtype)
        for row in range(grid_height)
        for column in range(grid_width)
    ]


@pytest.fixture
def stitched(tiles: list[np.ndarray], grid_width: int) -> np.ndarray:
    """Stitch the tiles into one image."""
    rows = [
        np.concatenate(tiles[start : start + grid_width], axis=1)
        for start in range(0, len(tiles), grid_width)
    ]
    return np.concatenate(rows, axis=0)


@pytest.fixture
def fill(channels: int, dtype: type[np.generic]) -> np.ndarray:
    """Make a frame fill: a ``(C,)`` array in the tiles' dtype.

    A value per channel, distinct between channels (``(1,)`` for planar tiles).
    """
    return np.array([7] if channels == 1 else range(1, channels + 1), dtype)


@pytest.fixture
def rgb_tile() -> np.ndarray:
    return np.zeros((32, 32, 3), np.uint8)


@pytest.mark.unittest
class TestDownscaleGrid:
    @pytest.mark.parametrize(
        ["tile_size", "edge_size", "grid_width", "grid_height"],
        [
            ((64, 64), (64, 64), 2, 2),  # four full children
            ((64, 96), (37, 51), 3, 2),  # odd edge tiles
            ((128, 128), (128, 40), 2, 3),
            ((32, 48), (32, 48), 1, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("channels", CHANNELS)
    @pytest.mark.parametrize(
        ["factor", "filter"],
        [
            (2, ResizeFilter.BOX),
            (2, ResizeFilter.BILINEAR),
            (4, ResizeFilter.BOX),
            (8, ResizeFilter.BOX),
        ],
    )
    def test_equals_downscale_of_stitched(
        self,
        tiles: list[np.ndarray],
        stitched: np.ndarray,
        grid_width: int,
        factor: int,
        filter: ResizeFilter,
    ) -> None:
        """A grid of tiles reduced to one image equals reducing the stitched grid.

        Smaller tiles at the right and bottom edges; through the per-tile box
        path and the stitching fallback alike.
        """
        # Act
        reduced = downscale_grid(tiles, grid_width, factor, filter)

        # Assert
        assert np.array_equal(reduced, downscale(stitched, factor, filter))

    @pytest.mark.parametrize(
        ["grid_width", "grid_height"], [(1, 1), (2, 1), (1, 2), (2, 2)]
    )
    @pytest.mark.parametrize(["tile_size", "edge_size"], [((32, 32), (32, 32))])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("channels", CHANNELS)
    @pytest.mark.parametrize("filter", [ResizeFilter.BOX, ResizeFilter.BILINEAR])
    def test_in_a_filled_frame(
        self,
        tiles: list[np.ndarray],
        grid_width: int,
        dtype: type[np.generic],
        fill: np.ndarray,
        filter: ResizeFilter,
    ) -> None:
        """size= places the reduced grid at the top left of a frame, fill= around it.

        A parent tile from a partial block at the image's edge keeps the full
        tile size. Both the per-tile box path and the stitching fallback.
        """
        # Arrange
        block = downscale_grid(tiles, grid_width, 2, filter)
        expected = np.empty((32, 32, *tiles[0].shape[2:]), dtype)
        expected[...] = fill
        expected[: block.shape[0], : block.shape[1]] = block

        # Act
        framed = downscale_grid(tiles, grid_width, 2, filter, size=(32, 32), fill=fill)

        # Assert
        assert framed.dtype == dtype
        assert np.array_equal(framed, expected)

    def test_rejects_frame_smaller_than_reduced_grid(
        self, rgb_tile: np.ndarray
    ) -> None:
        """The frame must hold the reduced grid."""
        # Act & Assert
        with pytest.raises(
            ValueError, match="size must be at least the reduced grid's"
        ):
            downscale_grid(
                [rgb_tile],
                1,
                2,
                ResizeFilter.BOX,
                size=(15, 16),
                fill=np.zeros(3, np.uint8),
            )

    @pytest.mark.parametrize(
        ["fill", "message"],
        [
            (256, "fill must have the tiles' dtype"),
            (np.array([1, 2], np.uint8), "one value per channel"),
        ],
        ids=["scalar", "not-one-per-channel"],
    )
    def test_rejects_unfit_fill(
        self, rgb_tile: np.ndarray, fill: Any, message: str
    ) -> None:
        """The fill must be a ``(C,)`` array, one sample per channel.

        A scalar is no longer broadcast to every channel.
        """
        # Act & Assert
        with pytest.raises(ValueError, match=message):
            downscale_grid([rgb_tile], 1, 2, ResizeFilter.BOX, size=(32, 32), fill=fill)

    @pytest.mark.parametrize(
        ["other_tiles", "message"],
        [
            (
                [np.zeros((32, 32, 3), np.uint8)] * 2,
                "tiles must fill whole grid rows of grid_width tiles",
            ),
            (
                [np.zeros((32, 32, 3), np.uint16)],
                "every tile must have the first tile's dtype",
            ),
            (
                [np.zeros((16, 32, 3), np.uint8)],
                "every tile must share its grid row's height",
            ),
        ],
        ids=["partial-row", "mixed-dtypes", "uneven-row-heights"],
    )
    def test_rejects_bad_grids(
        self, rgb_tile: np.ndarray, other_tiles: list[np.ndarray], message: str
    ) -> None:
        """Tiles that do not form a grid of one dtype."""
        # Act & Assert
        with pytest.raises(ValueError, match=message):
            downscale_grid([rgb_tile, *other_tiles], 2, 2, ResizeFilter.BOX)

    def test_rejects_empty_grid(self) -> None:
        """No tiles at all."""
        # Act & Assert
        with pytest.raises(
            ValueError, match="tiles must fill whole grid rows of grid_width tiles"
        ):
            downscale_grid([], 2, 2, ResizeFilter.BOX)
