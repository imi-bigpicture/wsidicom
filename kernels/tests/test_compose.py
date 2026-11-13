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

"""Tests for compose and stitch.

compose copies tiles onto the canvas at integer positions, planar or
interleaved, later tiles over earlier ones.
"""

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import (
    ResizeFilter,
    compose,
    downscale,
    resize,
    stitch,
)

DTYPES = [np.uint8, np.uint16]


@pytest.fixture
def random_tile(
    random_image: Callable[[tuple[int, ...], Any], np.ndarray],
    tile_shape: tuple[int, ...],
    dtype: type[np.generic],
) -> np.ndarray:
    """Make a random tile from an indirect ``(tile_shape, dtype)`` param."""
    return random_image(tile_shape, dtype)


@pytest.mark.unittest
class TestStitch:
    def test_places_tiles_on_integer_grid(self) -> None:
        """Tiles are laid out row by row, ``grid_width`` tiles per row."""
        # Arrange
        tiles = [np.full((4, 4), index, np.uint8) for index in range(6)]

        # Act
        stitched = stitch(tiles, grid_width=3)

        # Assert
        assert stitched.shape == (8, 12)
        assert (stitched[0:4, 0:4] == 0).all()
        assert (stitched[0:4, 4:8] == 1).all()
        assert (stitched[4:8, 8:12] == 5).all()

    @pytest.mark.parametrize(
        ["tile_size", "value", "expected_shape"],
        [
            (16, 200, (32, 32)),  # Y at full resolution
            (8, 128, (16, 16)),  # Cb/Cr at half resolution
        ],
    )
    def test_subsampled_chroma_shapes(
        self, tile_size: int, value: int, expected_shape: tuple[int, int]
    ) -> None:
        """Subsampled chroma planes stitch independently, with no special case."""
        # Arrange
        tiles = [np.full((tile_size, tile_size), value, np.uint8)] * 4

        # Act
        stitched = stitch(tiles, grid_width=2)

        # Assert
        assert stitched.shape == expected_shape


@pytest.mark.unittest
class TestComposeAtIntegerPositions:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("tile_shape", [(4, 5), (4, 5, 3)])
    def test_tile_is_copied_unchanged(
        self, random_tile: np.ndarray, dtype: type[np.generic]
    ) -> None:
        """A tile inside the canvas is copied there unchanged, and only there."""
        # Arrange
        canvas = np.zeros((9, 11, *random_tile.shape[2:]), dtype)

        # Act
        compose(canvas, [random_tile], np.array([[3, 2]], np.int64))

        # Assert
        np.testing.assert_array_equal(canvas[2:6, 3:8], random_tile)
        canvas[2:6, 3:8] = 0
        assert not canvas.any()

    def test_interleaved_tiles_on_grid(self) -> None:
        """Interleaved RGB tiles are placed on an integer grid."""
        # Arrange
        tiles = [
            np.full((4, 4, 3), (index, index + 1, index + 2), np.uint8)
            for index in range(6)
        ]
        canvas = np.zeros((8, 12, 3), np.uint8)
        positions = np.array(
            [[(index % 3) * 4, (index // 3) * 4] for index in range(6)], np.int64
        )

        # Act
        compose(canvas, tiles, positions)

        # Assert
        assert (canvas[0:4, 0:4] == (0, 1, 2)).all()
        assert (canvas[4:8, 8:12] == (5, 6, 7)).all()

    def test_later_tiles_overwrite_earlier(self) -> None:
        """Where tiles overlap, the last one wins."""
        # Arrange
        first = np.full((4, 4), 1, np.uint8)
        second = np.full((4, 4), 2, np.uint8)
        canvas = np.zeros((6, 6), np.uint8)

        # Act
        compose(canvas, [first, second], np.array([[0, 0], [2, 2]], np.int64))

        # Assert
        assert (canvas[0:2, 0:4] == 1).all()
        assert (canvas[2:4, 0:2] == 1).all()
        assert (canvas[2:6, 2:6] == 2).all()

    def test_returns_the_canvas(self) -> None:
        """The canvas is mutated in place and returned as that same array."""
        # Arrange
        canvas = np.zeros((4, 4), np.uint8)
        tile = np.ones((2, 2), np.uint8)

        # Act
        composed = compose(canvas, [tile], np.zeros((1, 2), np.int64))

        # Assert
        assert composed is canvas

    def test_no_tiles_leave_the_canvas_unchanged(self) -> None:
        """An empty tile list with (0, 2) positions writes nothing."""
        # Arrange
        canvas = np.full((4, 4), 7, np.uint8)

        # Act
        compose(canvas, [], np.zeros((0, 2), np.int64))

        # Assert
        assert (canvas == 7).all()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        ["tile_shape", "canvas_shape"],
        [
            ((37, 41), (50, 50)),
            ((37, 41, 3), (50, 50, 3)),
        ],
    )
    def test_simd_matches_scalar(
        self,
        simd_and_scalar: Callable[[Callable[[], Any]], tuple[Any, Any]],
        dtype: type[np.generic],
        canvas_shape: tuple[int, ...],
        random_tile: np.ndarray,
    ) -> None:
        """The SIMD kernels equal the scalar reference, planar and interleaved.

        Odd sizes exercise both the vector body and the tail.
        """

        # Arrange
        def compose_onto_empty_canvas() -> np.ndarray:
            canvas = np.zeros(canvas_shape, dtype)
            compose(canvas, [random_tile], np.array([[3, 2]], np.int64))
            return canvas

        # Act
        simd, scalar = simd_and_scalar(compose_onto_empty_canvas)

        # Assert
        np.testing.assert_array_equal(simd, scalar)


@pytest.mark.unittest
class TestComposePiecesOfTiles:
    """Crops and row slices of tiles compose as copies of those pieces would.

    Pieces butt, overlap each other and hang off the canvas's edges.
    """

    CROPS = np.array(
        [
            [0, 0, 0, 7, 9],
            [0, 7, 0, 6, 9],
            [1, 2, 3, 9, 5],
            [0, 0, 9, 13, 4],
            [1, 0, 0, 11, 8],
            [0, 3, 2, 1, 1],
        ],
        np.int64,
    )
    POSITIONS = np.array(
        [[1, 2], [8, 2], [3, 11], [-4, 21], [14, 6], [27, -1]], np.int64
    )

    @pytest.fixture
    def tiles(
        self,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        dtype: type[np.generic],
        channels: int,
    ) -> list[np.ndarray]:
        """Two tiles of different sizes for the crops to cut from."""
        return [
            random_image((13, 13) if channels == 1 else (13, 13, channels), dtype),
            random_image((8, 11) if channels == 1 else (8, 11, channels), dtype),
        ]

    def copied_pieces(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        return [
            tiles[index][y : y + height, x : x + width].copy()
            for index, x, y, width, height in self.CROPS
        ]

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("channels", [1, 3])
    def test_crops_equal_copied_pieces(self, tiles: list[np.ndarray]) -> None:
        """Placing crops of tiles equals placing copies of the pieces."""
        # Arrange
        shape = (24, 28, *tiles[0].shape[2:])
        expected = compose(
            np.zeros(shape, tiles[0].dtype),
            self.copied_pieces(tiles),
            self.POSITIONS,
        )

        # Act
        composed = compose(
            np.zeros(shape, tiles[0].dtype),
            tiles,
            self.POSITIONS,
            crops=self.CROPS,
        )

        # Assert
        np.testing.assert_array_equal(composed, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("channels", [1, 3])
    def test_slices_equal_copied_pieces(self, tiles: list[np.ndarray]) -> None:
        """Row slices of tiles, passed as tiles, compose as their copies."""
        # Arrange
        shape = (24, 28, *tiles[0].shape[2:])
        slices = [
            tiles[index][y : y + height, x : x + width]
            for index, x, y, width, height in self.CROPS
        ]
        expected = compose(
            np.zeros(shape, tiles[0].dtype),
            self.copied_pieces(tiles),
            self.POSITIONS,
        )

        # Act
        composed = compose(np.zeros(shape, tiles[0].dtype), slices, self.POSITIONS)

        # Assert
        np.testing.assert_array_equal(composed, expected)


@pytest.mark.unittest
class TestComposeBorrowing:
    """Buffer sharing raises a catchable ValueError, never a PanicException.

    Sharing between canvas and tiles, or with another thread, never raises an
    AlreadyBorrowed PanicException (a BaseException that ``except Exception``
    cannot catch).
    """

    @pytest.fixture
    def canvas(self) -> np.ndarray:
        return np.arange(100, dtype=np.uint8).reshape(10, 10).copy()

    @pytest.fixture
    def canvas_being_composed(
        self, random_image: Callable[[tuple[int, ...], Any], np.ndarray]
    ) -> Iterator[np.ndarray]:
        """Yield a canvas another thread composes onto, repeatedly, until teardown."""
        canvas = np.zeros((2048, 2048, 3), np.uint8)
        tile = random_image((2046, 2046, 3), np.uint8)
        stop = threading.Event()

        def compose_until_stopped() -> None:
            while not stop.is_set():
                compose(canvas, [tile], np.array([[1, 1]], np.int64))

        thread = threading.Thread(target=compose_until_stopped)
        thread.start()
        try:
            yield canvas
        finally:
            stop.set()
            thread.join()

    @pytest.mark.parametrize(
        ["alias", "position"],
        [
            (lambda canvas: canvas, np.array([[2, 2]], np.int64)),
            (lambda canvas: canvas[2:6], np.array([[5, 5]], np.int64)),
        ],
        ids=["canvas_itself", "contiguous_view"],
    )
    def test_aliased_tile_raises_value_error(
        self,
        canvas: np.ndarray,
        alias: Callable[[np.ndarray], np.ndarray],
        position: np.ndarray,
    ) -> None:
        """A tile sharing the canvas's buffer raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="a tile shares memory with the canvas"):
            compose(canvas, [alias(canvas)], position)

    def test_non_contiguous_view_of_canvas_raises_value_error(
        self, canvas: np.ndarray
    ) -> None:
        """A non-contiguous view is no longer copied, so it still raises ValueError.

        It shares the canvas's buffer.
        """
        # Act & Assert
        with pytest.raises(ValueError, match="a tile shares memory with the canvas"):
            compose(canvas, [canvas[0:4, 0:4]], np.array([[5, 5]], np.int64))

    @pytest.mark.parametrize(
        "read",
        [
            lambda canvas: downscale(canvas, 2, ResizeFilter.BOX),
            lambda canvas: resize(canvas, 64, 64, ResizeFilter.BILINEAR),
        ],
        ids=["downscale", "resize"],
    )
    def test_read_during_concurrent_compose_raises_value_error_at_worst(
        self,
        canvas_being_composed: np.ndarray,
        read: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Another thread reading the canvas mid-compose succeeds or raises.

        Compose releases the GIL while holding the canvas mutably borrowed, so
        another thread may read that canvas mid-compose: that succeeds or raises
        ValueError.
        """
        # Act & Assert
        for _ in range(100):
            with contextlib.suppress(ValueError):
                read(canvas_being_composed)
