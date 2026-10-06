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

Integer positions copy tiles onto the canvas; fractional positions resample
them (bilinear or bicubic), planar or interleaved, and a 4-sample canvas takes
3-sample tiles by expanding them as it writes.
"""

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import (
    Interpolation,
    ResizeFilter,
    compose,
    downscale,
    resize,
    rgb_to_rgbx,
    stitch,
)

DTYPES = [np.uint8, np.uint16, np.float32]
# An integer position on ONE axis (fractional on the other) still resamples, with
# the integer axis's second tap carrying weight exactly 0. That tap must be
# clamped (x: step=0, y: row clamp) or it reads past the tile.
MIXED_INTEGER_FRACTIONAL_POSITIONS = [[2.0, 1.5], [1.5, 2.0], [0.0, 0.5], [0.5, 0.0]]


def fully_covered(
    position: list[float], tile_shape: tuple[int, ...]
) -> tuple[slice, slice]:
    """Rows and columns a tile at ``position`` covers fully with any filter.

    A whole axis at an integer position; else two pixels in from each end,
    clear of bicubic's partly covered edge pixels.
    """

    def axis(start: float, length: int) -> slice:
        if start == int(start):
            return slice(int(start), int(start) + length)
        return slice(int(np.floor(start)) + 2, int(np.floor(start)) + length - 1)

    return axis(position[1], tile_shape[0]), axis(position[0], tile_shape[1])


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
    def test_bilinear_is_identity(self) -> None:
        """An integer position copies the tile even with a resampling filter."""
        # Arrange
        tile = np.arange(16, dtype=np.float32).reshape(4, 4)
        canvas = np.zeros((4, 4), np.float32)

        # Act
        compose(canvas, [tile], np.array([[0, 0]], np.float64), Interpolation.BILINEAR)

        # Assert
        assert np.allclose(canvas, tile)

    def test_interleaved_tiles_on_grid(self) -> None:
        """Interleaved RGB tiles are placed on an integer grid."""
        # Arrange
        tiles = [
            np.full((4, 4, 3), (index, index + 1, index + 2), np.uint8)
            for index in range(6)
        ]
        canvas = np.zeros((8, 12, 3), np.uint8)
        positions = np.array(
            [[(index % 3) * 4, (index // 3) * 4] for index in range(6)], np.float64
        )

        # Act
        compose(canvas, tiles, positions, Interpolation.NEAREST)

        # Assert
        assert (canvas[0:4, 0:4] == (0, 1, 2)).all()
        assert (canvas[4:8, 8:12] == (5, 6, 7)).all()

    def test_later_tiles_overwrite_earlier(self) -> None:
        """Where tiles overlap, the last one wins."""
        # Arrange
        first = np.full((4, 4), 1, np.uint8)
        second = np.full((4, 4), 2, np.uint8)
        canvas = np.zeros((4, 4), np.uint8)

        # Act
        compose(canvas, [first, second], np.zeros((2, 2)), Interpolation.NEAREST)

        # Assert
        assert (canvas == 2).all()


@pytest.mark.unittest
class TestComposeAtFractionalPositions:
    def test_bilinear_constant_shift(self) -> None:
        """A constant field stays constant under sub-pixel translation."""
        # Arrange
        tile = np.full((5, 5), 100.0, np.float32)
        canvas = np.zeros((6, 6), np.float32)

        # Act
        compose(canvas, [tile], np.array([[0.5, 0.5]]), Interpolation.BILINEAR)

        # Assert
        assert np.allclose(canvas[1:5, 1:5], 100.0)

    def test_bicubic_runs_and_stays_in_range(self, rng: np.random.Generator) -> None:
        """Bicubic resampling of a 16-bit tile stays within the dtype's range."""
        # Arrange
        tile = rng.integers(0, 256, (8, 8)).astype(np.uint16)
        canvas = np.zeros((8, 8), np.uint16)

        # Act
        compose(canvas, [tile], np.array([[1.3, 2.7]]), Interpolation.BICUBIC)

        # Assert
        assert canvas.max() <= 65535

    def test_bilinear_no_channel_bleed(self) -> None:
        """Interleaved resampling must not mix channels.

        A per-channel constant RGB tile stays constant per channel under a
        sub-pixel shift.
        """
        # Arrange
        channel_values = (10.0, 20.0, 30.0)
        tile = np.empty((5, 5, 3), np.float32)
        tile[...] = channel_values
        canvas = np.zeros((6, 6, 3), np.float32)

        # Act
        compose(canvas, [tile], np.array([[0.5, 0.5]]), Interpolation.BILINEAR)

        # Assert
        for channel, value in enumerate(channel_values):
            assert np.allclose(canvas[1:5, 1:5, channel], value)

    def test_bilinear_matches_reference(
        self, random_image: Callable[[tuple[int, ...], Any], np.ndarray]
    ) -> None:
        """Every pixel equals the plain four-weight formula.

        A tap outside the tile reads the canvas (zero here), so the pixels the
        tile covers only partly blend it into the canvas by coverage.
        """
        # Arrange
        tile = random_image((7, 9), np.float32)
        tile_height, tile_width = tile.shape
        px, py = 3.25, 2.7
        canvas = np.zeros((20, 20), np.float32)
        fx, fy = np.ceil(px) - px, np.ceil(py) - py
        x_start, y_start = int(np.ceil(px)), int(np.ceil(py))

        def sample(y: int, x: int) -> float:
            inside = 0 <= y < tile_height and 0 <= x < tile_width
            return float(tile[y, x]) if inside else 0.0

        expected = np.zeros_like(canvas)
        for cy in range(canvas.shape[0]):
            y0 = cy - y_start
            for cx in range(canvas.shape[1]):
                x0 = cx - x_start
                expected[cy, cx] = (
                    (1 - fx) * (1 - fy) * sample(y0, x0)
                    + fx * (1 - fy) * sample(y0, x0 + 1)
                    + (1 - fx) * fy * sample(y0 + 1, x0)
                    + fx * fy * sample(y0 + 1, x0 + 1)
                )

        # Act
        compose(canvas, [tile], np.array([[px, py]]), Interpolation.BILINEAR)

        # Assert
        assert np.allclose(canvas, expected, atol=1e-5)

    def test_bicubic_reproduces_linear(self) -> None:
        """Catmull-Rom (a=-0.5) reproduces a linear field exactly in the interior.

        The clamped border frame is not linear.
        """
        # Arrange
        tile_height, tile_width = 9, 11
        yy, xx = np.mgrid[0:tile_height, 0:tile_width]
        tile = (2.0 * xx + 3.0 * yy + 5.0).astype(np.float32)
        px, py = 4.2, 3.6
        canvas = np.zeros((20, 20), np.float32)
        x_start, y_start = int(np.ceil(px)), int(np.ceil(py))
        x_end = int(np.floor(px + tile_width - 1))
        y_end = int(np.floor(py + tile_height - 1))
        interior_y, interior_x = np.mgrid[y_start + 1 : y_end, x_start + 1 : x_end]
        expected = 2.0 * (interior_x - px) + 3.0 * (interior_y - py) + 5.0

        # Act
        compose(canvas, [tile], np.array([[px, py]]), Interpolation.BICUBIC)

        # Assert
        interior = canvas[y_start + 1 : y_end, x_start + 1 : x_end]
        assert (np.abs(interior - expected) < 1e-3).all()

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize("tile_width", [1, 2, 5])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize(
        ["tile_samples", "canvas_samples"], [(1, 1), (3, 3), (3, 4)]
    )
    def test_just_below_an_integer_composes_as_the_integer(
        self,
        interpolation: Interpolation,
        tile_width: int,
        axis: int,
        tile_samples: int,
        canvas_samples: int,
    ) -> None:
        """A position one ulp below an integer composes as that integer.

        The tile covers exactly its width from there, though
        ``position + width`` rounds up to the next integer, and no tap reads
        past it.
        """
        # Arrange
        size = [4, 4]
        size[1 - axis] = tile_width
        tile_shape = (*size, tile_samples) if tile_samples > 1 else tuple(size)
        tile = np.full(tile_shape, 200, np.uint8)
        canvas_shape = (12, 12, canvas_samples) if canvas_samples > 1 else (12, 12)
        canvas = np.zeros(canvas_shape, np.uint8)
        just_below = float(np.nextafter(3.0, 0.0))
        position = [3.0, 3.0]
        position[axis] = just_below

        # Act
        compose(canvas, [tile], np.array([position]), interpolation)

        # Assert — along the fractional axis, tile_width pixels from 3
        first = canvas.reshape(12, 12, -1)[..., 0]
        extent = (first != 0).any(axis=axis).nonzero()[0]
        assert extent.tolist() == list(range(3, 3 + tile_width))
        assert (np.abs(first[first != 0].astype(int) - 200) <= 1).all()

    @pytest.mark.parametrize("position", [[7.5, 0.0], [0.0, 7.5], [7.5, 3.25]])
    @pytest.mark.parametrize(
        ["canvas_shape", "pad_argument"],
        [((64, 64, 3), {}), ((64, 64, 4), {"pad": 255})],
    )
    def test_bicubic_mixed_integer_fractional_axis(
        self,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        position: list[float],
        canvas_shape: tuple[int, ...],
        pad_argument: dict[str, int],
    ) -> None:
        """A position integer on one axis resamples with ``x_start == px``.

        Not ``floor(px) + 1``, on that axis; the bicubic 4x4 neighbourhood must
        still stay inside the tile.
        """
        # Arrange
        tile = random_image((32, 40, 3), np.uint8)
        canvas = np.zeros(canvas_shape, np.uint8)

        # Act
        composed = compose(
            canvas,
            [tile],
            np.array([position]),
            Interpolation.BICUBIC,
            pad=pad_argument.get("pad", 255),
        )

        # Assert — the shape is kept and something was actually written
        assert composed.shape == canvas_shape
        assert composed[..., :3].any()

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize("position", MIXED_INTEGER_FRACTIONAL_POSITIONS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mixed_integer_fractional_axes_planar(
        self,
        interpolation: Interpolation,
        position: list[float],
        dtype: type[np.generic],
    ) -> None:
        """A planar constant field comes through unchanged where the tile covers.

        At a mixed integer/fractional position; the pixels it covers only
        partly blend it into the zero canvas, never past the field.
        """
        # Arrange
        tile = np.full((7, 9), 100, dtype)
        canvas = np.zeros((14, 14), dtype)

        # Act
        compose(canvas, [tile], np.array([position]), interpolation)

        # Assert
        covered = canvas[fully_covered(position, tile.shape)]
        assert covered.size
        assert (np.abs(covered.astype(float) - 100) <= 1).all()
        assert (canvas.astype(float) <= 101).all()

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize("position", MIXED_INTEGER_FRACTIONAL_POSITIONS)
    @pytest.mark.parametrize("canvas_samples", [3, 4])
    def test_mixed_integer_fractional_axes_interleaved(
        self, interpolation: Interpolation, position: list[float], canvas_samples: int
    ) -> None:
        """An interleaved RGB constant field survives a fractional position.

        A mixed integer/fractional position, onto an RGB canvas and onto an RGBX
        canvas (the fused expand path): unchanged where the tile covers, never
        past it where the tile blends into the zero canvas.
        """
        # Arrange
        color = (40, 80, 120)
        tile = np.empty((7, 9, 3), np.uint8)
        tile[...] = color
        canvas = np.zeros((14, 14, canvas_samples), np.uint8)

        # Act
        compose(canvas, [tile], np.array([position]), interpolation)

        # Assert
        rgb = canvas[..., :3].astype(int)
        covered = rgb[fully_covered(position, tile.shape)]
        assert covered.size
        assert (np.abs(covered - color) <= 1).all()
        assert (rgb <= np.array(color) + 1).all()

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("tile_shape", [(23, 29, 3)])
    def test_interleaved_matches_per_channel_planar(
        self,
        interpolation: Interpolation,
        dtype: type[np.generic],
        random_tile: np.ndarray,
    ) -> None:
        """Composing an interleaved tile equals composing each channel as planar.

        The interleaved kernels are the planar kernels with tap offset c.
        """
        # Arrange
        position = np.array([[2.4, 1.8]])
        canvas = np.zeros((30, 34, 3), dtype)

        # Act
        compose(canvas, [random_tile], position, interpolation)

        # Assert — not bit-exact: the SIMD body uses FMA and the scalar tail does
        # not, and the body/tail boundary falls on different elements for
        # interleaved (n*c) and planar (n). Same <=1 LSB latitude as the parity
        # test.
        for channel in range(3):
            planar = np.zeros((30, 34), dtype)
            planar_tile = np.ascontiguousarray(random_tile[:, :, channel])
            compose(planar, [planar_tile], position, interpolation)
            if np.issubdtype(dtype, np.integer):
                difference = np.abs(
                    canvas[:, :, channel].astype(int) - planar.astype(int)
                )
                assert difference.max() <= 1
            else:
                assert np.allclose(canvas[:, :, channel], planar, atol=1e-5)

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        ["tile_shape", "canvas_shape"],
        [((37, 41), (50, 50)), ((37, 41, 3), (50, 50, 3))],
    )
    def test_simd_matches_scalar(
        self,
        simd_and_scalar: Callable[[Callable[[], Any]], tuple[Any, Any]],
        interpolation: Interpolation,
        dtype: type[np.generic],
        canvas_shape: tuple[int, ...],
        random_tile: np.ndarray,
    ) -> None:
        """The SIMD kernels agree with the scalar reference, planar and interleaved.

        Odd sizes exercise both the vector body and the tail.
        """

        # Arrange
        def compose_onto_empty_canvas() -> np.ndarray:
            canvas = np.zeros(canvas_shape, dtype)
            compose(canvas, [random_tile], np.array([[3.3, 2.6]]), interpolation)
            return canvas

        # Act
        simd, scalar = simd_and_scalar(compose_onto_empty_canvas)

        # Assert — 1 LSB latitude for FMA and association order
        if np.issubdtype(dtype, np.integer):
            assert np.abs(simd.astype(int) - scalar.astype(int)).max() <= 1
        else:
            assert np.allclose(simd, scalar, atol=1e-4)


@pytest.mark.unittest
class TestComposeOntoRgbxCanvas:
    """A 4-sample canvas takes 3-sample tiles and expands them as it writes."""

    @pytest.fixture
    def rgb_tiles(
        self, random_image: Callable[[tuple[int, ...], Any], np.ndarray]
    ) -> list[np.ndarray]:
        """Odd-sized RGB tiles, so both the vector body and the tail run."""
        return [random_image((23, 29, 3), np.uint8) for _ in range(4)]

    @pytest.fixture
    def compose_fractional(
        self, rgb_tiles: list[np.ndarray]
    ) -> Callable[..., np.ndarray]:
        """Return a bilinear compose of the tiles at fractional offsets.

        It composes onto a new empty canvas of the given shape.
        """
        positions = np.array([[0.0, 0.0], [27.5, 0.0], [0.0, 21.5], [27.5, 21.5]])

        def run(canvas_shape: tuple[int, ...], pad: int = 255) -> np.ndarray:
            canvas = np.zeros(canvas_shape, np.uint8)
            return compose(
                canvas, rgb_tiles, positions, Interpolation.BILINEAR, pad=pad
            )

        return run

    def test_rgb_equals_rgb_canvas_compose(
        self, compose_fractional: Callable[..., np.ndarray]
    ) -> None:
        """The RGB samples equal an RGB compose exactly where tiles were written."""
        # Act
        rgb = compose_fractional((64, 64, 3))
        rgbx = compose_fractional((64, 64, 4), pad=255)

        # Assert
        covered = rgbx[..., 3] == 255
        assert covered.any()
        assert (rgbx[..., :3][covered] == rgb[covered]).all()

    def test_fourth_sample_is_pad(
        self, compose_fractional: Callable[..., np.ndarray]
    ) -> None:
        """The fourth sample is ``pad`` wherever tiles were written."""
        # Arrange
        covered = compose_fractional((64, 64, 4), pad=255)[..., 3] == 255

        # Act
        rgbx = compose_fractional((64, 64, 4), pad=9)

        # Assert
        assert covered.any()
        assert (rgbx[..., 3][covered] == 9).all()

    def test_simd_matches_scalar(
        self,
        compose_fractional: Callable[..., np.ndarray],
        simd_and_scalar: Callable[[Callable[[], Any]], tuple[Any, Any]],
    ) -> None:
        """The fused SIMD kernel agrees with the scalar fallback within 1 LSB."""
        # Act
        simd, scalar = simd_and_scalar(lambda: compose_fractional((64, 64, 4), pad=255))

        # Assert
        assert np.abs(simd.astype(int) - scalar.astype(int)).max() <= 1

    def test_integer_positions_expand_copy(self, rgb_tiles: list[np.ndarray]) -> None:
        """Integer positions take the expand-copy path."""
        # Arrange
        positions = np.array([[0.0, 0.0], [29.0, 0.0], [0.0, 23.0], [29.0, 23.0]])
        rgb_canvas = np.zeros((64, 64, 3), np.uint8)
        rgbx_canvas = np.zeros((64, 64, 4), np.uint8)

        # Act
        rgb = compose(rgb_canvas, rgb_tiles, positions, Interpolation.NEAREST)
        rgbx = compose(rgbx_canvas, rgb_tiles, positions, Interpolation.NEAREST, pad=1)

        # Assert
        assert (rgbx[..., :3] == rgb).all()
        assert (rgbx[:23, :29, 3] == 1).all()


@pytest.mark.unittest
class TestComposePiecesOfTiles:
    """Crops and row slices of tiles compose as copies of those pieces would.

    Pieces butt and overlap at fractional positions, so their edges blend
    with each other.
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
        [
            [1.25, 2.5],
            [8.25, 2.75],
            [3.5, 11.25],
            [1.0, 11.5],
            [14.75, 6.0],
            [20.0, 1.0],
        ]
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
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("interpolation", list(Interpolation))
    def test_crops_equal_copied_pieces(
        self, tiles: list[np.ndarray], interpolation: Interpolation
    ) -> None:
        """Placing crops of tiles equals placing copies of the pieces."""
        # Arrange
        shape = (24, 28, *tiles[0].shape[2:])
        expected = compose(
            np.zeros(shape, tiles[0].dtype),
            self.copied_pieces(tiles),
            self.POSITIONS,
            interpolation,
        )

        # Act
        composed = compose(
            np.zeros(shape, tiles[0].dtype),
            tiles,
            self.POSITIONS,
            interpolation,
            crops=self.CROPS,
        )

        # Assert
        np.testing.assert_array_equal(composed, expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("interpolation", list(Interpolation))
    def test_slices_equal_copied_pieces(
        self, tiles: list[np.ndarray], interpolation: Interpolation
    ) -> None:
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
            interpolation,
        )

        # Act
        composed = compose(
            np.zeros(shape, tiles[0].dtype), slices, self.POSITIONS, interpolation
        )

        # Assert
        np.testing.assert_array_equal(composed, expected)

    @pytest.mark.parametrize("dtype", [np.uint8])
    @pytest.mark.parametrize("channels", [3])
    @pytest.mark.parametrize("interpolation", list(Interpolation))
    def test_crops_onto_rgbx_canvas_equal_copied_pieces(
        self, tiles: list[np.ndarray], interpolation: Interpolation
    ) -> None:
        """An RGBX canvas takes crops of RGB tiles as it takes their copies."""
        # Arrange
        expected = compose(
            np.zeros((24, 28, 4), np.uint8),
            self.copied_pieces(tiles),
            self.POSITIONS,
            interpolation,
            pad=7,
        )

        # Act
        composed = compose(
            np.zeros((24, 28, 4), np.uint8),
            tiles,
            self.POSITIONS,
            interpolation,
            pad=7,
            crops=self.CROPS,
        )

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
                compose(canvas, [tile], np.array([[0.3, 0.7]]), Interpolation.BICUBIC)

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
            (lambda canvas: canvas, np.array([[2.5, 2.5]])),
            (lambda canvas: canvas[2:6], np.array([[5.5, 5.5]])),
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
            compose(canvas, [alias(canvas)], position, Interpolation.BILINEAR)

    def test_non_contiguous_view_of_canvas_raises_value_error(
        self, canvas: np.ndarray
    ) -> None:
        """A non-contiguous view is no longer copied, so it still raises ValueError.

        It shares the canvas's buffer.
        """
        # Act & Assert
        with pytest.raises(ValueError, match="a tile shares memory with the canvas"):
            compose(
                canvas,
                [canvas[0:4, 0:4]],
                np.array([[5.5, 5.5]]),
                Interpolation.BILINEAR,
            )

    @pytest.mark.parametrize(
        "read",
        [
            lambda canvas: downscale(canvas, 2, ResizeFilter.BOX),
            lambda canvas: resize(canvas, 64, 64, ResizeFilter.BILINEAR),
            lambda canvas: downscale(canvas, 2, ResizeFilter.BOX, pad=0),
            lambda canvas: rgb_to_rgbx(canvas),
        ],
        ids=["downscale", "resize", "downscale_pad", "rgb_to_rgbx"],
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
