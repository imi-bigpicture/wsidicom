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

"""Tests for boundary values.

- Integer saturation: all-max images come out all-max (no wraparound in any
  sum or rounding), and max/min checkerboards keep SIMD/scalar parity.
- Float32 specials (inf, NaN, denormal, near overflow) neither crash nor break
  parity.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import (
    Interpolation,
    ResizeFilter,
    compose,
    downscale,
    rgb_to_rgbx,
)

ParityCheck = Callable[[Callable[[], np.ndarray], float], np.ndarray]

INTEGER_MAX = [(np.uint8, 255), (np.uint16, 65535)]
FLOAT_SPECIALS = [np.inf, -np.inf, np.nan, 3.0e38, -3.0e38, 1e-40, -0.0, 0.0]
BOX_SHAPES = [(33, 35), (33, 35, 3), (33, 35, 4)]


@pytest.fixture
def assert_parity(
    simd_and_scalar: Callable[[Callable[[], Any]], tuple[Any, Any]],
) -> ParityCheck:
    """Assert a call agrees between the SIMD kernels and the scalar reference.

    Returns the SIMD result.
    The same samples must be non-finite, and finite ones agree within the given
    tolerance plus 1e-6 relative (one FMA ulp near 3e38 is ~2e31 absolute).
    """

    def check(call: Callable[[], np.ndarray], tolerance: float) -> np.ndarray:
        simd, scalar = (result.astype(np.float64) for result in simd_and_scalar(call))
        assert (np.isfinite(simd) == np.isfinite(scalar)).all()
        finite = np.isfinite(simd)
        difference = np.abs(simd[finite] - scalar[finite])
        scale = np.maximum(np.abs(simd[finite]), np.abs(scalar[finite]))
        assert (difference <= tolerance + 1e-6 * scale).all()
        return simd

    return check


@pytest.mark.unittest
class TestIntegerSaturation:
    """Integer kernels saturate instead of wrapping around."""

    @pytest.mark.parametrize("factor", [2, 4])
    @pytest.mark.parametrize("shape", BOX_SHAPES)
    @pytest.mark.parametrize(["dtype", "top"], INTEGER_MAX)
    def test_box_all_max_stays_max(
        self,
        assert_parity: ParityCheck,
        dtype: type[np.unsignedinteger],
        top: int,
        shape: tuple[int, ...],
        factor: int,
    ) -> None:
        """A box downscale of an all-max image is all-max."""
        # Arrange
        all_max = np.full(shape, top, dtype)

        # Act
        result = assert_parity(lambda: downscale(all_max, factor, ResizeFilter.BOX), 1)

        # Assert
        assert (result == top).all()

    @pytest.mark.parametrize("shape", BOX_SHAPES)
    @pytest.mark.parametrize(["dtype", "top"], INTEGER_MAX)
    def test_box_max_min_checkerboard_keeps_parity(
        self,
        assert_parity: ParityCheck,
        dtype: type[np.unsignedinteger],
        top: int,
        shape: tuple[int, ...],
    ) -> None:
        """A box downscale of a max/min checkerboard matches the scalar reference."""
        # Arrange
        rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
        checkerboard = (((columns + rows) % 2) * top).astype(dtype)
        if len(shape) == 3:
            checkerboard = np.repeat(checkerboard[:, :, None], shape[2], axis=2)

        # Act & Assert
        assert_parity(lambda: downscale(checkerboard, 2, ResizeFilter.BOX), 1)

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    @pytest.mark.parametrize(
        ["dtype", "top", "tile_shape", "canvas_shape"],
        [
            (np.uint8, 255, (9, 11), (16, 16)),
            (np.uint8, 255, (9, 11, 3), (16, 16, 3)),
            (np.uint8, 255, (9, 11, 3), (16, 16, 4)),
            (np.uint16, 65535, (9, 11), (16, 16)),
            (np.uint16, 65535, (9, 11, 3), (16, 16, 3)),
        ],
    )
    def test_compose_all_max_stays_max(
        self,
        dtype: type[np.unsignedinteger],
        top: int,
        tile_shape: tuple[int, ...],
        canvas_shape: tuple[int, ...],
        interpolation: Interpolation,
    ) -> None:
        """An all-max tile composes within one of max where it covers fully.

        Onto a zero canvas, at (2.5, 1.25): rows 1 to 10 and columns 2 to 13
        reach the tile, rows 3 to 8 and columns 4 to 11 lie two pixels clear of
        its edges. The RGBX canvas's pad channel is not a written sample.
        """
        # Arrange
        tile = np.full(tile_shape, top, dtype)
        canvas = np.zeros(canvas_shape, dtype)

        # Act
        compose(canvas, [tile], np.array([[2.5, 1.25]]), interpolation)

        # Assert
        samples = canvas[:, :, :3] if canvas_shape[-1] == 4 else canvas
        covered = samples[3:9, 4:12].astype(np.int64)
        assert (np.abs(covered - top) <= 1).all()

    def test_rgb_to_rgbx_all_max(self) -> None:
        """Packing an all-max RGB image with an all-max pad is all-max."""
        # Act
        result = rgb_to_rgbx(np.full((7, 9, 3), 255, np.uint8), 255)

        # Assert
        assert (result == 255).all()


@pytest.mark.unittest
@pytest.mark.parametrize("special", FLOAT_SPECIALS)
class TestFloatSpecials:
    """A float32 special value in the image keeps SIMD/scalar parity."""

    @pytest.mark.parametrize(
        ["shape", "special_index", "factor"],
        [
            ((33, 35), (5, 7), 2),
            ((33, 35), (5, 7), 4),
            ((33, 35, 3), (5, 7, 1), 2),
        ],
    )
    def test_box_keeps_parity(
        self,
        assert_parity: ParityCheck,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        special: float,
        shape: tuple[int, ...],
        special_index: tuple[int, ...],
        factor: int,
    ) -> None:
        """A box downscale with a special sample matches the scalar reference."""
        # Arrange
        image = random_image(shape, np.float32)
        image[special_index] = special

        # Act & Assert
        assert_parity(lambda: downscale(image, factor, ResizeFilter.BOX), 1e-3)

    @pytest.mark.parametrize(
        "interpolation", [Interpolation.BILINEAR, Interpolation.BICUBIC]
    )
    def test_compose_keeps_parity(
        self,
        assert_parity: ParityCheck,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        special: float,
        interpolation: Interpolation,
    ) -> None:
        """Composing a tile with a special sample matches the scalar reference."""
        # Arrange
        tile = random_image((9, 11), np.float32)
        tile[3, 4] = special

        def composed() -> np.ndarray:
            canvas = np.zeros((16, 16), np.float32)
            compose(canvas, [tile], np.array([[2.5, 1.25]]), interpolation)
            return canvas

        # Act & Assert
        assert_parity(composed, 1e-2)


@pytest.mark.unittest
class TestDenormals:
    """Denormal float32 images are handled exactly."""

    def test_all_denormal_box_is_exact(self, assert_parity: ParityCheck) -> None:
        """A box downscale of an all-denormal image equals the scalar reference."""
        # Arrange
        denormal = np.full((33, 35), 1e-40, np.float32)

        # Act & Assert
        assert_parity(lambda: downscale(denormal, 2, ResizeFilter.BOX), 0.0)
