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

Integer saturation: all-max images come out all-max (no wraparound in any sum
or rounding), and max/min checkerboards keep SIMD/scalar parity.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale

ParityCheck = Callable[[Callable[[], np.ndarray], int], np.ndarray]

INTEGER_MAX = [(np.uint8, 255), (np.uint16, 65535)]
BOX_SHAPES = [(33, 35), (33, 35, 3)]


@pytest.fixture
def assert_parity(
    simd_and_scalar: Callable[[Callable[[], Any]], tuple[Any, Any]],
) -> ParityCheck:
    """Assert a call agrees between the SIMD kernels and the scalar reference.

    Returns the SIMD result. The samples agree within the given tolerance.
    """

    def check(call: Callable[[], np.ndarray], tolerance: int) -> np.ndarray:
        simd, scalar = simd_and_scalar(call)
        difference = np.abs(simd.astype(np.int64) - scalar.astype(np.int64))
        assert (difference <= tolerance).all()
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
