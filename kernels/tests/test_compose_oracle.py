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

"""Tests of compose against an independent numpy oracle.

The oracle catches bugs the SIMD and scalar paths share (parity tests cannot
see those):

- bilinear: the direct four-weight formula with edge-clamped taps;
- bicubic: Catmull-Rom reproduces any linear field exactly in the interior.

Swept over positions covering integer, fractional, mixed and negative cases.
"""

import itertools
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pytest
from wsidicom_kernels import Interpolation, compose

TILE_WIDTH, TILE_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT = 9, 7, 14, 13
POSITIONS = [-2.0, -1.5, -0.25, 0.0, 0.25, 1.0, 2.5, 3.0, 4.75]
FRACTIONAL_POSITIONS = [
    (px, py)
    for px, py in itertools.product(POSITIONS, POSITIONS)
    if not (px == int(px) and py == int(py))  # integer positions only copy
]


class WrittenRegion(NamedTuple):
    """Return the canvas region (inclusive bounds) a tile writes, and its start.

    The start is the canvas column and row of the tile's first resampled sample.
    """

    x_first: int
    x_last: int
    y_first: int
    y_last: int
    tile_x: int
    tile_y: int


@pytest.fixture
def written_region(px: float, py: float) -> WrittenRegion:
    """Return the region a tile at an indirect ``(px, py)`` param writes."""
    tile_x, tile_y = math.ceil(px), math.ceil(py)
    return WrittenRegion(
        x_first=max(tile_x, 0),
        x_last=min(math.floor(px + TILE_WIDTH - 1), CANVAS_WIDTH - 1),
        y_first=max(tile_y, 0),
        y_last=min(math.floor(py + TILE_HEIGHT - 1), CANVAS_HEIGHT - 1),
        tile_x=tile_x,
        tile_y=tile_y,
    )


@pytest.fixture
def tile(
    random_image: Callable[[tuple[int, ...], Any], np.ndarray],
    dtype: type[np.generic],
) -> np.ndarray:
    """Make a random tile of an indirect ``dtype`` param."""
    return random_image((TILE_HEIGHT, TILE_WIDTH), dtype)


@pytest.fixture
def bilinear_reference(
    tile: np.ndarray, px: float, py: float, written_region: WrittenRegion
) -> np.ndarray:
    """Compute the direct four-weight bilinear formula over the written region.

    The taps are edge-clamped.
    """
    fx, fy = math.ceil(px) - px, math.ceil(py) - py
    samples = tile.astype(np.float64)
    region = written_region
    reference = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
    for cy in range(region.y_first, region.y_last + 1):
        ty = cy - region.tile_y
        ty_next = min(ty + 1, TILE_HEIGHT - 1)
        for cx in range(region.x_first, region.x_last + 1):
            tx = cx - region.tile_x
            tx_next = min(tx + 1, TILE_WIDTH - 1)
            reference[cy, cx] = (
                (1 - fx) * (1 - fy) * samples[ty, tx]
                + fx * (1 - fy) * samples[ty, tx_next]
                + (1 - fx) * fy * samples[ty_next, tx]
                + fx * fy * samples[ty_next, tx_next]
            )
    return reference[
        region.y_first : region.y_last + 1, region.x_first : region.x_last + 1
    ]


@pytest.mark.unittest
class TestComposeOracle:
    @pytest.mark.parametrize(["px", "py"], FRACTIONAL_POSITIONS)
    @pytest.mark.parametrize(
        ["dtype", "tolerance"], [(np.uint8, 1.0), (np.float32, 1e-4)]
    )
    def test_bilinear_matches_direct_formula(
        self,
        tile: np.ndarray,
        px: float,
        py: float,
        dtype: type[np.generic],
        tolerance: float,
        written_region: WrittenRegion,
        bilinear_reference: np.ndarray,
    ) -> None:
        """Bilinear compose equals the direct formula within rounding."""
        # Arrange
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype)
        region = written_region

        # Act
        compose(canvas, [tile], np.array([[px, py]]), Interpolation.BILINEAR)

        # Assert
        written = canvas[
            region.y_first : region.y_last + 1, region.x_first : region.x_last + 1
        ].astype(np.float64)
        assert written.size
        assert np.abs(written - bilinear_reference).max() <= tolerance

    @pytest.mark.parametrize(["px", "py"], FRACTIONAL_POSITIONS)
    def test_bicubic_reproduces_linear_field(
        self, px: float, py: float, written_region: WrittenRegion
    ) -> None:
        """Bicubic compose of a linear field is that field, away from the border.

        The tile's border is clamped, so not linear.
        """
        # Arrange
        yy, xx = np.mgrid[0:TILE_HEIGHT, 0:TILE_WIDTH]
        linear = (2.0 * xx + 3.0 * yy + 5.0).astype(np.float32)
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), np.float32)
        region = written_region
        y_first = max(region.y_first, region.tile_y + 1)
        y_last = min(region.y_last, region.tile_y + TILE_HEIGHT - 3)
        x_first = max(region.x_first, region.tile_x + 1)
        x_last = min(region.x_last, region.tile_x + TILE_WIDTH - 3)
        cy, cx = np.mgrid[y_first : y_last + 1, x_first : x_last + 1]
        expected = 2.0 * (cx - px) + 3.0 * (cy - py) + 5.0

        # Act
        compose(canvas, [linear], np.array([[px, py]]), Interpolation.BICUBIC)

        # Assert
        interior = canvas[y_first : y_last + 1, x_first : x_last + 1]
        assert (np.abs(interior - expected) <= 2e-3).all()
