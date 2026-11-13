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

"""Tests that compose keeps no state between or within calls.

- compose(canvas, [t1..tn], [p1..pn]) equals the single-tile calls in turn,
  bit for bit, overlaps included, on every path;
- other calls in between change nothing;
- toggling the SIMD kernels leaves no residue.

Each test draws ``TRIALS`` random tile sets.
"""

import contextlib
from collections.abc import Callable

import numpy as np
import pytest
from wsidicom_kernels import (
    ResizeFilter,
    compose,
    downscale,
    resize,
)
from wsidicom_kernels._kernels import set_simd

TRIALS = 50

TilesAndPositions = tuple[list[np.ndarray], np.ndarray]


@pytest.fixture(params=[True, False], ids=["simd", "scalar"])
def simd_enabled(request: pytest.FixtureRequest) -> None:
    """Run the test with the SIMD kernels, and with the scalar reference."""
    set_simd(request.param)


@pytest.fixture
def canvas_shape(canvas_samples: int) -> tuple[int, ...]:
    """Return the canvas shape for an indirect ``canvas_samples`` param."""
    return (26, 27) if canvas_samples == 1 else (26, 27, canvas_samples)


@pytest.fixture
def random_tiles_and_positions(
    rng: np.random.Generator, tile_samples: int
) -> Callable[[], TilesAndPositions]:
    """Return a factory of 2 to 5 random uint8 tiles and their positions.

    The tiles have 2 to 29 rows and columns and ``tile_samples`` samples, at
    positions (an ``(N, 2)`` int64 array) that may fall partly outside the
    canvas and overlap each other.
    """

    def create() -> TilesAndPositions:
        tiles: list[np.ndarray] = []
        positions: list[list[int]] = []
        for _ in range(int(rng.integers(2, 6))):
            height, width = int(rng.integers(2, 30)), int(rng.integers(2, 30))
            shape = (
                (height, width) if tile_samples == 1 else (height, width, tile_samples)
            )
            tiles.append(rng.integers(0, 256, shape).astype(np.uint8))
            positions.append([int(rng.integers(-3, 20)) for _ in range(2)])
        return tiles, np.array(positions, np.int64)

    return create


@pytest.mark.unittest
@pytest.mark.parametrize(
    ["tile_samples", "canvas_samples"],
    [(1, 1), (3, 3)],
    ids=["planar", "rgb"],
)
class TestComposeIsStateless:
    @pytest.mark.usefixtures("simd_enabled")
    def test_batched_equals_single_tile_calls(
        self,
        rng: np.random.Generator,
        canvas_shape: tuple[int, ...],
        random_tiles_and_positions: Callable[[], TilesAndPositions],
    ) -> None:
        """One call with all tiles equals one call per tile in turn."""
        for _ in range(TRIALS):
            # Arrange
            tiles, positions = random_tiles_and_positions()
            batched = rng.integers(0, 256, canvas_shape).astype(np.uint8)
            one_by_one = batched.copy()

            # Act
            compose(batched, tiles, positions)
            for tile, position in zip(tiles, positions, strict=True):
                compose(one_by_one, [tile], position[np.newaxis])

            # Assert
            assert (batched == one_by_one).all()

    @pytest.mark.usefixtures("simd_enabled")
    def test_other_calls_in_between_change_nothing(
        self,
        rng: np.random.Generator,
        canvas_shape: tuple[int, ...],
        random_tiles_and_positions: Callable[[], TilesAndPositions],
    ) -> None:
        """Downscale and resize calls between composes leave no trace."""
        for _ in range(TRIALS):
            # Arrange
            tiles, positions = random_tiles_and_positions()
            alone = np.zeros(canvas_shape, np.uint8)
            interleaved = alone.copy()

            # Act
            for tile, position in zip(tiles, positions, strict=True):
                compose(alone, [tile], position[np.newaxis])
            for tile, position in zip(tiles, positions, strict=True):
                downscale(
                    rng.integers(0, 256, (17, 19)).astype(np.uint8), 2, ResizeFilter.BOX
                )
                resize(
                    rng.integers(0, 256, (5, 7, 3)).astype(np.uint8),
                    3,
                    4,
                    ResizeFilter.BILINEAR,
                )
                compose(interleaved, [tile], position[np.newaxis])

            # Assert
            assert (alone == interleaved).all()

    def test_toggling_simd_leaves_no_residue(
        self,
        scalar_kernels: Callable[[], contextlib.AbstractContextManager[None]],
        canvas_shape: tuple[int, ...],
        random_tiles_and_positions: Callable[[], TilesAndPositions],
    ) -> None:
        """A scalar compose after a SIMD compose equals a scalar-only compose."""
        for _ in range(TRIALS):
            # Arrange
            tiles, positions = random_tiles_and_positions()
            after_simd = np.zeros(canvas_shape, np.uint8)
            scalar_only = np.zeros(canvas_shape, np.uint8)

            # Act
            compose(np.zeros(canvas_shape, np.uint8), tiles, positions)
            with scalar_kernels():
                compose(after_simd, tiles, positions)
                compose(scalar_only, tiles, positions)

            # Assert
            assert (after_simd == scalar_only).all()
