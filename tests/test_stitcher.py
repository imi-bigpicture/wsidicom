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

"""Tests for the stitchers.

Every contract test runs against both implementations: ``NumpyStitcher`` (the
production slice-write stitcher) and ``PillowStitcher`` (the paste-based
reference). The two agreeing is itself the cross-check that the numpy stitcher
composes tiles correctly.
"""

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from wsidicom.geometry import Point, Region, Size
from wsidicom.stitcher import NumpyStitcher, PillowStitcher, Stitcher
from wsidicom.thread import ReadExecutor

TILE_SIZE = 4  # tile pixel size used in tests

# A 2x2 grid of distinct solid colors keyed by (column, row).
COLORS = {
    (0, 0): (255, 0, 0),
    (1, 0): (0, 255, 0),
    (0, 1): (0, 0, 255),
    (1, 1): (255, 255, 0),
}


def solid_tile(color: tuple[int, int, int], size: int = TILE_SIZE) -> np.ndarray:
    return np.full((size, size, 3), color, dtype=np.uint8)


def tile_slot(result: np.ndarray, column: int, row: int) -> np.ndarray:
    """Return the TILE_SIZE block at grid position (column, row)."""
    return result[
        row * TILE_SIZE : (row + 1) * TILE_SIZE,
        column * TILE_SIZE : (column + 1) * TILE_SIZE,
    ]


def fetch_from(tiles: dict[Point, np.ndarray]):
    """A fetch callable yielding one tile per requested point, in order."""

    def fetch(points: Sequence[Point]) -> Iterator[np.ndarray]:
        for point in points:
            yield tiles[point]

    return fetch


def full_pixel_region(tile_region: Region) -> Region:
    """The pixel region covering every tile in ``tile_region`` (no clipping)."""
    return Region(tile_region.start * TILE_SIZE, tile_region.size * TILE_SIZE)


@pytest.fixture(params=[NumpyStitcher, PillowStitcher], ids=["numpy", "pillow"])
def stitcher(request: pytest.FixtureRequest) -> Stitcher:
    return request.param()


@pytest.fixture
def single_thread_executor() -> ReadExecutor:
    """An inline, single-thread read executor."""
    return ReadExecutor(1, None)


@pytest.fixture
def parallel_executor() -> ReadExecutor:
    """A multi-worker read executor, so the region is split into chunks."""
    return ReadExecutor(4, None)


@pytest.mark.unittest
class TestStitchGrid:
    def test_dimensions(self, stitcher: Stitcher):
        """Composed shape = grid dimensions times tile size."""
        # Arrange
        grid = [[solid_tile((255, 0, 0)) for _ in range(3)] for _ in range(2)]

        # Act
        result = stitcher.stitch_grid(grid)

        # Assert
        assert result.shape == (2 * TILE_SIZE, 3 * TILE_SIZE, 3)

    def test_row_major_layout(self, stitcher: Stitcher):
        """Tiles land at their row-major grid positions."""
        # Arrange
        grid = [
            [solid_tile(COLORS[(0, 0)]), solid_tile(COLORS[(1, 0)])],
            [solid_tile(COLORS[(0, 1)]), solid_tile(COLORS[(1, 1)])],
        ]

        # Act
        result = stitcher.stitch_grid(grid)

        # Assert
        for (column, row), color in COLORS.items():
            assert np.array_equal(tile_slot(result, column, row), solid_tile(color))

    def test_variable_edge_blocks(self, stitcher: Stitcher):
        """A rectangular grid with a narrower last column and shorter last row."""
        # Arrange - column widths [4, 2], row heights [4, 3]
        grid = [
            [np.full((4, 4, 3), 1, np.uint8), np.full((4, 2, 3), 2, np.uint8)],
            [np.full((3, 4, 3), 3, np.uint8), np.full((3, 2, 3), 4, np.uint8)],
        ]

        # Act
        result = stitcher.stitch_grid(grid)

        # Assert
        assert result.shape == (4 + 3, 4 + 2, 3)
        assert np.array_equal(result[:4, :4], np.full((4, 4, 3), 1, np.uint8))
        assert np.array_equal(result[:4, 4:], np.full((4, 2, 3), 2, np.uint8))
        assert np.array_equal(result[4:, :4], np.full((3, 4, 3), 3, np.uint8))
        assert np.array_equal(result[4:, 4:], np.full((3, 2, 3), 4, np.uint8))

    def test_irregular_rows_raises(self, stitcher: Stitcher):
        # Arrange
        grid = [
            [solid_tile((0, 0, 0))],
            [solid_tile((0, 0, 0)), solid_tile((0, 0, 0))],
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="same length"):
            stitcher.stitch_grid(grid)

    def test_empty_grid_raises(self, stitcher: Stitcher):
        # Act & Assert
        with pytest.raises(ValueError, match="empty tile grid"):
            stitcher.stitch_grid([])
        with pytest.raises(ValueError, match="empty tile grid"):
            stitcher.stitch_grid([[]])


@pytest.mark.unittest
class TestStitchParallel:
    def test_dimensions(self, stitcher: Stitcher, single_thread_executor: ReadExecutor):
        """Canvas shape = pixel region size."""
        # Arrange
        tile_region = Region(Point(0, 0), Size(3, 2))
        region = full_pixel_region(tile_region)
        tiles = {
            Point(x, y): solid_tile((10, 20, 30)) for x in range(3) for y in range(2)
        }

        # Act
        result = stitcher.stitch_parallel(
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
            executor=single_thread_executor,
        )

        # Assert
        assert result.shape == (2 * TILE_SIZE, 3 * TILE_SIZE, 3)

    def test_positions_from_region_start(
        self, stitcher: Stitcher, single_thread_executor: ReadExecutor
    ):
        """tile_region.start maps to canvas position (0, 0)."""
        # Arrange - region offset so start != origin
        start = Point(5, 7)
        tile_region = Region(start, Size(2, 2))
        region = full_pixel_region(tile_region)
        tiles = {
            Point(start.x + column, start.y + row): solid_tile(color)
            for (column, row), color in COLORS.items()
        }

        # Act
        result = stitcher.stitch_parallel(
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
            executor=single_thread_executor,
        )

        # Assert
        for (column, row), color in COLORS.items():
            assert np.array_equal(tile_slot(result, column, row), solid_tile(color))

    def test_clips_tiles_to_region(
        self, stitcher: Stitcher, single_thread_executor: ReadExecutor
    ):
        """A region straddling all four tiles is filled from each tile's clipped
        corner, into a region-sized canvas."""
        # Arrange - 2x2 solid-color tiles cover pixels (0,0)-(8,8); the region is
        # the central 4x4 window, taking a corner from each tile.
        tile_region = Region(Point(0, 0), Size(2, 2))
        region = Region(Point(2, 2), Size(4, 4))
        tiles = {Point(c, r): solid_tile(color) for (c, r), color in COLORS.items()}

        # Act
        result = stitcher.stitch_parallel(
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
            executor=single_thread_executor,
        )

        # Assert - contiguous, region-sized, each quadrant from its tile
        assert result.shape == (4, 4, 3)
        assert result.flags["C_CONTIGUOUS"]
        assert np.array_equal(
            result[:2, :2], np.full((2, 2, 3), COLORS[(0, 0)], np.uint8)
        )
        assert np.array_equal(
            result[:2, 2:], np.full((2, 2, 3), COLORS[(1, 0)], np.uint8)
        )
        assert np.array_equal(
            result[2:, :2], np.full((2, 2, 3), COLORS[(0, 1)], np.uint8)
        )
        assert np.array_equal(
            result[2:, 2:], np.full((2, 2, 3), COLORS[(1, 1)], np.uint8)
        )

    def test_multiple_chunks_assembled_correctly(
        self, stitcher: Stitcher, parallel_executor: ReadExecutor
    ):
        """A region split across workers reassembles in the right positions."""
        # Arrange
        tile_region = Region(Point(0, 0), Size(4, 4))
        region = full_pixel_region(tile_region)
        tiles = {
            Point(x, y): solid_tile((x * 10, y * 10, 0))
            for x in range(4)
            for y in range(4)
        }

        # Act
        result = stitcher.stitch_parallel(
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
            executor=parallel_executor,
        )

        # Assert
        for x in range(4):
            for y in range(4):
                assert np.array_equal(
                    tile_slot(result, x, y), solid_tile((x * 10, y * 10, 0))
                )

    def test_worker_exception_propagates(
        self, stitcher: Stitcher, single_thread_executor: ReadExecutor
    ):
        """An exception from fetch propagates through stitch_parallel."""
        # Arrange
        tile_region = Region(Point(0, 0), Size(2, 2))
        region = full_pixel_region(tile_region)

        def fetch(points: Sequence[Point]) -> Iterator[np.ndarray]:
            raise RuntimeError("fetch failed")
            yield  # unreachable, makes this a generator

        # Act & Assert
        with pytest.raises(RuntimeError, match="fetch failed"):
            stitcher.stitch_parallel(
                region,
                tile_region,
                fetch,
                Size(TILE_SIZE, TILE_SIZE),
                np.dtype(np.uint8),
                3,
                executor=single_thread_executor,
            )

    def test_fetch_returns_too_few_tiles_raises(
        self, stitcher: Stitcher, single_thread_executor: ReadExecutor
    ):
        # Arrange - two points requested, one tile yielded
        tile_region = Region(Point(0, 0), Size(2, 1))
        region = full_pixel_region(tile_region)

        def fetch(points: Sequence[Point]) -> Iterator[np.ndarray]:
            yield solid_tile((0, 0, 0))

        # Act & Assert
        with pytest.raises(ValueError):
            stitcher.stitch_parallel(
                region,
                tile_region,
                fetch,
                Size(TILE_SIZE, TILE_SIZE),
                np.dtype(np.uint8),
                3,
                executor=single_thread_executor,
            )

    def test_fetch_returns_too_many_tiles_raises(
        self, stitcher: Stitcher, single_thread_executor: ReadExecutor
    ):
        # Arrange - one point requested, two tiles yielded
        tile_region = Region(Point(0, 0), Size(1, 1))
        region = full_pixel_region(tile_region)

        def fetch(points: Sequence[Point]) -> Iterator[np.ndarray]:
            yield solid_tile((0, 0, 0))
            yield solid_tile((0, 0, 0))

        # Act & Assert
        with pytest.raises(ValueError):
            stitcher.stitch_parallel(
                region,
                tile_region,
                fetch,
                Size(TILE_SIZE, TILE_SIZE),
                np.dtype(np.uint8),
                3,
                executor=single_thread_executor,
            )


@pytest.mark.unittest
class TestReferenceAgreement:
    """The Pillow reference must compose to the same array as the numpy
    stitcher; this is what makes it a valid cross-check."""

    def test_stitch_grid_agrees(self):
        # Arrange - random tiles so agreement is not trivial
        rng = np.random.default_rng(0)
        grid = [
            [
                rng.integers(0, 256, (TILE_SIZE, TILE_SIZE, 3), np.uint8)
                for _ in range(3)
            ]
            for _ in range(2)
        ]

        # Act
        numpy_result = NumpyStitcher().stitch_grid(grid)
        pillow_result = PillowStitcher().stitch_grid(grid)

        # Assert
        assert np.array_equal(numpy_result, pillow_result)

    def test_stitch_parallel_agrees(self, single_thread_executor: ReadExecutor):
        # Arrange
        rng = np.random.default_rng(1)
        tile_region = Region(Point(0, 0), Size(3, 2))
        region = full_pixel_region(tile_region)
        tiles = {
            Point(x, y): rng.integers(0, 256, (TILE_SIZE, TILE_SIZE, 3), np.uint8)
            for x in range(3)
            for y in range(2)
        }
        args = (
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
        )

        # Act
        numpy_result = NumpyStitcher().stitch_parallel(
            *args, executor=single_thread_executor
        )
        pillow_result = PillowStitcher().stitch_parallel(
            *args, executor=single_thread_executor
        )

        # Assert
        assert np.array_equal(numpy_result, pillow_result)

    def test_stitch_parallel_clipped_agrees(self, single_thread_executor: ReadExecutor):
        """Agreement must also hold when the region is clipped inside its tiles."""
        # Arrange - a misaligned region straddling a 3x3 tile block
        rng = np.random.default_rng(2)
        tile_region = Region(Point(1, 1), Size(3, 3))
        region = Region(Point(1 * TILE_SIZE + 1, 1 * TILE_SIZE + 2), Size(9, 8))
        tiles = {
            Point(x, y): rng.integers(0, 256, (TILE_SIZE, TILE_SIZE, 3), np.uint8)
            for x in range(1, 4)
            for y in range(1, 4)
        }
        args = (
            region,
            tile_region,
            fetch_from(tiles),
            Size(TILE_SIZE, TILE_SIZE),
            np.dtype(np.uint8),
            3,
        )

        # Act
        numpy_result = NumpyStitcher().stitch_parallel(
            *args, executor=single_thread_executor
        )
        pillow_result = PillowStitcher().stitch_parallel(
            *args, executor=single_thread_executor
        )

        # Assert
        assert numpy_result.shape == (8, 9, 3)
        assert np.array_equal(numpy_result, pillow_result)
