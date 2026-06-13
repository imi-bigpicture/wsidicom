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

"""Tests for PillowStitcher."""

from collections.abc import Iterator, Sequence

import pytest
from PIL import Image

from wsidicom.geometry import Point, Region, Size
from wsidicom.stitcher import PillowStitcher

TILE_SIZE = 4  # tile pixel size used in tests


def solid_tile(color: int | tuple[int, ...], mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (TILE_SIZE, TILE_SIZE), color=color)


def expected_2x2() -> dict[tuple[int, int], tuple[int, int, int]]:
    """Reference 2x2 grid of distinct solid colors keyed by (column, row)."""
    return {
        (0, 0): (255, 0, 0),
        (1, 0): (0, 255, 0),
        (0, 1): (0, 0, 255),
        (1, 1): (255, 255, 0),
    }


@pytest.fixture
def stitcher() -> PillowStitcher:
    return PillowStitcher()


@pytest.mark.unittest
class TestPillowStitcherStitch:
    def test_canvas_dimensions(self, stitcher: PillowStitcher):
        """Stitched canvas size = grid_size * tile_size."""
        # Arrange
        tiles = [solid_tile((255, 0, 0)) for _ in range(6)]

        # Act
        result = stitcher.stitch(tiles, Size(3, 2))

        # Assert
        assert result.size == (3 * TILE_SIZE, 2 * TILE_SIZE)

    def test_row_major_layout(self, stitcher: PillowStitcher):
        """Tiles appear in row-major order at their expected canvas positions."""
        # Arrange
        colors = expected_2x2()
        tiles = [
            solid_tile(colors[(0, 0)]),
            solid_tile(colors[(1, 0)]),
            solid_tile(colors[(0, 1)]),
            solid_tile(colors[(1, 1)]),
        ]

        # Act
        result = stitcher.stitch(tiles, Size(2, 2))

        # Assert
        for (column, row), color in colors.items():
            sample_x = column * TILE_SIZE + 1
            sample_y = row * TILE_SIZE + 1
            assert result.getpixel((sample_x, sample_y)) == color

    def test_fill_color(self, stitcher: PillowStitcher):
        """Fill color shows through where fewer tiles than grid_size.area."""
        # Arrange
        tiles = [solid_tile((255, 0, 0))]  # only one tile for a 2x2 grid
        fill = (10, 20, 30)

        # Act
        result = stitcher.stitch(tiles, Size(2, 2), fill=fill)

        # Assert
        # Tile 0 occupies (0,0) quadrant; rest should be fill
        assert result.getpixel((1, 1)) == (255, 0, 0)
        assert result.getpixel((TILE_SIZE + 1, 1)) == fill
        assert result.getpixel((1, TILE_SIZE + 1)) == fill
        assert result.getpixel((TILE_SIZE + 1, TILE_SIZE + 1)) == fill

    def test_grayscale_fill(self, stitcher: PillowStitcher):
        """Integer fill is accepted for grayscale ('L') mode."""
        # Arrange
        tiles = [solid_tile(200, mode="L")]  # only one tile for 2x2 grid
        fill = 128  # grayscale background

        # Act
        result = stitcher.stitch(tiles, Size(2, 2), fill=fill)

        # Assert
        assert result.mode == "L"
        assert result.getpixel((1, 1)) == 200
        assert result.getpixel((TILE_SIZE + 1, 1)) == fill
        assert result.getpixel((TILE_SIZE + 1, TILE_SIZE + 1)) == fill

    def test_empty_iterable_raises(self, stitcher: PillowStitcher):
        """Empty iterable raises ValueError."""
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            stitcher.stitch([], Size(2, 2))


@pytest.mark.unittest
class TestPillowStitcherStitchGrid:
    def test_2x2_grid(self, stitcher: PillowStitcher):
        """stitch_grid produces same layout as flat stitch for rectangular input."""
        # Arrange
        colors = expected_2x2()
        grid = [
            [solid_tile(colors[(0, 0)]), solid_tile(colors[(1, 0)])],
            [solid_tile(colors[(0, 1)]), solid_tile(colors[(1, 1)])],
        ]

        # Act
        result = stitcher.stitch_grid(grid)

        # Assert
        assert result.size == (2 * TILE_SIZE, 2 * TILE_SIZE)
        for (column, row), color in colors.items():
            assert (
                result.getpixel((column * TILE_SIZE + 1, row * TILE_SIZE + 1)) == color
            )

    def test_irregular_rows_raises(self, stitcher: PillowStitcher):
        """Rows of unequal length raise ValueError."""
        # Arrange
        grid = [
            [solid_tile((255, 0, 0)), solid_tile((0, 255, 0))],
            [solid_tile((0, 0, 255))],
        ]

        # Act / Assert
        with pytest.raises(ValueError):
            stitcher.stitch_grid(grid)

    def test_empty_grid_raises(self, stitcher: PillowStitcher):
        """Empty outer or inner sequence raises ValueError."""
        # Arrange / Act / Assert
        with pytest.raises(ValueError):
            stitcher.stitch_grid([])
        with pytest.raises(ValueError):
            stitcher.stitch_grid([[]])


@pytest.mark.unittest
class TestPillowStitcherStitchParallel:
    def test_canvas_dimensions_default_grid_size(self, stitcher: PillowStitcher):
        """Default canvas size = tile_region.size * tile_size."""
        # Arrange
        tile_region = Region(Point(0, 0), Size(3, 2))
        color_by_point = {
            Point(x, y): (40 + x * 10, 60 + y * 10, 80)
            for x in range(3)
            for y in range(2)
        }

        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for point in chunk:
                yield solid_tile(color_by_point[point])

        # Act
        result = stitcher.stitch_parallel(
            tile_region=tile_region,
            fetch=fetch,
            tile_size=Size(TILE_SIZE, TILE_SIZE),
            mode="RGB",
            threads=1,
        )

        # Assert
        assert result.size == (3 * TILE_SIZE, 2 * TILE_SIZE)

    def test_canvas_positions_from_tile_region_start(self, stitcher: PillowStitcher):
        """Points are pasted at (point - tile_region.start) * tile_size."""
        # Arrange — tile_region anchored at (10, 20)
        tile_region = Region(Point(10, 20), Size(2, 2))
        colors = expected_2x2()
        color_by_point = {
            Point(10, 20): colors[(0, 0)],
            Point(11, 20): colors[(1, 0)],
            Point(10, 21): colors[(0, 1)],
            Point(11, 21): colors[(1, 1)],
        }

        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for point in chunk:
                yield solid_tile(color_by_point[point])

        # Act
        result = stitcher.stitch_parallel(
            tile_region=tile_region,
            fetch=fetch,
            tile_size=Size(TILE_SIZE, TILE_SIZE),
            mode="RGB",
            threads=1,
        )

        # Assert
        for (column, row), color in colors.items():
            assert (
                result.getpixel((column * TILE_SIZE + 1, row * TILE_SIZE + 1)) == color
            )

    def test_multiple_chunks_assembled_correctly(self, stitcher: PillowStitcher):
        """Multiple chunks dispatched in parallel still produce correct layout."""
        # Arrange — region naturally splits into multiple chunks when threads > 1
        tile_region = Region(Point(0, 0), Size(2, 2))
        colors = expected_2x2()
        color_by_point = {
            Point(0, 0): colors[(0, 0)],
            Point(1, 0): colors[(1, 0)],
            Point(0, 1): colors[(0, 1)],
            Point(1, 1): colors[(1, 1)],
        }

        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for point in chunk:
                yield solid_tile(color_by_point[point])

        # Act
        result = stitcher.stitch_parallel(
            tile_region=tile_region,
            fetch=fetch,
            tile_size=Size(TILE_SIZE, TILE_SIZE),
            mode="RGB",
            threads=2,
        )

        # Assert
        for (column, row), color in colors.items():
            assert (
                result.getpixel((column * TILE_SIZE + 1, row * TILE_SIZE + 1)) == color
            )

    def test_canvas_grid_size_larger_than_region_fills_uncovered(
        self, stitcher: PillowStitcher
    ):
        """Explicit canvas_grid_size larger than tile_region embeds in a bigger canvas."""
        # Arrange — fetch covers a 1x1 region, but canvas is 2x2
        tile_region = Region(Point(0, 0), Size(1, 1))
        fill = (10, 20, 30)
        target_color = (200, 100, 50)

        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for _ in chunk:
                yield solid_tile(target_color)

        # Act
        result = stitcher.stitch_parallel(
            tile_region=tile_region,
            fetch=fetch,
            tile_size=Size(TILE_SIZE, TILE_SIZE),
            mode="RGB",
            canvas_grid_size=Size(2, 2),
            fill=fill,
            threads=1,
        )

        # Assert — covered slot has target color, uncovered slots show fill
        assert result.size == (2 * TILE_SIZE, 2 * TILE_SIZE)
        assert result.getpixel((1, 1)) == target_color
        assert result.getpixel((TILE_SIZE + 1, TILE_SIZE + 1)) == fill

    def test_smaller_tile_keeps_fill_in_remaining_slot(self, stitcher: PillowStitcher):
        """A fetched tile smaller than tile_size leaves blank fill in its slot."""
        # Arrange — fetch returns a half-width tile
        tile_region = Region(Point(0, 0), Size(1, 1))
        fill = (10, 20, 30)
        target_color = (200, 100, 50)

        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for _ in chunk:
                # Tile is half the slot width
                yield Image.new("RGB", (TILE_SIZE // 2, TILE_SIZE), color=target_color)

        # Act
        result = stitcher.stitch_parallel(
            tile_region=tile_region,
            fetch=fetch,
            tile_size=Size(TILE_SIZE, TILE_SIZE),
            mode="RGB",
            fill=fill,
            threads=1,
        )

        # Assert — left half is the cropped tile, right half is fill
        assert result.getpixel((1, 1)) == target_color
        assert result.getpixel((TILE_SIZE - 1, 1)) == fill

    def test_worker_exception_propagates(self, stitcher: PillowStitcher):
        """An exception from fetch propagates through stitch_parallel."""

        # Arrange
        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            raise RuntimeError("fetch failed")
            yield  # unreachable, makes this a generator

        # Act / Assert
        with pytest.raises(RuntimeError, match="fetch failed"):
            stitcher.stitch_parallel(
                tile_region=Region(Point(0, 0), Size(1, 1)),
                fetch=fetch,
                tile_size=Size(TILE_SIZE, TILE_SIZE),
                mode="RGB",
                threads=1,
            )

    def test_canvas_grid_size_smaller_than_region_raises(
        self, stitcher: PillowStitcher
    ):
        """canvas_grid_size smaller than tile_region.size raises ValueError."""

        # Arrange
        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for _ in chunk:
                yield solid_tile((0, 0, 0))

        # Act / Assert
        with pytest.raises(ValueError, match="canvas_grid_size"):
            stitcher.stitch_parallel(
                tile_region=Region(Point(0, 0), Size(2, 2)),
                fetch=fetch,
                tile_size=Size(TILE_SIZE, TILE_SIZE),
                mode="RGB",
                canvas_grid_size=Size(1, 1),
                threads=1,
            )

    def test_fetch_returns_too_few_tiles_raises(self, stitcher: PillowStitcher):
        """fetch yielding fewer tiles than points raises ValueError (strict zip)."""

        # Arrange
        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            # Yields one less tile than the chunk requires
            for _ in chunk[:-1]:
                yield solid_tile((0, 0, 0))

        # Act / Assert
        with pytest.raises(ValueError):
            stitcher.stitch_parallel(
                tile_region=Region(Point(0, 0), Size(2, 1)),
                fetch=fetch,
                tile_size=Size(TILE_SIZE, TILE_SIZE),
                mode="RGB",
                threads=1,
            )

    def test_fetch_returns_too_many_tiles_raises(self, stitcher: PillowStitcher):
        """fetch yielding more tiles than points raises ValueError (strict zip)."""

        # Arrange
        def fetch(chunk: Sequence[Point]) -> Iterator[Image.Image]:
            for _ in chunk:
                yield solid_tile((0, 0, 0))
            yield solid_tile((0, 0, 0))  # one too many

        # Act / Assert
        with pytest.raises(ValueError):
            stitcher.stitch_parallel(
                tile_region=Region(Point(0, 0), Size(2, 1)),
                fetch=fetch,
                tile_size=Size(TILE_SIZE, TILE_SIZE),
                mode="RGB",
                threads=1,
            )
