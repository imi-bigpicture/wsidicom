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

The oracle pastes with numpy slicing, clipped to the canvas, so it catches bugs
the SIMD and scalar paths share (parity tests cannot see those). Positions
sweep every edge of the canvas: tiles missing it, hanging off it by all but one
sample, flush with it, and inside it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import compose

TILE_WIDTH, TILE_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT = 9, 7, 14, 13
DTYPES = [np.uint8, np.uint16, np.float32]
# Image shape suffixes: planar, then interleaved with 1, 3 and 4 samples.
LAYOUTS = [(), (1,), (3,), (4,)]
LAYOUT_IDS = ["planar", "1-sample", "rgb", "rgba"]


def edge_positions(tile_length: int, canvas_length: int) -> list[int]:
    """Return positions along one axis that put a tile across each edge.

    Just missing the canvas before it, one sample on it, partly off it, at 0,
    inside, flush with its far edge, partly off that edge, one sample on it,
    and just missing it after it.
    """
    return [
        -tile_length,
        -tile_length + 1,
        -3,
        0,
        2,
        canvas_length - tile_length,
        canvas_length - 3,
        canvas_length - 1,
        canvas_length,
    ]


def paste(canvas: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    """Paste ``tile`` with its top-left at ``(x, y)``, skipping what is off ``canvas``.

    An RGB tile onto an RGBX canvas fills only the first three samples.
    """
    height, width = tile.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]
    top, bottom = max(y, 0), min(y + height, canvas_height)
    left, right = max(x, 0), min(x + width, canvas_width)
    if top >= bottom or left >= right:
        return
    samples = tile.shape[2] if tile.ndim == 3 else None
    target = canvas[top:bottom, left:right]
    if samples is not None and samples < canvas.shape[2]:
        target = target[..., :samples]
    target[...] = tile[top - y : bottom - y, left - x : right - x]


@pytest.fixture
def tile(
    random_image: Callable[[tuple[int, ...], Any], np.ndarray],
    dtype: type[np.generic],
    layout: tuple[int, ...],
) -> np.ndarray:
    """Make a random tile of indirect ``dtype`` and ``layout`` params."""
    return random_image((TILE_HEIGHT, TILE_WIDTH, *layout), dtype)


@pytest.fixture
def canvas(
    random_image: Callable[[tuple[int, ...], Any], np.ndarray],
    dtype: type[np.generic],
    layout: tuple[int, ...],
) -> np.ndarray:
    """Make a random canvas, so samples compose must leave alone are not zero."""
    return random_image((CANVAS_HEIGHT, CANVAS_WIDTH, *layout), dtype)


@pytest.mark.unittest
class TestComposeOracle:
    @pytest.mark.parametrize("x", edge_positions(TILE_WIDTH, CANVAS_WIDTH))
    @pytest.mark.parametrize("y", edge_positions(TILE_HEIGHT, CANVAS_HEIGHT))
    @pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_tile_equals_numpy_paste(
        self, canvas: np.ndarray, tile: np.ndarray, x: int, y: int
    ) -> None:
        """A tile at any position equals the clipped numpy paste."""
        # Arrange
        expected = canvas.copy()
        paste(expected, tile, x, y)

        # Act
        compose(canvas, [tile], np.array([[x, y]], np.int64))

        # Assert
        np.testing.assert_array_equal(canvas, expected)

    @pytest.mark.parametrize("x", edge_positions(TILE_WIDTH, CANVAS_WIDTH))
    @pytest.mark.parametrize("y", edge_positions(TILE_HEIGHT, CANVAS_HEIGHT))
    @pytest.mark.parametrize("pad", [0, 200])
    def test_rgb_tile_onto_rgbx_canvas_equals_numpy_paste(
        self,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        x: int,
        y: int,
        pad: int,
    ) -> None:
        """An RGB tile onto an RGBX canvas pastes RGB and ``pad`` where it lands."""
        # Arrange
        tile = random_image((TILE_HEIGHT, TILE_WIDTH, 3), np.uint8)
        canvas = random_image((CANVAS_HEIGHT, CANVAS_WIDTH, 4), np.uint8)
        padded_tile = np.dstack([tile, np.full(tile.shape[:2], pad, np.uint8)])
        expected = canvas.copy()
        paste(expected, padded_tile, x, y)

        # Act
        compose(canvas, [tile], np.array([[x, y]], np.int64), pad=pad)

        # Assert
        np.testing.assert_array_equal(canvas, expected)

    @pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_overlapping_tiles_and_crops_equal_numpy_pastes_in_order(
        self,
        rng: np.random.Generator,
        random_image: Callable[[tuple[int, ...], Any], np.ndarray],
        canvas: np.ndarray,
        dtype: type[np.generic],
        layout: tuple[int, ...],
    ) -> None:
        """Many overlapping crops of a few tiles equal numpy pastes in turn."""
        # Arrange
        tiles = [
            random_image(
                (int(rng.integers(1, 12)), int(rng.integers(1, 12)), *layout), dtype
            )
            for _ in range(3)
        ]
        crops, positions = [], []
        for _ in range(20):
            index = int(rng.integers(0, len(tiles)))
            height, width = tiles[index].shape[:2]
            crop_x, crop_y = int(rng.integers(0, width)), int(rng.integers(0, height))
            crop_width = int(rng.integers(1, width - crop_x + 1))
            crop_height = int(rng.integers(1, height - crop_y + 1))
            crops.append([index, crop_x, crop_y, crop_width, crop_height])
            positions.append(
                [
                    int(rng.integers(-crop_width, CANVAS_WIDTH + 1)),
                    int(rng.integers(-crop_height, CANVAS_HEIGHT + 1)),
                ]
            )
        expected = canvas.copy()
        for (index, crop_x, crop_y, crop_width, crop_height), (x, y) in zip(
            crops, positions, strict=True
        ):
            piece = tiles[index][
                crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
            ]
            paste(expected, piece, x, y)

        # Act
        compose(
            canvas,
            tiles,
            np.array(positions, np.int64),
            crops=np.array(crops, np.int64),
        )

        # Assert
        np.testing.assert_array_equal(canvas, expected)
