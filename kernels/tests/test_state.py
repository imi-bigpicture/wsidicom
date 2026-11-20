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

"""Tests that the state the resize side keeps never shows in a result.

- Outputs are allocated without clearing, and the allocator recycles a freed
  output of the same size: a result equals itself after such an output full
  of junk was freed (a kernel writing past a scratch buffer, or skipping an
  output sample, shows here and nowhere else);
- plans are cached per thread, built under whichever SIMD tier was on: a plan
  built with SIMD serves scalar calls as a scalar-built one does, and the
  reverse;
- other calls in between (other sizes, filters and dtypes) change nothing.

Inputs are small, narrow (1 to 12 pixels wide, where the vector kernels take
their tails and scalar fallbacks) and odd-sized.
"""

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, downscale_grid, resize

RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]
Operation = Callable[[np.ndarray], np.ndarray]
ScalarKernels = Callable[[], contextlib.AbstractContextManager[None]]

DTYPES = [np.uint8, np.uint16]
CHANNELS = [1, 3]
SIZES = [
    *((height, width) for width in range(1, 13) for height in (1, 6)),
    (7, 13),
    (11, 37),
    (17, 70),
    (33, 29),
]
# Values the junk outputs are filled with, as fractions of the dtype's maximum.
JUNK_FRACTIONS = [0.0, 1.0, 0.5]
CONVOLUTION_FILTERS = [
    ResizeFilter.BOX,
    ResizeFilter.BILINEAR,
    ResizeFilter.CATMULLROM,
    ResizeFilter.LANCZOS3,
]


def resize_sizes(height: int, width: int) -> list[tuple[int, int]]:
    """Return output sizes: half (rounded up), double and an uneven ratio."""
    return [
        (-(-height // 2), -(-width // 2)),
        (2 * height, 2 * width),
        (height + 1, max(1, 2 * width // 3)),
    ]


def resizing(resample: ResizeFilter) -> list[Operation]:
    """Return resizes by ``filter`` to each of an image's ``resize_sizes``."""

    def to(index: int) -> Operation:
        def run(image: np.ndarray) -> np.ndarray:
            output_height, output_width = resize_sizes(*image.shape[:2])[index]
            return resize(image, output_height, output_width, resample)

        return run

    return [to(index) for index in range(len(resize_sizes(1, 1)))]


def grid_halving(resample: ResizeFilter) -> Operation:
    """Return a halving of the image split into a grid, in a filled frame.

    The inner tiles have even sizes, so ``BOX`` reduces each tile on its own
    into the frame; other filters stitch the grid first.
    """

    def run(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        row_split, column_split = 2 * (height // 4), 2 * (width // 4)
        rows = [image[:row_split], image[row_split:]] if row_split else [image]
        tiles = [
            np.ascontiguousarray(piece)
            for row in rows
            for piece in (
                [row[:, :column_split], row[:, column_split:]]
                if column_split
                else [row]
            )
        ]
        channels = image.shape[2] if image.ndim == 3 else 1
        fill = np.arange(1, channels + 1).astype(image.dtype)
        size = (-(-height // 2) + 1, -(-width // 2) + 2)
        return downscale_grid(
            tiles, 2 if column_split else 1, 2, resample, size=size, fill=fill
        )

    return run


OPERATIONS: dict[str, list[Operation]] = {
    **{f"resize_{resample}": resizing(resample) for resample in CONVOLUTION_FILTERS},
    "resize_nearest": resizing(ResizeFilter.NEAREST),
    "downscale_box": [lambda image: downscale(image, 2, ResizeFilter.BOX)],
    "grid_box": [grid_halving(ResizeFilter.BOX)],
    "grid_bilinear": [grid_halving(ResizeFilter.BILINEAR)],
}


def evict_plans() -> None:
    """Push every cached plan out of this thread's caches with other geometries."""
    for dtype in DTYPES:
        image = np.zeros((3, 3), dtype)
        for output_width in range(101, 109):
            resize(image, 4, output_width, ResizeFilter.BILINEAR)


def other_calls() -> None:
    """Resize and downscale other sizes, filters, dtypes and layouts."""
    resize(np.full((9, 13, 3), 40000, np.uint16), 5, 31, ResizeFilter.LANCZOS3)
    resize(np.full((4, 3), 200, np.uint8), 8, 6, ResizeFilter.BILINEAR)
    resize(np.full((19, 5, 3), 17, np.uint8), 7, 9, ResizeFilter.CATMULLROM)
    resize(np.full((6, 8), 3, np.uint16), 3, 4, ResizeFilter.BOX)
    resize(np.full((5, 7, 3), 99, np.uint8), 11, 2, ResizeFilter.NEAREST)
    downscale(np.full((21, 22, 3), 1234, np.uint16), 2, ResizeFilter.BOX)
    downscale(np.full((30, 33), 77, np.uint8), 4, ResizeFilter.BOX)


@pytest.fixture
def operations(operation: str) -> list[Operation]:
    """Return the calls of an indirect ``operation`` param."""
    return OPERATIONS[operation]


@pytest.fixture
def images(
    random_image: RandomImage, dtype: type[np.generic], channels: int
) -> list[np.ndarray]:
    """Make a random image of every size in ``SIZES``."""
    return [
        random_image(size if channels == 1 else (*size, channels), dtype)
        for size in SIZES
    ]


def leave_junk_output(operation: Operation, image: np.ndarray, fraction: float) -> None:
    """Leave a freed output of ``operation`` on ``image``'s shape full of junk.

    The next call of the same size then reuses that buffer.
    """
    value = int(np.iinfo(image.dtype).max * fraction)
    junk = operation(np.full(image.shape, value, image.dtype))
    del junk


@pytest.mark.unittest
@pytest.mark.parametrize("channels", CHANNELS, ids=["planar", "rgb"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", list(OPERATIONS))
class TestResizeKeepsNoState:
    def test_recycled_junk_outputs_change_nothing(
        self, operations: list[Operation], images: list[np.ndarray]
    ) -> None:
        """A result equals itself after freed same-sized outputs full of junk."""
        for operation in operations:
            for image in images:
                # Arrange
                results = []

                # Act
                for fraction in JUNK_FRACTIONS:
                    leave_junk_output(operation, image, fraction)
                    results.append(operation(image))

                # Assert
                for result in results[1:]:
                    np.testing.assert_array_equal(
                        result, results[0], err_msg=f"{image.shape}"
                    )

    def test_plan_built_with_simd_serves_scalar_calls(
        self,
        scalar_kernels: ScalarKernels,
        operations: list[Operation],
        images: list[np.ndarray],
    ) -> None:
        """A scalar call on a plan built with SIMD equals a scalar-built one."""
        for operation in operations:
            for image in images:
                # Arrange
                evict_plans()
                operation(image)

                # Act
                with scalar_kernels():
                    after_simd = operation(image)
                    evict_plans()
                    scalar_only = operation(image)

                # Assert
                np.testing.assert_array_equal(
                    after_simd, scalar_only, err_msg=f"{image.shape}"
                )

    def test_plan_built_scalar_serves_simd_calls(
        self,
        scalar_kernels: ScalarKernels,
        operations: list[Operation],
        images: list[np.ndarray],
    ) -> None:
        """A SIMD call on a plan built scalar equals one on a SIMD-built plan."""
        for operation in operations:
            for image in images:
                # Arrange
                evict_plans()
                with scalar_kernels():
                    operation(image)

                # Act
                after_scalar = operation(image)
                evict_plans()
                simd_only = operation(image)

                # Assert
                np.testing.assert_array_equal(
                    after_scalar, simd_only, err_msg=f"{image.shape}"
                )

    def test_other_calls_in_between_change_nothing(
        self, operations: list[Operation], images: list[np.ndarray]
    ) -> None:
        """Calls of other sizes, filters and dtypes in between change no result."""
        for operation in operations:
            for image in images:
                # Arrange
                alone = operation(image)

                # Act
                other_calls()
                interleaved = operation(image)

                # Assert
                np.testing.assert_array_equal(
                    interleaved, alone, err_msg=f"{image.shape}"
                )
