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

"""Tests for input validation across the API.

Images the kernels cannot process raise a ValueError, not a panic or an empty
result; filter strings, non-contiguous arrays and unfit fills are rejected, not
converted.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import (
    ResizeFilter,
    downscale,
    downscale_grid,
    resize,
)


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((8, 8, 3), np.uint8)


@pytest.fixture
def non_contiguous_image() -> np.ndarray:
    """Every other column of an image: a strided, non-C-contiguous view."""
    return np.zeros((8, 16, 3), np.uint8)[:, ::2]


@pytest.mark.unittest
class TestValidation:
    @pytest.mark.parametrize(
        "shape", [(0, 8), (8, 0), (0, 0), (8, 0, 3), (0, 8, 3), (8, 8, 0)]
    )
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.BOX), id="resize"
            ),
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX), id="downscale"
            ),
        ],
    )
    def test_resample_rejects_empty_images(
        self, shape: tuple[int, ...], operation: Callable[[np.ndarray], Any]
    ) -> None:
        """An image with no pixels or no channels is rejected."""
        # Arrange
        image = np.zeros(shape, np.uint8)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match=r"Image must have at least one pixel and one channel"
            r"|3D arrays must have 1 or 3 channels",
        ):
            operation(image)

    @pytest.mark.parametrize("channels", [2, 4])
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX), id="downscale"
            ),
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.BILINEAR), id="resize"
            ),
            pytest.param(
                lambda image: downscale_grid([image], 1, 2, ResizeFilter.BOX),
                id="downscale_grid",
            ),
        ],
    )
    def test_rejects_channel_counts_other_than_1_3(
        self, channels: int, operation: Callable[[np.ndarray], Any]
    ) -> None:
        """Only 1 or 3 channels are supported, by every operation."""
        # Arrange
        image = np.zeros((8, 8, channels), np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match="1 or 3 channels"):
            operation(image)

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX), id="downscale"
            ),
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.BOX), id="resize"
            ),
        ],
    )
    def test_non_contiguous_image_raises_value_error(
        self, non_contiguous_image: np.ndarray, operation: Callable[[np.ndarray], Any]
    ) -> None:
        """A strided view is rejected, not silently copied."""
        # Act & Assert
        with pytest.raises(ValueError, match="C-contiguous"):
            operation(non_contiguous_image)

    @pytest.mark.parametrize(
        "fill",
        [
            np.uint8(7),
            7,
            [1, 2, 3],
            np.array([7], np.uint8),
            np.array([1, 2], np.uint8),
            np.array([1, 2, 3], np.uint16),
            np.array([1, 2, 3], np.float32),
        ],
        ids=[
            "numpy-scalar",
            "python-scalar",
            "list",
            "single-value",
            "wrong-length",
            "uint16",
            "float32",
        ],
    )
    def test_downscale_grid_rejects_unfit_fill(
        self, image: np.ndarray, fill: Any
    ) -> None:
        """The fill is a (C,) array in the tiles' dtype.

        Scalars, lists, a single value for every channel and other dtypes are
        rejected.
        """
        # Act & Assert
        with pytest.raises(ValueError, match="Fill"):
            downscale_grid([image], 1, 2, ResizeFilter.BOX, size=(8, 8), fill=fill)

    @pytest.mark.parametrize(
        ["size", "fill"],
        [((8, 8), None), (None, np.zeros(3, np.uint8))],
        ids=["size-without-fill", "fill-without-size"],
    )
    def test_downscale_grid_size_and_fill_go_together(
        self,
        image: np.ndarray,
        size: tuple[int, int] | None,
        fill: np.ndarray | None,
    ) -> None:
        """Size and fill are given together or not at all."""
        # Act & Assert
        with pytest.raises(ValueError, match="Size and fill"):
            downscale_grid([image], 1, 2, ResizeFilter.BOX, size=size, fill=fill)

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX), id="downscale"
            ),
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.BILINEAR), id="resize"
            ),
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.NEAREST),
                id="resize-nearest",
            ),
            pytest.param(
                lambda image: downscale_grid([image] * 2, 2, 2, ResizeFilter.BOX),
                id="downscale_grid",
            ),
        ],
    )
    def test_fortran_ordered_image_raises_value_error(
        self, operation: Callable[[np.ndarray], Any]
    ) -> None:
        """A Fortran-ordered array is rejected, not read as if it were C-ordered.

        Its samples are contiguous, but not in (H, W, C) order.
        """
        # Arrange
        image = np.asfortranarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))

        # Act & Assert
        with pytest.raises(ValueError, match="C-contiguous"):
            operation(image)

    @pytest.mark.parametrize(
        "size", [(2**33, 2**31), (2**62, 1)], ids=["wraps-to-zero", "wraps-small"]
    )
    def test_downscale_grid_rejects_overflowing_frame(
        self, image: np.ndarray, size: tuple[int, int]
    ) -> None:
        """A frame whose samples overflow is a ValueError, not a wrapped buffer."""
        # Arrange
        fill = np.zeros(3, np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match="too large"):
            downscale_grid([image], 1, 2, ResizeFilter.BOX, size=size, fill=fill)

    def test_resize_rejects_overflowing_output(self, image: np.ndarray) -> None:
        """An output whose samples overflow is a ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="too large"):
            resize(image, 2**40, 2**40, ResizeFilter.NEAREST)

    def test_resize_unallocatable_output_raises_memory_error(self) -> None:
        """An output no allocator can provide raises, not aborts the process."""
        # Arrange
        image = np.zeros((8, 8), np.uint8)

        # Act & Assert
        with pytest.raises(MemoryError):
            resize(image, 2**31, 2**31, ResizeFilter.NEAREST)

    def test_downscale_grid_rejects_zero_factor(self, image: np.ndarray) -> None:
        """A factor below 1 is a ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Factor must be >= 1"):
            downscale_grid([image] * 4, 2, 0, ResizeFilter.BOX)

    def test_misaligned_image_is_reported_as_such(self) -> None:
        """A C-contiguous view at an odd byte offset is rejected as misaligned."""
        # Arrange
        buffer = np.zeros(8 * 8 * 2 + 1, np.uint8)
        image = np.frombuffer(buffer.data, np.uint16, 8 * 8, 1).reshape(8, 8)

        # Act & Assert
        assert image.flags.c_contiguous
        with pytest.raises(ValueError, match="must be aligned"):
            resize(image, 4, 4, ResizeFilter.BOX)

    def test_big_endian_image_names_the_byte_order(self) -> None:
        """A non-native byte order is named in the error."""
        # Arrange
        image = np.zeros((8, 8), np.dtype(np.uint16).newbyteorder())

        # Act & Assert
        with pytest.raises(ValueError, match="native byte order"):
            resize(image, 4, 4, ResizeFilter.BOX)

    @pytest.mark.parametrize("dtype", [np.float32, np.int16])
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX), id="downscale"
            ),
            pytest.param(
                lambda image: resize(image, 4, 4, ResizeFilter.BILINEAR), id="resize"
            ),
        ],
    )
    def test_rejects_dtypes_other_than_uint8_uint16(
        self, dtype: type, operation: Callable[[np.ndarray], Any]
    ) -> None:
        """Only uint8 and uint16 images are supported, by every operation."""
        # Arrange
        image = np.zeros((8, 8), dtype)

        # Act & Assert
        with pytest.raises(ValueError, match="uint8 or uint16"):
            operation(image)

    @pytest.mark.parametrize(
        ["operation", "match"],
        [
            pytest.param(
                lambda image: resize(image, -1, 4, ResizeFilter.BOX),
                "Height and width must be >= 1",
                id="resize-negative",
            ),
            pytest.param(
                lambda image: downscale(image, -2, ResizeFilter.BOX),
                "Factor must be >= 1",
                id="downscale-negative",
            ),
            pytest.param(
                lambda image: downscale(image, 0, ResizeFilter.BOX),
                "Factor must be >= 1",
                id="downscale-zero",
            ),
        ],
    )
    def test_sizes_and_factors_below_one_raise_value_error(
        self,
        image: np.ndarray,
        operation: Callable[[np.ndarray], Any],
        match: str,
    ) -> None:
        """Negative sizes and factors are a ValueError like zero, not an overflow."""
        # Act & Assert
        with pytest.raises(ValueError, match=match):
            operation(image)

    @pytest.mark.parametrize(
        ["operation", "name"],
        [
            (lambda image, name: resize(image, 4, 4, name), "lanczos"),
            (lambda image, name: resize(image, 4, 4, name), "bicubic"),
        ],
        ids=["lanczos", "bicubic"],
    )
    def test_filter_names_are_only_the_enum_values(
        self,
        image: np.ndarray,
        operation: Callable[[np.ndarray, Any], Any],
        name: str,
    ) -> None:
        """Names other than the enum's values are unknown, not silent aliases."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown filter"):
            operation(image, name)


@pytest.mark.unittest
class TestShapes:
    @pytest.mark.parametrize(
        ["operation", "expected"],
        [
            pytest.param(
                lambda image: resize(image, 4, 6, ResizeFilter.BILINEAR),
                (4, 6, 1),
                id="resize",
            ),
            pytest.param(
                lambda image: downscale(image, 2, ResizeFilter.BOX),
                (4, 4, 1),
                id="downscale",
            ),
            pytest.param(
                lambda image: downscale_grid([image] * 4, 2, 2, ResizeFilter.BOX),
                (8, 8, 1),
                id="downscale_grid-box",
            ),
            pytest.param(
                lambda image: downscale_grid([image] * 4, 2, 2, ResizeFilter.BILINEAR),
                (8, 8, 1),
                id="downscale_grid-stitched",
            ),
            pytest.param(
                lambda image: downscale_grid(
                    [image] * 4,
                    2,
                    2,
                    ResizeFilter.BILINEAR,
                    size=(10, 10),
                    fill=np.zeros(1, np.uint8),
                ),
                (10, 10, 1),
                id="downscale_grid-framed",
            ),
        ],
    )
    def test_single_channel_axis_is_kept(
        self, operation: Callable[[np.ndarray], np.ndarray], expected: tuple[int, ...]
    ) -> None:
        """An (H, W, 1) input gives an (H, W, 1) output, as a 2D one gives 2D."""
        # Arrange
        image = np.arange(64, dtype=np.uint8).reshape(8, 8, 1)

        # Act
        result = operation(image)

        # Assert
        assert result.shape == expected
        np.testing.assert_array_equal(result[..., 0], operation(image[..., 0]))


@pytest.mark.unittest
class TestTallReductions:
    @pytest.mark.parametrize(
        ["dtype", "rows", "value"],
        [(np.uint8, 600_000, 100), (np.uint16, 300_000, 1000)],
    )
    @pytest.mark.parametrize("resample", [ResizeFilter.BILINEAR, ResizeFilter.LANCZOS3])
    def test_heavy_vertical_reduction_of_tall_image(
        self, dtype: type, rows: int, value: int, resample: ResizeFilter
    ) -> None:
        """A batch of rows larger than the strip budget still resizes.

        Its column strip once came out 0 samples wide and panicked.
        """
        # Arrange
        image = np.full((rows, 2), value, dtype)

        # Act
        result = resize(image, 1, 2, resample)

        # Assert
        np.testing.assert_allclose(result, np.full((1, 2), value), rtol=1e-3)
