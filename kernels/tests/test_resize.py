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

"""Tests for resize: accuracy, parity, layouts, exact ratios, nearest and caching.

Accuracy against the exact convolution, SIMD/scalar parity, layouts (planar,
interleaved, RGBX output), exact integer ratios, nearest sample positions and
the per-thread plan cache.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, resize, rgb_to_rgbx

ImageSize = tuple[int, int]
RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]
SimdAndScalar = Callable[[Callable[[], Any]], tuple[Any, Any]]
ExactResize = Callable[[np.ndarray, int, int, ResizeFilter], np.ndarray]
Weights = Callable[[int], np.ndarray]

DTYPES = [np.uint8, np.uint16, np.float32]

CONVOLUTION_FILTERS = [
    ResizeFilter.BOX,
    ResizeFilter.BILINEAR,
    ResizeFilter.CATMULLROM,
    ResizeFilter.LANCZOS3,
]

ALL_FILTERS = [ResizeFilter.NEAREST, *CONVOLUTION_FILTERS]

NATIVE_CASES: list[tuple[ImageSize, ImageSize]] = [
    ((64, 96), (32, 48)),  # exact 2x: phased plan
    ((60, 90), (20, 30)),  # exact 3x
    ((64, 96), (16, 24)),  # exact 4x
    ((50, 70), (37, 52)),  # fractional: block plan
    ((40, 60), (60, 90)),  # upscale 1.5x (edge weights > 1.0)
    ((40, 60), (80, 120)),  # upscale 2x
    ((5, 7), (3, 4)),  # edge-only
    ((8, 8), (2, 2)),
]


def takes_power_of_two_box(size: ImageSize, output_size: ImageSize) -> bool:
    """Whether a box resize takes the power-of-2 box kernels.

    The resize is from ``size`` to ``output_size``; otherwise it takes the
    convolution.
    """
    height, width = size
    output_height, output_width = output_size
    factor = width // output_width
    return (
        height == factor * output_height
        and width == factor * output_width
        and factor > 1
        and factor & (factor - 1) == 0
    )


def convolution_cases(
    cases: Iterable[tuple[tuple[int, ...], ImageSize]],
) -> list[tuple[tuple[int, ...], ImageSize, ResizeFilter]]:
    """Pair each case with each convolution filter.

    The box cases that take the power-of-2 box kernels (not the convolution
    under test) are left out.
    """
    return [
        (shape, output_size, filter)
        for shape, output_size in cases
        for filter in CONVOLUTION_FILTERS
        if not (
            filter == ResizeFilter.BOX
            and takes_power_of_two_box((shape[0], shape[1]), output_size)
        )
    ]


@pytest.fixture
def image_shape(height: int, width: int, channels: int) -> tuple[int, ...]:
    """Return a planar shape for one channel, else an interleaved one."""
    return (height, width) if channels == 1 else (height, width, channels)


@pytest.mark.unittest
class TestResizeChannels:
    @pytest.mark.parametrize(
        ["dtype", "level"],
        [(np.uint8, 200), (np.uint16, 50000), (np.float32, 0.75)],
    )
    @pytest.mark.parametrize("filter", ALL_FILTERS)
    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda image, f: resize(image, 21, 30, f), id="fractional"),
            pytest.param(lambda image, f: downscale(image, 3, f), id="downscale-3"),
            pytest.param(lambda image, f: resize(image, 32, 48, f), id="exact-2x"),
        ],
    )
    def test_resize_filters_channels_independently(
        self,
        dtype: type[np.generic],
        level: float,
        filter: ResizeFilter,
        operation: Callable[[np.ndarray, ResizeFilter], np.ndarray],
    ) -> None:
        """4-channel images are generic channels, not colour + alpha.

        No channel is scaled by another (fast_image_resize would premultiply by
        a "last channel = alpha" unless told not to).
        """
        # Arrange - constant colour; last channel 0 ("transparent")
        image = np.zeros((64, 96, 4), dtype)
        image[..., 0] = level

        # Act
        result = operation(image, filter)

        # Assert
        assert np.allclose(result[..., 0], level, rtol=1e-6)
        assert (result[..., -1] == 0).all()


@pytest.mark.unittest
class TestResizeAccuracy:
    @pytest.mark.parametrize(["size", "output_size"], NATIVE_CASES)
    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_native_resize_within_one_lsb_of_exact(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        exact_resize_u8: ExactResize,
        size: ImageSize,
        output_size: ImageSize,
        filter: ResizeFilter,
    ) -> None:
        """Planar uint8 convolution resizes are native, within 1 LSB of exact.

        They compute fast_image_resize's convolution natively, on every CPU tier
        and in forced-scalar mode (fast_image_resize itself rounds and clamps an
        intermediate: up to ~20 LSB off near edges that clip at 0/255).
        """
        # Arrange
        image = random_image(size, np.uint8)
        expected = exact_resize_u8(image, *output_size, filter)

        # Act
        simd, scalar = simd_and_scalar(lambda: resize(image, *output_size, filter))

        # Assert
        assert np.abs(simd.astype(float) - expected).max() <= 1
        assert np.abs(scalar.astype(float) - expected).max() <= 1

    @pytest.mark.parametrize(
        ["dtype", "shape", "output_size", "filter"],
        [
            (dtype, *case)
            for dtype in (np.uint16, np.float32)
            for case in convolution_cases(
                [
                    ((37, 53), (19, 71)),
                    ((64, 80, 3), (32, 40)),
                    ((50, 61, 4), (150, 90)),
                    ((90, 70, 4), (11, 23)),
                ]
            )
        ],
    )
    def test_native_float_resize_exact_and_identical_on_every_tier(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        exact_resize_float: ExactResize,
        dtype: type[np.generic],
        shape: tuple[int, ...],
        output_size: ImageSize,
        filter: ResizeFilter,
    ) -> None:
        """uint16 and float32 resize natively in f32 (FMA chains in tap order).

        The SIMD tiers equal the scalar reference bit for bit; uint16 within
        1 LSB and float32 within 1e-5 of the exact convolution.
        """
        # Arrange
        image = random_image(shape, dtype)
        exact = exact_resize_float(image, *output_size, filter)

        # Act
        simd, scalar = simd_and_scalar(lambda: resize(image, *output_size, filter))

        # Assert
        assert np.array_equal(simd, scalar)
        if dtype == np.uint16:
            expected = np.clip(np.rint(exact), 0, 65535)
            assert np.abs(simd.astype(float) - expected).max() <= 1
        else:
            assert np.abs(simd.astype(float) - exact).max() <= 1e-5

    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_native_large_reductions_within_one_lsb(
        self,
        random_image: RandomImage,
        exact_resize_float: ExactResize,
        channels: int,
        dtype: type[np.generic],
        filter: ResizeFilter,
    ) -> None:
        """Large reductions stay native for every layout: within 1 LSB of exact.

        No block plan; the transposed path for 1 and 3 channels, tap runs for
        RGBA.
        """
        # Arrange
        shape = (96, 160, channels) if channels > 1 else (96, 160)
        image = random_image(shape, dtype)
        exact = exact_resize_float(image, 12, 13, filter)
        if dtype == np.float32:
            expected, tolerance = exact, 1e-5
        else:
            expected = np.clip(np.rint(exact), 0, np.iinfo(np.dtype(dtype).name).max)
            tolerance = 1

        # Act
        result = resize(image, 12, 13, filter)

        # Assert
        assert np.abs(result.astype(float) - expected).max() <= tolerance

    @pytest.mark.parametrize(
        ["size", "output_size"], [((256, 8), (2, 8)), ((8, 256), (8, 2))]
    )
    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_uint8_reduction_to_1_128_within_one_lsb_on_flat_images(
        self,
        simd_and_scalar: SimdAndScalar,
        size: ImageSize,
        output_size: ImageSize,
        filter: ResizeFilter,
    ) -> None:
        """The uint8 fixed-point limit: each tap rounds to the nearest 1/64 LSB.

        So the error grows with the window; flat images are the worst case, and
        down to 1/128 per axis (where larger reductions switch to the f32
        engine) they stay within 1 LSB.
        """
        # Arrange
        values = range(1, 256, 2)

        # Act
        results = [
            simd_and_scalar(
                lambda value=value: resize(
                    np.full(size, value, np.uint8), *output_size, filter
                )
            )
            for value in values
        ]

        # Assert
        for value, (simd, scalar) in zip(values, results, strict=True):
            assert np.abs(simd.astype(int) - value).max() <= 1, value
            assert np.abs(scalar.astype(int) - value).max() <= 1, value

    @pytest.mark.parametrize(
        ["shape", "output_size"],
        [
            ((3, 40000), (1, 3)),
            ((20000, 2), (2, 1)),
            ((1200, 1300, 3), (3, 4)),
            ((520, 300, 4), (2, 1)),
        ],
    )
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_extreme_reductions_within_one_lsb_on_every_tier(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        exact_resize_float: ExactResize,
        shape: tuple[int, ...],
        output_size: ImageSize,
        dtype: type[np.generic],
        filter: ResizeFilter,
    ) -> None:
        """Reductions by hundreds to tens of thousands stay within 1 LSB.

        uint8 takes the f32 engine past 1/128, and windows of thousands of
        taps sum in chunks, so no rounding adds up over the window; the
        result is identical on every tier.
        """
        # Arrange
        image = random_image(shape, dtype)
        exact = exact_resize_float(image, *output_size, filter)
        if dtype == np.float32:
            expected, tolerance = exact, 1e-5
        else:
            expected = np.clip(np.rint(exact), 0, np.iinfo(np.dtype(dtype).name).max)
            tolerance = 1

        # Act
        simd, scalar = simd_and_scalar(lambda: resize(image, *output_size, filter))

        # Assert
        assert np.array_equal(simd, scalar)
        assert np.abs(simd.astype(float) - expected).max() <= tolerance

    @pytest.mark.parametrize(
        ["dtype", "value"], [(np.uint8, 255), (np.uint16, 40000), (np.float32, 0.61)]
    )
    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_flat_image_keeps_its_value_at_any_reduction(
        self,
        dtype: type[np.generic],
        value: float,
        filter: ResizeFilter,
    ) -> None:
        """A flat image reduced by 40000 to one pixel keeps its value.

        No rounding adds up over the window, even on flat input.
        """
        # Arrange
        image = np.full((2, 40000), value, dtype)

        # Act
        result = resize(image, 1, 1, filter)

        # Assert
        assert np.allclose(result.astype(float), value, rtol=1e-6, atol=0)


@pytest.mark.unittest
class TestResizeSimdParity:
    @pytest.mark.parametrize(
        ["size", "output_size", "filter"],
        convolution_cases(
            [
                *NATIVE_CASES,
                ((300, 410), (222, 303)),
                ((256, 256), (1024, 1024)),
                # catmullrom: output 161's center falls on a sample (a
                # single-tap window, starting after output 162's) right at a
                # block's start.
                ((20, 203), (33, 323)),
            ]
        ),
    )
    def test_native_resize_simd_matches_scalar_reference(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        size: ImageSize,
        output_size: ImageSize,
        filter: ResizeFilter,
    ) -> None:
        """Identical results on every CPU: SIMD matches the scalar reference.

        The SIMD path (AVX-512 or AVX2) matches it bit for bit.
        """
        # Arrange
        image = random_image(size, np.uint8)

        # Act
        simd, scalar = simd_and_scalar(lambda: resize(image, *output_size, filter))

        # Assert
        assert np.array_equal(simd, scalar)

    @pytest.mark.parametrize(
        ["dtype", "shape"],
        [
            (np.uint8, (120, 1024, 4)),
            (np.uint8, (100, 4096)),
            (np.uint16, (100, 512, 4)),
            (np.float32, (100, 1024)),
        ],
    )
    @pytest.mark.parametrize("reduction", [4, 6, 8])
    def test_vertical_tap_groups_equal_scalar(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        dtype: type[np.generic],
        shape: tuple[int, ...],
        reduction: int,
    ) -> None:
        """Row strides a multiple of 4 KB group the taps, identical to scalar.

        With enough taps (16+ on AVX-512, 12+ on AVX2) the vertical pass runs
        with the taps in groups: the same sums in the same order, so identical
        to the scalar reference.
        """
        # Arrange
        image = random_image(shape, dtype)
        output_height = shape[0] // reduction
        output_width = shape[1] // reduction + 3

        # Act
        simd, scalar = simd_and_scalar(
            lambda: resize(image, output_height, output_width, ResizeFilter.LANCZOS3)
        )

        # Assert
        assert np.array_equal(simd, scalar)


@pytest.mark.unittest
class TestResizeLayouts:
    @pytest.mark.parametrize(
        ["size", "output_size", "filter"],
        convolution_cases(
            [
                *NATIVE_CASES,
                ((300, 410), (222, 303)),
                ((96, 128), (240, 320)),
                ((90, 120), (30, 40)),
                ((64, 400), (16, 50)),
                ((32, 48), (128, 192)),  # integer upscales: the phase path
                ((50, 96), (100, 192)),
                ((96, 160), (24, 40)),  # large reductions: tap runs
                ((100, 205), (20, 41)),
                ((91, 150), (30, 50)),
                ((120, 400), (15, 41)),  # lanczos3 past the blocks: planar runs too
            ]
        ),
    )
    @pytest.mark.parametrize("channels", [3, 4])
    def test_native_interleaved_resize_equals_planar_per_channel(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        size: ImageSize,
        output_size: ImageSize,
        filter: ResizeFilter,
        channels: int,
    ) -> None:
        """Interleaved uint8 images resize natively with the planar arithmetic.

        Per channel: bit-identical to resizing each channel as a planar image
        (so within 1 LSB of exact too), SIMD and scalar alike.
        """
        # Arrange
        image = random_image((*size, channels), np.uint8)
        planes = [np.ascontiguousarray(image[..., c]) for c in range(channels)]
        expected = np.stack(
            [resize(plane, *output_size, filter) for plane in planes], axis=-1
        )

        # Act
        simd, scalar = simd_and_scalar(lambda: resize(image, *output_size, filter))

        # Assert
        assert np.array_equal(simd, expected)
        assert np.array_equal(scalar, expected)

    @pytest.mark.parametrize(
        ["size", "output_size"],
        [((90, 120), (61, 77)), ((40, 60), (80, 120)), ((64, 96), (32, 48))],
    )
    @pytest.mark.parametrize("filter", list(ResizeFilter))
    def test_resize_pad_equals_resize_then_rgbx(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        size: ImageSize,
        output_size: ImageSize,
        filter: ResizeFilter,
    ) -> None:
        """pad= (RGB in, RGBX out) is the RGB resize with the 4th sample set.

        On every path and filter: written as RGBX directly, transposed,
        scalar, and nearest expanded.
        """
        # Arrange
        image = random_image((*size, 3), np.uint8)

        # Act
        simd, scalar = simd_and_scalar(
            lambda: (
                rgb_to_rgbx(resize(image, *output_size, filter), pad=7),
                resize(image, *output_size, filter, pad=7),
            )
        )

        # Assert
        for expected, padded in (simd, scalar):
            assert np.array_equal(padded, expected)


@pytest.fixture
def reduction_weights(factor: int, taps: tuple[int, ...]) -> Weights:
    """Return the integer weights of an exact reduction by ``factor``.

    An (n // factor, n) matrix with the taps clipped at the edges.
    """

    def weights(length: int) -> np.ndarray:
        matrix = np.zeros((length // factor, length), np.int64)
        first = (factor - len(taps)) // 2
        for output in range(length // factor):
            for tap, weight in enumerate(taps):
                if 0 <= factor * output + first + tap < length:
                    matrix[output, factor * output + first + tap] = weight
        return matrix

    return weights


@pytest.fixture
def reduction_image(
    random_image: RandomImage,
    factor: int,
    image_shape: tuple[int, ...],
    dtype: type[np.integer],
    content: str,
) -> np.ndarray:
    """Make an image ``factor`` times the indirect size, random or saturated."""
    shape = (factor * image_shape[0], factor * image_shape[1], *image_shape[2:])
    if content == "saturated":
        return np.full(shape, np.iinfo(dtype).max, dtype)
    return random_image(shape, dtype)


@pytest.fixture
def double_size_weights() -> Weights:
    """Return the integer weights of bilinear to exactly twice the size.

    A (2n, n) matrix: 3 and 1 over 4 per axis, the clipped edge window
    renormalized to the edge sample alone (the same as weight 4 on it).
    """

    def weights(length: int) -> np.ndarray:
        matrix = np.zeros((2 * length, length), np.int64)
        for output in range(2 * length):
            near = output // 2
            far = near - 1 if output % 2 == 0 else near + 1
            matrix[output, near] += 3
            matrix[output, min(max(far, 0), length - 1)] += 1
        return matrix

    return weights


def integer_convolution(
    rows: np.ndarray, image: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Return the separable integer convolution sums, for any channel count."""
    return np.einsum(
        "oh,hw...,pw->op...", rows, image.astype(np.int64), columns, optimize=True
    )


@pytest.mark.unittest
class TestResizeExactRatios:
    @pytest.mark.parametrize(
        ["filter", "factor", "taps", "dtype"],
        [
            (ResizeFilter.BILINEAR, 2, (1, 3, 3, 1), np.uint8),
            (ResizeFilter.CATMULLROM, 2, (-3, -9, 29, 111, 111, 29, -9, -3), np.uint8),
            (ResizeFilter.BILINEAR, 4, (1, 3, 5, 7, 7, 5, 3, 1), np.uint8),
            (ResizeFilter.BILINEAR, 2, (1, 3, 3, 1), np.uint16),
        ],
    )
    @pytest.mark.parametrize(
        ["height", "width"], [(1, 1), (2, 3), (5, 33), (17, 35), (65, 129)]
    )
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("content", ["random", "saturated"])
    def test_integer_reduction_is_exact(
        self,
        simd_and_scalar: SimdAndScalar,
        reduction_weights: Weights,
        reduction_image: np.ndarray,
        filter: ResizeFilter,
        factor: int,
        dtype: type[np.integer],
    ) -> None:
        """The exact-ratio integer resizes compute the convolution exactly.

        uint8 bilinear and catmullrom to exactly half the size, uint8 bilinear
        to exactly a quarter and uint16 bilinear to exactly half: integer
        weights per axis (clipped at the edges), one rounded division per
        output, clamped. Every tier and layout.
        """
        # Arrange
        height, width = reduction_image.shape[:2]
        rows, columns = reduction_weights(height), reduction_weights(width)
        sums = integer_convolution(rows, reduction_image, columns)
        divisor = rows.sum(1)[:, None] * columns.sum(1)[None, :]
        if reduction_image.ndim == 3:
            divisor = divisor[..., None]
        expected = np.clip(
            (2 * sums + divisor) // (2 * divisor), 0, np.iinfo(dtype).max
        )

        # Act
        simd, scalar = simd_and_scalar(
            lambda: resize(reduction_image, height // factor, width // factor, filter)
        )

        # Assert
        assert np.array_equal(simd, expected)
        assert np.array_equal(scalar, expected)

    @pytest.mark.parametrize(
        ["height", "width"], [(1, 1), (3, 5), (10, 17), (33, 70), (65, 129)]
    )
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_bilinear_to_double_size_is_exact(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        double_size_weights: Weights,
        image_shape: tuple[int, ...],
        height: int,
        width: int,
    ) -> None:
        """uint8 bilinear to exactly twice the size: weights 3 and 1 over 4.

        Per axis; one rounded division per output.
        """
        # Arrange
        image = random_image(image_shape, np.uint8)
        sums = integer_convolution(
            double_size_weights(height), image, double_size_weights(width)
        )
        expected = (sums + 8) // 16

        # Act
        simd, scalar = simd_and_scalar(
            lambda: resize(image, 2 * height, 2 * width, ResizeFilter.BILINEAR)
        )

        # Assert
        assert np.array_equal(simd, expected)
        assert np.array_equal(scalar, expected)

    @pytest.mark.parametrize("shape", [(64, 96), (64, 96, 3), (64, 96, 4)])
    @pytest.mark.parametrize("exponent", [1, 2])
    def test_resize_exact_power_of_two_box_matches_downscale(
        self, random_image: RandomImage, shape: tuple[int, ...], exponent: int
    ) -> None:
        """Resize to exactly 1/2^k with box is the same operation as downscale.

        The same (exactly rounded) result.
        """
        # Arrange
        image = random_image(shape, np.uint8)
        expected = downscale(image, 2**exponent, ResizeFilter.BOX)

        # Act
        result = resize(image, 64 >> exponent, 96 >> exponent, ResizeFilter.BOX)

        # Assert
        assert np.array_equal(result, expected)

    def test_resize_fractional(self, random_image: RandomImage) -> None:
        """A non-integer ratio (the read_region_mpp case) gives the exact size.

        The exact target size, for any dtype, planar and interleaved, and it
        still antialiases.
        """
        # Arrange
        image = random_image((100, 140, 3), np.uint8)
        uint16_image = random_image((64, 64), np.uint16)
        float_image = random_image((64, 64), np.float32)
        flat = np.full((100, 140, 3), 200, np.uint8)

        # Act
        reduced = resize(image, 41, 57, ResizeFilter.BOX)
        planar = resize(
            np.ascontiguousarray(image[..., 0]), 33, 33, ResizeFilter.LANCZOS3
        )
        enlarged = resize(image, 200, 280, ResizeFilter.CATMULLROM)
        uint16_reduced = resize(uint16_image, 25, 25, ResizeFilter.BILINEAR)
        float_reduced = resize(float_image, 25, 25, ResizeFilter.BOX)
        flat_reduced = resize(flat, 41, 57, ResizeFilter.BOX)

        # Assert
        assert reduced.shape == (41, 57, 3)
        assert reduced.dtype == np.uint8
        assert planar.shape == (33, 33)
        assert enlarged.shape == (200, 280, 3)
        assert uint16_reduced.dtype == np.uint16
        assert float_reduced.dtype == np.float32
        # Antialiasing: a flat field survives a fractional reduction unchanged.
        assert np.abs(flat_reduced.astype(int) - 200).max() <= 1


@pytest.fixture
def nearest_reference() -> Callable[[np.ndarray, int, int], np.ndarray]:
    """Return fast_image_resize's nearest sample positions.

    Column floor((x + 0.5) * sx), rows by accumulating sy from sy / 2 (only
    rows inside the image).
    """

    def resize_nearest(
        image: np.ndarray, output_height: int, output_width: int
    ) -> np.ndarray:
        height, width = image.shape[:2]
        step_x, step_y = width / output_width, height / output_height
        columns = np.minimum(
            (step_x * 0.5 + step_x * np.arange(output_width)).astype(int), width - 1
        )
        result = np.zeros((output_height, output_width, *image.shape[2:]), image.dtype)
        rows = min(
            output_height, int(np.ceil(max((height - step_y * 0.5) / step_y, 0.0)))
        )
        y = step_y * 0.5
        for row in range(rows):
            result[row] = image[min(int(y), height - 1)][columns]
            y += step_y
        return result

    return resize_nearest


@pytest.mark.unittest
class TestResizeNearest:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        ["shape", "output_size"],
        [((37, 53), (19, 71)), ((64, 48, 3), (64, 16)), ((30, 31, 4), (97, 13))],
    )
    def test_nearest_matches_fast_image_resize_positions(
        self,
        rng: np.random.Generator,
        nearest_reference: Callable[[np.ndarray, int, int], np.ndarray],
        dtype: type[np.generic],
        shape: tuple[int, ...],
        output_size: ImageSize,
    ) -> None:
        """Nearest is native for every dtype and channel count.

        With fast_image_resize's sample positions.
        """
        # Arrange
        image = (rng.random(shape) * 200).astype(dtype)

        # Act
        result = resize(image, *output_size, ResizeFilter.NEAREST)

        # Assert
        assert np.array_equal(result, nearest_reference(image, *output_size))


@pytest.mark.unittest
class TestResizePlanCache:
    def test_native_resize_reuses_plans_correctly(
        self, random_image: RandomImage
    ) -> None:
        """Plans are cached per thread by geometry and filter.

        Repeating a call, evicting it (more geometries than the cache holds) and
        calling from other threads all give the first call's result.
        """
        # Arrange
        calls = [
            (random_image((30 + i, 50), np.uint8), 20 + i, filter)
            for i in range(10)
            for filter in (ResizeFilter.BILINEAR, ResizeFilter.LANCZOS3)
        ]

        def resize_all() -> list[np.ndarray]:
            return [
                resize(image, output_height, 40, filter)
                for image, output_height, filter in calls
            ]

        first = resize_all()

        # Act
        with ThreadPoolExecutor(4) as pool:
            repeated = [resize_all(), *pool.map(lambda _: resize_all(), range(4))]

        # Assert
        for results in repeated:
            for expected, result in zip(first, results, strict=True):
                assert np.array_equal(result, expected)
