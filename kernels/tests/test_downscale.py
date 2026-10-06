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

"""Tests for downscale: box reduction, box pyramids, RGBX output and filters.

The box reduction against the true mean, the box pyramids' interleaved/planar
and SIMD/scalar agreement, the fused RGB to RGBX output, and antialiasing of
every filter when minifying.
"""

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, resize, rgb_to_rgbx

RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]
ScalarKernels = Callable[[], contextlib.AbstractContextManager[None]]
SimdAndScalar = Callable[[Callable[[], Any]], tuple[Any, Any]]

DTYPES = [np.uint8, np.uint16, np.float32]
CONVOLUTION_FILTERS = [
    ResizeFilter.BOX,
    ResizeFilter.BILINEAR,
    ResizeFilter.CATMULLROM,
    ResizeFilter.LANCZOS3,
]
BOX_PYRAMID_SIZES = [(64, 64), (67, 133), (129, 517), (96, 1032)]
BOX_PYRAMID_FACTORS = [2, 4, 8, 16, 32]


def is_integer(dtype: type[np.generic]) -> bool:
    return np.issubdtype(dtype, np.integer)


@pytest.fixture
def odd_rgb_image(random_image: RandomImage) -> np.ndarray:
    """Make a uint8 RGB image of odd size, so the vector body and scalar tail run."""
    return random_image((67, 91, 3), np.uint8)


@pytest.fixture
def kernels(simd: bool, scalar_kernels: ScalarKernels) -> ScalarKernels:
    """Return a context manager running the SIMD kernels or the scalar reference.

    The scalar reference for ``simd=False``.
    """
    return contextlib.nullcontext if simd else scalar_kernels


@pytest.fixture
def noise(random_image: RandomImage) -> np.ndarray:
    """White noise: a discriminator between resamplers and interpolators."""
    return random_image((256, 256), np.uint8)


@pytest.mark.unittest
class TestDownscaleToRgbx:
    """``pad`` reduces RGB straight to RGBX, mapping into Pillow with no copy."""

    @pytest.mark.parametrize("factor", [1, 2, 4])
    def test_box_pad_equals_box_then_expand(
        self, odd_rgb_image: np.ndarray, factor: int
    ) -> None:
        """Bit-exact with reducing to RGB and expanding after.

        Every box kernel rounds exactly ((sum+2)>>2), vector body and scalar tail
        alike.
        """
        # Act
        rgb = downscale(odd_rgb_image, factor, ResizeFilter.BOX)
        rgbx = downscale(odd_rgb_image, factor, ResizeFilter.BOX, pad=255)

        # Assert
        assert rgbx.shape == (*rgb.shape[:2], 4)
        assert rgbx.dtype == np.uint8
        assert (rgbx[..., :3] == rgb).all()
        assert (rgbx[..., 3] == 255).all()

    def test_pad_value_fills_fourth_sample(self, odd_rgb_image: np.ndarray) -> None:
        """The fourth sample is the given pad value."""
        # Act
        rgbx = downscale(odd_rgb_image, 2, ResizeFilter.BOX, pad=3)

        # Assert
        assert (rgbx[..., 3] == 3).all()

    @pytest.mark.parametrize("factor", [2, 4, 8])
    def test_simd_equals_scalar(
        self, odd_rgb_image: np.ndarray, simd_and_scalar: SimdAndScalar, factor: int
    ) -> None:
        """The fused SIMD kernels are bit-exact with the scalar fallback."""
        # Act
        simd, scalar = simd_and_scalar(
            lambda: downscale(odd_rgb_image, factor, ResizeFilter.BOX, pad=255)
        )

        # Assert
        assert (simd == scalar).all()

    def test_lanczos_resizes_then_expands(self, odd_rgb_image: np.ndarray) -> None:
        """A non-box filter resizes to RGB and expands, still 4 samples out."""
        # Act
        lanczos = downscale(odd_rgb_image, 2, ResizeFilter.LANCZOS3, pad=7)

        # Assert
        assert lanczos.shape == (34, 46, 4)
        assert (lanczos[..., 3] == 7).all()

    def test_fractional_box_resize_expands(self, odd_rgb_image: np.ndarray) -> None:
        """A fractional box resize expands to RGBX too."""
        # Act
        fractional = resize(odd_rgb_image, 21, 30, ResizeFilter.BOX, pad=1)

        # Assert
        assert fractional.shape == (21, 30, 4)
        assert (fractional[..., 3] == 1).all()

    @pytest.mark.parametrize(
        ["unsupported", "message"],
        [
            (lambda image: image[..., 0], r"needs a 3-sample \(H, W, 3\) uint8 image"),
            (lambda image: image.astype(np.uint16), "needs a uint8 image"),
        ],
        ids=["planar", "uint16"],
    )
    def test_rejects_unsupported_images(
        self,
        odd_rgb_image: np.ndarray,
        unsupported: Callable[[np.ndarray], np.ndarray],
        message: str,
    ) -> None:
        """Planar and non-uint8 images are rejected, not silently mangled."""
        # Arrange
        image = unsupported(odd_rgb_image)

        # Act & Assert
        with pytest.raises(ValueError, match=message):
            downscale(image, 2, ResizeFilter.BOX, pad=0)


@pytest.mark.unittest
class TestDownscaleBox:
    """The box reduction is the mean of each block, in every implementation.

    SIMD, scalar, interleaved and planar all agree.
    """

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_planar_matches_mean(
        self, random_image: RandomImage, dtype: type[np.generic]
    ) -> None:
        """A planar 2x box reduction is the block mean."""
        # Arrange
        image = random_image((64, 64), dtype)
        mean = image.reshape(32, 2, 32, 2).mean(axis=(1, 3))
        tolerance = 0.5 if is_integer(dtype) else 1e-5  # integers: rounded once

        # Act
        reduced = downscale(image, 2, ResizeFilter.BOX)

        # Assert
        assert reduced.shape == (32, 32)
        assert reduced.dtype == dtype
        assert np.abs(reduced.astype(float) - mean).max() <= tolerance

    def test_interleaved_rgb_matches_mean(self, random_image: RandomImage) -> None:
        """Interleaved uint8 RGB averages within each channel."""
        # Arrange
        image = random_image((64, 64, 3), np.uint8)
        mean = np.rint(image.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))).astype(int)

        # Act
        reduced = downscale(image, 2, ResizeFilter.BOX)

        # Assert
        assert reduced.shape == (32, 32, 3)
        assert reduced.dtype == np.uint8
        assert np.abs(reduced.astype(int) - mean).max() <= 1

    @pytest.mark.parametrize("dtype", [np.uint16, np.float32])
    @pytest.mark.parametrize("channels", [3, 4])
    def test_interleaved_matches_mean(
        self, random_image: RandomImage, dtype: type[np.generic], channels: int
    ) -> None:
        """uint16 and float32 average within each channel.

        At every interleaved channel count.
        """
        # Arrange
        image = random_image((48, 48, channels), dtype)
        mean = image.reshape(24, 2, 24, 2, channels).mean(axis=(1, 3))
        tolerance = 1.0 if is_integer(dtype) else 1e-4

        # Act
        reduced = downscale(image, 2, ResizeFilter.BOX)

        # Assert
        assert reduced.shape == (24, 24, channels)
        assert reduced.dtype == dtype
        assert np.abs(reduced.astype(float) - mean).max() <= tolerance

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("simd", [True, False])
    def test_factor_4_is_the_rounded_mean(
        self, random_image: RandomImage, kernels: ScalarKernels, dtype: type[np.generic]
    ) -> None:
        """Factor 4 is the 4x4 mean rounded once, on every tier.

        Integer results are exactly (sum + 8) >> 4; float32 is the mean to
        rounding.
        """
        # Arrange
        image = random_image((64, 64), dtype)
        sums = image.astype(np.float64).reshape(16, 4, 16, 4).sum(axis=(1, 3))

        # Act
        with kernels():
            reduced = downscale(image, 4, ResizeFilter.BOX)

        # Assert
        assert reduced.shape == (16, 16)
        assert reduced.dtype == dtype
        if is_integer(dtype):
            assert np.array_equal(reduced, (sums.astype(np.int64) + 8) >> 4)
        else:
            assert np.allclose(reduced, sums / 16, rtol=1e-6)

    @pytest.mark.parametrize(
        ["dtype", "channels", "height", "width", "factor"],
        [
            (dtype, channels, height, width, factor)
            for dtype in DTYPES
            for channels in [1, 3, 4]
            for height, width in BOX_PYRAMID_SIZES
            for factor in [4, 8, 16]
        ],
    )
    def test_pyramid_simd_equals_scalar(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        dtype: type[np.generic],
        channels: int,
        height: int,
        width: int,
        factor: int,
    ) -> None:
        """Box pyramids of 4x and more are the same with SIMD and scalar kernels.

        Every tier fuses two levels per pass with one rounding, so the result
        does not depend on the CPU.
        """
        # Arrange
        shape = (height, width) if channels == 1 else (height, width, channels)
        image = random_image(shape, dtype)

        # Act
        simd, scalar = simd_and_scalar(
            lambda: downscale(image, factor, ResizeFilter.BOX)
        )

        # Assert — float32 within rounding: summation order may differ in tails
        if is_integer(dtype):
            assert np.array_equal(simd, scalar)
        else:
            assert np.allclose(simd, scalar, rtol=1e-6)

    @pytest.mark.parametrize(
        ["dtype", "channels", "height", "width", "factor"],
        [
            (dtype, channels, height, width, factor)
            for dtype in DTYPES
            for channels in [1, 3, 4]
            for height, width in [(9, 13), (64, 64), (67, 133), (129, 517)]
            for factor in [2, 4, 8, 16]
            if height >= factor and width >= factor
        ],
    )
    @pytest.mark.parametrize("simd", [True, False])
    def test_writes_every_output_sample(
        self,
        kernels: ScalarKernels,
        dtype: type[np.generic],
        channels: int,
        height: int,
        width: int,
        factor: int,
    ) -> None:
        """The box kernels don't clear their output before writing it.

        A zero image reduced right after a full-scale one (whose freed output the
        allocator hands back) comes out all zero: a skipped sample would show
        the earlier output's value.
        """
        # Arrange
        shape = (height, width) if channels == 1 else (height, width, channels)
        full_scale = np.iinfo(np.dtype(dtype).name).max if is_integer(dtype) else 1.0

        # Act
        with kernels():
            downscale(np.full(shape, full_scale, dtype), factor, ResizeFilter.BOX)
            reduced = downscale(np.zeros(shape, dtype), factor, ResizeFilter.BOX)

        # Assert
        assert not reduced.any()

    @pytest.mark.parametrize(
        ["dtype", "channels", "height", "width", "factor"],
        [
            (dtype, channels, height, width, factor)
            for dtype in DTYPES
            for channels in [3, 4]
            for height, width in BOX_PYRAMID_SIZES
            for factor in BOX_PYRAMID_FACTORS
            if height >= factor and width >= factor
        ],
    )
    @pytest.mark.parametrize("simd", [True, False])
    def test_interleaved_equals_planar_per_channel(
        self,
        random_image: RandomImage,
        kernels: ScalarKernels,
        dtype: type[np.generic],
        channels: int,
        height: int,
        width: int,
        factor: int,
    ) -> None:
        """Each channel of an interleaved box pyramid equals it reduced as planar.

        Interleaved box pyramids take the planar ones' passes (fused 4x, the
        last one 8x for an odd level count) on every tier.
        """
        # Arrange
        image = random_image((height, width, channels), dtype)

        # Act
        with kernels():
            reduced = downscale(image, factor, ResizeFilter.BOX)
            planar = [
                downscale(
                    np.ascontiguousarray(image[..., channel]), factor, ResizeFilter.BOX
                )
                for channel in range(channels)
            ]

        # Assert
        assert np.array_equal(reduced, np.stack(planar, -1))

    @pytest.mark.parametrize(
        ["height", "width", "factor"],
        [
            (height, width, factor)
            for height, width in BOX_PYRAMID_SIZES
            for factor in BOX_PYRAMID_FACTORS
            if height % factor == 0 and width % factor == 0
        ],
    )
    @pytest.mark.parametrize("simd", [True, False])
    def test_pad_equals_rgbx_of_reduced(
        self,
        random_image: RandomImage,
        kernels: ScalarKernels,
        height: int,
        width: int,
        factor: int,
    ) -> None:
        """A uint8 RGB box pyramid with pad= equals the RGB one expanded to RGBX.

        With a whole factor.
        """
        # Arrange
        image = random_image((height, width, 3), np.uint8)

        # Act
        with kernels():
            reduced = downscale(image, factor, ResizeFilter.BOX)
            padded = resize(
                image, height // factor, width // factor, ResizeFilter.BOX, pad=7
            )

        # Assert
        assert (padded == rgb_to_rgbx(reduced, 7)).all()

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [(129, 131), (65, 67, 3), (65, 67, 4)])
    def test_simd_equals_scalar(
        self,
        random_image: RandomImage,
        simd_and_scalar: SimdAndScalar,
        dtype: type[np.generic],
        shape: tuple[int, ...],
    ) -> None:
        """Planar and interleaved box kernels equal the scalar reference for integers.

        Bit-identical: every path rounds exactly ((sum+2)>>2, matching Pillow
        reduce), on AVX2 and NEON alike. Odd sizes exercise both the vector body
        and the tail.
        """
        # Arrange
        image = random_image(shape, dtype)

        # Act
        simd, scalar = simd_and_scalar(lambda: downscale(image, 2, ResizeFilter.BOX))

        # Assert
        if is_integer(dtype):
            assert (simd == scalar).all()
        else:
            assert np.allclose(simd, scalar, rtol=1e-6)


@pytest.mark.unittest
class TestDownscaleFilters:
    """The resampling filters keep shape and dtype, and low-pass before decimating."""

    @pytest.mark.parametrize("channels", [3, 4])
    def test_interleaved_lanczos_keeps_channels(self, channels: int) -> None:
        """Interleaved RGB and RGBA reduce with lanczos3."""
        # Arrange
        image = np.ones((32, 32, channels), np.uint8)

        # Act
        reduced = downscale(image, 2, ResizeFilter.LANCZOS3)

        # Assert
        assert reduced.shape == (16, 16, channels)
        assert reduced.dtype == np.uint8

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_lanczos_factor_4_keeps_dtype(self, dtype: type[np.generic]) -> None:
        """A factor-4 lanczos3 reduction keeps every dtype."""
        # Arrange
        image = np.ones((32, 32), dtype)

        # Act
        reduced = downscale(image, 4, ResizeFilter.LANCZOS3)

        # Assert
        assert reduced.shape == (8, 8)
        assert reduced.dtype == dtype

    def test_box_averages_checkerboard_to_gray(self) -> None:
        """The highest-frequency pattern averages to gray, without aliasing."""
        # Arrange
        rows, columns = np.mgrid[0:256, 0:256]
        checkerboard = (((columns + rows) % 2) * 255).astype(np.uint8)

        # Act
        reduced = downscale(checkerboard, 2, ResizeFilter.BOX).astype(float)

        # Assert
        assert reduced.std() < 1.0
        assert abs(reduced.mean() - 128) < 2

    @pytest.mark.parametrize("filter", CONVOLUTION_FILTERS)
    def test_convolution_filters_antialias_noise(
        self, noise: np.ndarray, filter: ResizeFilter
    ) -> None:
        """Every filter except nearest low-passes before decimating.

        At integer and fractional ratios: the convolution scales the kernel
        support with the ratio, unlike cv2's INTER_LINEAR/CUBIC/LANCZOS4, which
        are fixed-kernel interpolators that alias when minifying. A resampler
        averages ~64 noise pixels per output pixel at 8x, collapsing the std
        ~8x; an aliasing interpolator keeps it ~intact.
        """
        # Act
        reduced = downscale(noise, 8, filter).astype(float)
        resized = resize(noise, 49, 49, filter).astype(float)

        # Assert
        assert reduced.std() < 20
        assert resized.std() < 25

    def test_nearest_aliases_noise(self, noise: np.ndarray) -> None:
        """Nearest is the documented aliaser: noise keeps its std."""
        # Act
        reduced = downscale(noise, 8, ResizeFilter.NEAREST).astype(float)

        # Assert
        assert reduced.std() > 60

    @pytest.mark.parametrize(
        "filter",
        [ResizeFilter.BILINEAR, ResizeFilter.CATMULLROM, ResizeFilter.LANCZOS3],
    )
    def test_windowed_filters_do_not_moire(self, filter: ResizeFilter) -> None:
        """Periodic stripes reduce without a moire beat.

        Box is left out: it legitimately keeps a low-frequency beat, as true area
        means oscillate with phase.
        """
        # Arrange
        stripes = np.zeros((256, 256), np.uint8)
        stripes[:, ::3] = 255

        # Act
        reduced = downscale(stripes, 4, filter).astype(float)

        # Assert
        assert reduced.std() < 10


def box_block_means(image: np.ndarray, factor: int) -> np.ndarray:
    """Return each block's mean: the blocks at the image's end are partial.

    Integers rounded half up.
    """
    height, width = image.shape[:2]
    out = np.empty(
        (-(-height // factor), -(-width // factor), *image.shape[2:]), np.float64
    )
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            block = image[
                y * factor : (y + 1) * factor, x * factor : (x + 1) * factor
            ].astype(np.float64)
            out[y, x] = block.mean(axis=(0, 1))
    if is_integer(image.dtype.type):
        return np.floor(out + 0.5).astype(image.dtype)
    return out.astype(image.dtype)


@pytest.mark.unittest
class TestDownscalePartialBlocks:
    """A factor that doesn't divide the size: ceil size, the scale exact."""

    @pytest.mark.parametrize("filter", list(ResizeFilter))
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("factor", [2, 3, 4, 8])
    @pytest.mark.parametrize("shape", [(37, 51), (37, 51, 3)])
    def test_output_size_rounds_up(
        self,
        random_image: RandomImage,
        filter: ResizeFilter,
        dtype: type[np.generic],
        factor: int,
        shape: tuple[int, ...],
    ) -> None:
        """Every filter gives ceil(dim / factor) per axis."""
        # Arrange
        image = random_image(shape, dtype)

        # Act
        reduced = downscale(image, factor, filter)

        # Assert
        assert reduced.shape == (-(-37 // factor), -(-51 // factor), *shape[2:])

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("factor", [2, 4])
    @pytest.mark.parametrize("shape", [(7, 9), (13, 6, 3), (5, 11, 4)])
    def test_box_partial_blocks_are_the_mean_of_what_they_cover(
        self,
        random_image: RandomImage,
        dtype: type[np.generic],
        factor: int,
        shape: tuple[int, ...],
    ) -> None:
        """Box at 2 and 4 is each block's mean, the partial edge blocks too."""
        # Arrange
        image = random_image(shape, dtype)

        # Act
        reduced = downscale(image, factor, ResizeFilter.BOX)

        # Assert
        expected = box_block_means(image, factor)
        if is_integer(dtype):
            assert np.array_equal(reduced, expected)
        else:
            assert np.allclose(reduced, expected, atol=1e-6)

    @pytest.mark.parametrize("filter", list(ResizeFilter))
    def test_factor_beyond_the_image_gives_one_pixel(
        self, random_image: RandomImage, filter: ResizeFilter
    ) -> None:
        """A factor past the image's size reduces it to one pixel."""
        # Arrange
        image = random_image((3, 5), np.uint8)

        # Act
        reduced = downscale(image, 8, filter)

        # Assert
        assert reduced.shape == (1, 1)
        if filter == ResizeFilter.BOX:
            assert reduced[0, 0] == np.floor(image.mean() + 0.5)

    def test_nearest_partial_block_takes_its_last_pixel(self) -> None:
        """Nearest samples each block's center, the image's last pixel past it."""
        # Arrange
        image = np.tile(np.arange(9, dtype=np.uint8), (9, 1))

        # Act
        reduced = downscale(image, 2, ResizeFilter.NEAREST)

        # Assert
        assert reduced[0].tolist() == [1, 3, 5, 7, 8]

    @pytest.mark.parametrize(
        "filter",
        [ResizeFilter.BOX, ResizeFilter.BILINEAR, ResizeFilter.CATMULLROM],
    )
    def test_scale_is_the_factor(
        self, random_image: RandomImage, filter: ResizeFilter
    ) -> None:
        """Outputs whose windows end inside the image ignore the remainder.

        The scale is the factor, not size / output size: the image is not
        stretched over the whole blocks.
        """
        # Arrange
        image = random_image((40, 40), np.uint8)
        whole_blocks = np.ascontiguousarray(image[:39, :39])

        # Act
        reduced = downscale(image, 3, filter)
        reduced_whole = downscale(whole_blocks, 3, filter)

        # Assert
        assert np.array_equal(reduced[:11, :11], reduced_whole[:11, :11])
