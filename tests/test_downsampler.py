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

"""Tests for Downsampler.

The shape/layout contract holds for every implementation, so ``TestDownsampler``
runs it against ``PillowDownsampler``, ``Cv2Downsampler`` and
``KernelsDownsampler`` (the latter two skipped when their extra is absent).
Behaviour specific to one backend lives in its own class.
"""

from typing import cast

import numpy as np
import pytest
from wsidicom_data import TestData

from wsidicom import downsampler as downsampler_module
from wsidicom.config import Settings, use_settings
from wsidicom.downsampler import (
    Cv2Downsampler,
    Downsampler,
    KernelsDownsampler,
    PillowDownsampler,
    ResampleFilterOption,
)
from wsidicom.geometry import Size
from wsidicom.options import DownsamplerOption


@pytest.fixture
def image_array(
    width: int, height: int, color: int | tuple[int, int, int], dtype: np.dtype
) -> np.ndarray:
    """Return an image from an indirect ``(width, height, color, dtype)`` param.

    An int color gives a 2D grayscale array; an RGB tuple gives a ``(h, w, 3)``
    array.
    """
    if isinstance(color, int):
        return np.full((height, width), color, dtype=dtype)
    return np.full((height, width, len(color)), color, dtype=dtype)


@pytest.fixture(
    params=[
        PillowDownsampler,
        pytest.param(
            Cv2Downsampler,
            marks=pytest.mark.skipif(
                not Cv2Downsampler.is_available(), reason="opencv not installed"
            ),
        ),
        pytest.param(
            KernelsDownsampler,
            marks=pytest.mark.skipif(
                not KernelsDownsampler.is_available(),
                reason="wsidicom-kernels not installed",
            ),
        ),
    ],
    ids=["pillow", "cv2", "kernels"],
)
def downsampler(request: pytest.FixtureRequest) -> Downsampler:
    """Each Downsampler implementation, so the contract is tested against all."""
    return request.param()


@pytest.mark.unittest
class TestDownsampler:
    """The shape/layout contract every Downsampler implementation must satisfy."""

    @pytest.mark.parametrize(
        ["width", "height", "color", "dtype", "output_size", "expected_shape"],
        [
            (512, 512, (255, 0, 0), np.uint8, Size(256, 256), (256, 256, 3)),
            (512, 256, (0, 255, 0), np.uint8, Size(256, 128), (128, 256, 3)),
            (511, 255, (0, 0, 255), np.uint8, Size(255, 127), (127, 255, 3)),
            (512, 512, 128, np.uint8, Size(256, 256), (256, 256)),
            (512, 256, 200, np.uint8, Size(256, 128), (128, 256)),
            # 16-bit grayscale (no 16-bit RGB mode): dtype must be preserved.
            (512, 512, 4096, np.uint16, Size(256, 256), (256, 256)),
            (512, 256, 40000, np.uint16, Size(256, 128), (128, 256)),
        ],
    )
    def test_downsample_produces_expected_shape(
        self,
        downsampler: Downsampler,
        image_array: np.ndarray,
        dtype: np.dtype,
        output_size: Size,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Downsampling yields the requested size and dtype.

        The channel layout is preserved (arrays are rows, columns[, samples]).
        """
        # Act
        result = downsampler.downsample(image_array, output_size)

        # Assert
        assert result.shape == expected_shape
        assert result.dtype == dtype

    @pytest.mark.parametrize(
        ["width", "height", "color", "dtype", "expected_value"],
        [
            (512, 512, (10, 20, 30), np.uint8, [10, 20, 30]),
            (512, 512, 128, np.uint8, 128),
            (512, 512, 4096, np.uint16, 4096),
        ],
    )
    def test_reduce_by_half(
        self,
        downsampler: Downsampler,
        image_array: np.ndarray,
        dtype: np.dtype,
        expected_value: object,
    ) -> None:
        """reduce_by_half halves each spatial dimension (write-path fast path)."""
        # Act
        result = downsampler.reduce_by_half(image_array)

        # Assert — half size and dtype, and a uniform tile is unchanged in value
        assert result.shape[:2] == (256, 256)
        assert result.dtype == dtype
        assert np.array_equal(result[0, 0], np.array(expected_value, dtype))

    @pytest.mark.parametrize(
        ["width", "height", "color", "dtype", "max_size", "expected_shape"],
        [
            # fits within the box, aspect ratio preserved (2:1)
            (512, 256, (0, 255, 0), np.uint8, Size(128, 128), (64, 128)),
            # never upscales an array smaller than max_size
            (64, 64, (0, 0, 255), np.uint8, Size(256, 256), (64, 64)),
            # grayscale fits within the box too
            (512, 256, 200, np.uint8, Size(128, 128), (64, 128)),
            # 16-bit grayscale: dtype must be preserved
            (512, 256, 40000, np.uint16, Size(128, 128), (64, 128)),
        ],
    )
    def test_thumbnail_produces_expected_shape(
        self,
        downsampler: Downsampler,
        image_array: np.ndarray,
        dtype: np.dtype,
        max_size: Size,
        expected_shape: tuple[int, int],
    ) -> None:
        """Thumbnail fits within max_size and preserves dtype and aspect ratio.

        It never upscales.
        """
        # Act
        result = downsampler.thumbnail(image_array, max_size)

        # Assert
        assert result.shape[:2] == expected_shape
        assert result.dtype == dtype


@pytest.mark.unittest
class TestPillowDownsampler:
    def test_downsample_with_custom_resampling(self) -> None:
        """The configured resampling filter is actually applied."""
        # Arrange — a non-uniform array so the filter choice changes the result
        array = np.zeros((512, 512, 3), dtype=np.uint8)
        array[:, :256] = 255  # sharp vertical edge

        # Act — same array downsampled with two different filters
        nearest = PillowDownsampler(resample=ResampleFilterOption.NEAREST).downsample(
            array, Size(256, 256)
        )
        bilinear = PillowDownsampler(resample=ResampleFilterOption.BILINEAR).downsample(
            array, Size(256, 256)
        )

        # Assert — both yield the requested size, and the filter genuinely
        # matters (NEAREST differs from BILINEAR at the edge); a solid array
        # would pass even if the filter were ignored.
        assert nearest.shape == (256, 256, 3)
        assert bilinear.shape == (256, 256, 3)
        assert nearest.tobytes() != bilinear.tobytes()

    @pytest.mark.parametrize(
        ["resample", "expected"],
        [
            (ResampleFilterOption.BOX, True),
            (ResampleFilterOption.BILINEAR, False),
            (ResampleFilterOption.BICUBIC, False),
            (ResampleFilterOption.NEAREST, False),
            (ResampleFilterOption.LANCZOS, False),
        ],
    )
    def test_commutes_with_stitch(
        self, resample: ResampleFilterOption, expected: bool
    ) -> None:
        """commutes_with_stitch is True only for BOX.

        BILINEAR's scaled kernel crosses tile boundaries and differs from the 2x2 box
        reduce.
        """
        # Arrange
        downsampler = PillowDownsampler(resample=resample)

        # Act & Assert
        assert downsampler.commutes_with_stitch is expected

    def test_reduce_by_half_applies_the_configured_filter(self) -> None:
        # Arrange — real detail so box and bilinear genuinely differ
        array = np.asarray(TestData.image(8, 3))
        half = Size(array.shape[1] // 2, array.shape[0] // 2)
        bilinear = PillowDownsampler(resample=ResampleFilterOption.BILINEAR)
        box = PillowDownsampler(resample=ResampleFilterOption.BOX)

        # Act
        reduced = bilinear.reduce_by_half(array)

        # Assert — matches a real bilinear resize, and is not the box average
        assert np.array_equal(reduced, bilinear.downsample(array, half))
        assert not np.array_equal(reduced, box.reduce_by_half(array))


@pytest.mark.unittest
@pytest.mark.skipif(not Cv2Downsampler.is_available(), reason="opencv not installed")
class TestCv2Downsampler:
    def test_commutes_with_stitch_true_for_inter_area(self) -> None:
        """commutes_with_stitch is True for the default INTER_AREA filter."""
        # Act & Assert
        assert Cv2Downsampler().commutes_with_stitch is True

    def test_commutes_with_stitch_false_for_inter_linear(self) -> None:
        """commutes_with_stitch is False for INTER_LINEAR."""
        # Arrange
        import cv2

        # Act & Assert
        assert (
            Cv2Downsampler(interpolation=cv2.INTER_LINEAR).commutes_with_stitch is False
        )


@pytest.mark.unittest
@pytest.mark.skipif(not Cv2Downsampler.is_available(), reason="opencv not installed")
class TestCv2FromFilter:
    @pytest.mark.parametrize(
        ["resample", "cv2_attr"],
        [
            (ResampleFilterOption.NEAREST, "INTER_NEAREST_EXACT"),
            (ResampleFilterOption.BOX, "INTER_AREA"),
        ],
    )
    def test_maps_filter_to_cv2_equivalent(
        self, resample: ResampleFilterOption, cv2_attr: str
    ) -> None:
        # Arrange
        import cv2

        # Act
        downsampler = Cv2Downsampler.from_filter(resample)

        # Assert
        assert downsampler is not None
        assert downsampler._interpolation == getattr(cv2, cv2_attr)

    @pytest.mark.parametrize(
        "resample",
        [
            ResampleFilterOption.BILINEAR,
            ResampleFilterOption.HAMMING,
            ResampleFilterOption.BICUBIC,
            ResampleFilterOption.LANCZOS,
        ],
    )
    def test_filter_without_suitable_equivalent_returns_none(
        self, resample: ResampleFilterOption
    ) -> None:
        # Act & Assert
        assert Cv2Downsampler.from_filter(resample) is None


@pytest.mark.unittest
class TestDownsamplerCreate:
    """Downsampler.create selects the backend from the preference argument.

    It falls back to settings.preferred_downsampler when the argument is None.
    """

    def test_pillow_argument_forces_pillow(self) -> None:
        # Act & Assert
        assert isinstance(
            Downsampler.create(preferred=DownsamplerOption.PILLOW), PillowDownsampler
        )

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_opencv_argument_selects_cv2(self) -> None:
        # Act & Assert — BOX is one of the filters cv2 handles
        assert isinstance(
            Downsampler.create(
                resample=ResampleFilterOption.BOX, preferred=DownsamplerOption.OPENCV
            ),
            Cv2Downsampler,
        )

    def test_opencv_argument_falls_back_to_pillow_for_unsuitable_filter(self) -> None:
        # Act & Assert
        assert isinstance(
            Downsampler.create(
                resample=ResampleFilterOption.BILINEAR,
                preferred=DownsamplerOption.OPENCV,
            ),
            PillowDownsampler,
        )

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_argument_overrides_settings(self) -> None:
        # Act & Assert — the argument wins over the setting
        with use_settings(Settings(preferred_downsampler=DownsamplerOption.PILLOW)):
            assert isinstance(
                Downsampler.create(
                    resample=ResampleFilterOption.BOX,
                    preferred=DownsamplerOption.OPENCV,
                ),
                Cv2Downsampler,
            )

    def test_defaults_to_settings(self) -> None:
        # Act & Assert — no argument falls back to the setting
        with use_settings(Settings(preferred_downsampler=DownsamplerOption.PILLOW)):
            assert isinstance(Downsampler.create(), PillowDownsampler)

    @pytest.mark.skipif(
        not KernelsDownsampler.is_available(), reason="wsidicom-kernels not installed"
    )
    def test_none_setting_prefers_kernels_when_available(self) -> None:
        # Act & Assert
        with use_settings(Settings(preferred_downsampler=None)):
            assert isinstance(
                Downsampler.create(resample=ResampleFilterOption.BOX),
                KernelsDownsampler,
            )

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_none_setting_prefers_cv2_without_kernels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — as if the kernels extra were not installed
        monkeypatch.setattr(downsampler_module, "wsidicom_kernels", None)

        # Act & Assert
        with use_settings(Settings(preferred_downsampler=None)):
            assert isinstance(
                Downsampler.create(resample=ResampleFilterOption.BOX), Cv2Downsampler
            )

    @pytest.mark.skipif(
        not KernelsDownsampler.is_available(), reason="wsidicom-kernels not installed"
    )
    def test_kernels_argument_selects_kernels(self) -> None:
        # Act & Assert
        assert isinstance(
            Downsampler.create(
                resample=ResampleFilterOption.BILINEAR,
                preferred=DownsamplerOption.KERNELS,
            ),
            KernelsDownsampler,
        )

    def test_kernels_argument_falls_back_for_unsupported_filter(self) -> None:
        # Act & Assert — HAMMING has no kernels (nor cv2) equivalent
        assert isinstance(
            Downsampler.create(
                resample=ResampleFilterOption.HAMMING,
                preferred=DownsamplerOption.KERNELS,
            ),
            PillowDownsampler,
        )

    def test_unknown_name_raises(self) -> None:
        # Arrange — an invalid value (cast so the test targets runtime coercion)
        bogus = cast(DownsamplerOption, "bogus")

        # Act & Assert — rejected by __post_init__ coercion
        with pytest.raises(ValueError, match=r"is not a valid DownsamplerOption"):
            Settings(preferred_downsampler=bogus)


@pytest.mark.unittest
@pytest.mark.skipif(not Cv2Downsampler.is_available(), reason="opencv not installed")
class TestBackendEquivalence:
    """The Pillow and OpenCV downsamplers must agree where the pipeline relies on it.

    Exact-2x reduction is bit-identical, arbitrary resizes close (their algorithms
    differ and are not byte-reproducible across platforms).
    """

    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.asarray(TestData.image(8, 3))

    @pytest.fixture
    def backends(
        self, resample: ResampleFilterOption
    ) -> tuple[PillowDownsampler, Cv2Downsampler]:
        """Return the Pillow and OpenCV downsamplers for a resampling filter."""
        opencv = Cv2Downsampler.from_filter(resample)
        assert opencv is not None
        return PillowDownsampler(resample=resample), opencv

    @pytest.mark.parametrize("resample", [ResampleFilterOption.BOX])
    def test_reduce_by_half_is_bit_exact(
        self,
        image: np.ndarray,
        backends: tuple[PillowDownsampler, Cv2Downsampler],
    ) -> None:
        """Exact-2x reduction is identical on all downsamplers."""
        # Arrange
        pillow, opencv = backends

        # Act & Assert
        assert np.array_equal(
            pillow.reduce_by_half(image), opencv.reduce_by_half(image)
        )

    def test_reduce_by_half_bit_exact_on_rounding_ties(self) -> None:
        """cv2 INTER_AREA and Pillow reduce(2) agree on every .5 tie block.

        The .5 tie (2x2 sum of 2 mod 4) is the only case where they could round
        differently; cover every such block.
        """
        # Arrange
        values = np.arange(0, 254, dtype=np.uint8)
        block = np.empty((2, 2 * len(values)), dtype=np.uint8)
        block[0, 0::2] = values
        block[0, 1::2] = values
        block[1, 0::2] = values
        block[1, 1::2] = values + 2
        image = np.dstack([block, block, block])
        pillow = PillowDownsampler(resample=ResampleFilterOption.BOX)
        opencv = Cv2Downsampler.from_filter(ResampleFilterOption.BOX)
        assert opencv is not None

        # Act & Assert
        assert np.array_equal(
            pillow.reduce_by_half(image), opencv.reduce_by_half(image)
        )

    @pytest.mark.parametrize(
        "resample", [ResampleFilterOption.NEAREST, ResampleFilterOption.BOX]
    )
    @pytest.mark.parametrize(
        ["source_size", "output_size"],
        [
            (Size(240, 240), Size(150, 150)),  # square, 0.625x in both axes
            (Size(240, 160), Size(150, 100)),  # non-square, still 0.625x in both
        ],
    )
    def test_downsample_stays_close_at_non_2x(
        self,
        image: np.ndarray,
        backends: tuple[PillowDownsampler, Cv2Downsampler],
        source_size: Size,
        output_size: Size,
    ) -> None:
        """cv2 substitutes only for the two filters it matches.

        These are NEAREST (bit-identical) and BOX (bit-exact at the 2x reduce, close
        otherwise). Assert correlation rather than a magnitude bound: it is scale-invariant, so
        it ignores cv2's un-checksummable sub-LSB platform differences yet still
        collapses for a transposed or garbage result. The non-square source also
        catches a width/height swap between the wrappers.
        """
        # Arrange
        pillow, opencv = backends
        source = image[: source_size.height, : source_size.width]

        # Act
        pillow_result = pillow.downsample(source, output_size)
        opencv_result = opencv.downsample(source, output_size)

        # Assert
        correlation = np.corrcoef(
            pillow_result.astype(np.float64).ravel(),
            opencv_result.astype(np.float64).ravel(),
        )[0, 1]
        assert correlation > 0.97


@pytest.mark.unittest
@pytest.mark.skipif(
    not KernelsDownsampler.is_available(), reason="wsidicom-kernels not installed"
)
class TestKernelsDownsampler:
    @pytest.mark.parametrize(
        ["resample", "kernel_filter"],
        [
            (ResampleFilterOption.NEAREST, "nearest"),
            (ResampleFilterOption.BOX, "box"),
            (ResampleFilterOption.BILINEAR, "bilinear"),
            (ResampleFilterOption.BICUBIC, "catmullrom"),
            (ResampleFilterOption.LANCZOS, "lanczos3"),
        ],
    )
    def test_maps_filter_to_kernel_filter(
        self, resample: ResampleFilterOption, kernel_filter: str
    ) -> None:
        # Act
        downsampler = KernelsDownsampler.from_filter(resample)

        # Assert
        assert downsampler is not None
        assert downsampler._kernel_filter == kernel_filter

    def test_filter_without_equivalent_returns_none(self) -> None:
        # Act & Assert
        assert KernelsDownsampler.from_filter(ResampleFilterOption.HAMMING) is None

    @pytest.mark.parametrize(
        ["resample", "expected"],
        [
            (ResampleFilterOption.BOX, True),
            (ResampleFilterOption.BILINEAR, False),
            (ResampleFilterOption.NEAREST, False),
        ],
    )
    def test_commutes_with_stitch(
        self, resample: ResampleFilterOption, expected: bool
    ) -> None:
        # Act & Assert
        assert KernelsDownsampler(resample=resample).commutes_with_stitch is expected

    @pytest.mark.parametrize(
        "resample", [ResampleFilterOption.BOX, ResampleFilterOption.BILINEAR]
    )
    @pytest.mark.parametrize(
        ["grid_width", "grid_height"], [(2, 2), (1, 2), (2, 1), (1, 1)]
    )
    @pytest.mark.parametrize(
        ["samples", "dtype", "fill"],
        [
            (3, np.uint8, np.array([255, 255, 255], np.uint8)),
            (1, np.uint8, np.array([0], np.uint8)),
            (1, np.uint16, np.array([0], np.uint16)),
        ],
    )
    def test_reduce_block_equals_stitch_halve_and_pad(
        self,
        resample: ResampleFilterOption,
        grid_width: int,
        grid_height: int,
        samples: int,
        dtype: type[np.unsignedinteger],
        fill: np.ndarray,
    ) -> None:
        """One reduce_block call equals the encoder pool's default block step.

        That is, the tiles stitched, halved, and padded with the background.
        """
        # Arrange
        rng = np.random.default_rng(1)
        tile_size = Size(64, 64)
        shape = (64, 64) if samples == 1 else (64, 64, samples)
        top = np.iinfo(dtype).max
        tiles = [
            [rng.integers(0, top, shape, dtype=dtype) for _ in range(grid_width)]
            for _ in range(grid_height)
        ]
        downsampler = KernelsDownsampler(resample=resample)
        stitched = np.concatenate([np.concatenate(row, axis=1) for row in tiles])
        half = downsampler.downsample(
            stitched, Size(stitched.shape[1] // 2, stitched.shape[0] // 2)
        )
        expected = np.empty(shape, dtype)
        expected[...] = fill
        expected[: half.shape[0], : half.shape[1]] = half

        # Act
        block = downsampler.reduce_block(tiles, tile_size, fill)

        # Assert
        assert np.array_equal(block, expected)

    def test_reduce_by_half_of_odd_size_resamples(self) -> None:
        """An odd size has no exact 2x box; the default resample halves it."""
        # Arrange
        array = np.full((65, 33, 3), 77, np.uint8)

        # Act
        result = KernelsDownsampler(resample=ResampleFilterOption.BOX).reduce_by_half(
            array
        )

        # Assert
        assert result.shape == (32, 16, 3)
        assert np.all(result == 77)


@pytest.mark.unittest
@pytest.mark.skipif(
    not KernelsDownsampler.is_available(), reason="wsidicom-kernels not installed"
)
class TestKernelsEquivalence:
    """wsidicom-kernels agrees with Pillow where the pipeline relies on it.

    The exact-2x box reduction is bit-identical, the other filters within a level
    of Pillow's results.
    """

    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.asarray(TestData.image(8, 3))

    def test_box_reduce_by_half_is_bit_exact(self, image: np.ndarray) -> None:
        # Arrange
        pillow = PillowDownsampler(resample=ResampleFilterOption.BOX)
        kernels = KernelsDownsampler(resample=ResampleFilterOption.BOX)

        # Act & Assert
        assert np.array_equal(
            pillow.reduce_by_half(image), kernels.reduce_by_half(image)
        )

    def test_box_reduce_by_half_bit_exact_on_rounding_ties(self) -> None:
        """Every 2x2 sum of 2 mod 4 (a .5 tie) rounds as Pillow's reduce(2)."""
        # Arrange
        values = np.arange(0, 254, dtype=np.uint8)
        block = np.empty((2, 2 * len(values)), dtype=np.uint8)
        block[0, 0::2] = values
        block[0, 1::2] = values
        block[1, 0::2] = values
        block[1, 1::2] = values + 2
        image = np.dstack([block, block, block])
        pillow = PillowDownsampler(resample=ResampleFilterOption.BOX)
        kernels = KernelsDownsampler(resample=ResampleFilterOption.BOX)

        # Act & Assert
        assert np.array_equal(
            pillow.reduce_by_half(image), kernels.reduce_by_half(image)
        )

    @pytest.mark.parametrize(
        ["resample", "max_difference", "max_fraction_beyond_one"],
        [
            (ResampleFilterOption.BOX, 2, 1.0),
            (ResampleFilterOption.BILINEAR, 2, 1.0),
            # Pillow rounds and clamps an 8-bit intermediate between its passes,
            # so where these filters overshoot (sharp edges) it lands a few
            # levels off the exact convolution, which wsidicom-kernels stays
            # within one level of.
            (ResampleFilterOption.BICUBIC, 8, 0.01),
            (ResampleFilterOption.LANCZOS, 8, 0.01),
        ],
    )
    @pytest.mark.parametrize(
        ["source_size", "output_size"],
        [
            (Size(240, 240), Size(150, 150)),
            (Size(240, 160), Size(120, 80)),
            (Size(240, 160), Size(97, 61)),
        ],
    )
    def test_downsample_stays_close_to_pillow(
        self,
        image: np.ndarray,
        resample: ResampleFilterOption,
        max_difference: int,
        max_fraction_beyond_one: float,
        source_size: Size,
        output_size: Size,
    ) -> None:
        """The same antialiased filters as Pillow are equal up to rounding.

        They are never transposed or shifted.
        """
        # Arrange
        source = image[: source_size.height, : source_size.width]
        pillow = PillowDownsampler(resample=resample)
        kernels = KernelsDownsampler(resample=resample)

        # Act
        pillow_result = pillow.downsample(source, output_size)
        kernels_result = kernels.downsample(source, output_size)

        # Assert
        difference = np.abs(pillow_result.astype(int) - kernels_result.astype(int))
        assert kernels_result.shape == pillow_result.shape
        assert difference.max() <= max_difference
        assert np.mean(difference > 1) <= max_fraction_beyond_one
