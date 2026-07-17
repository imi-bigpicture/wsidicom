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
runs it against both ``PillowDownsampler`` and ``Cv2Downsampler`` (the latter
skipped when opencv is absent). Behaviour specific to one backend lives in its
own class.
"""

import numpy as np
import pytest

from wsidicom.config import settings
from wsidicom.downsampler import (
    Cv2Downsampler,
    Downsampler,
    PillowDownsampler,
    ResampleFilterOption,
)
from wsidicom.geometry import Size
from wsidicom.options import DownsamplerOption


@pytest.fixture
def image_array(
    width: int, height: int, color: int | tuple[int, int, int], dtype: np.dtype
) -> np.ndarray:
    """An image from an indirect ``(width, height, color, dtype)`` param. An int
    color gives a 2D grayscale array; an RGB tuple gives a ``(h, w, 3)`` array."""
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
    ],
    ids=["pillow", "cv2"],
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
    ):
        """Downsampling yields the requested size and dtype, preserving the
        channel layout (arrays are rows, columns[, samples])."""
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
    ):
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
    ):
        """Thumbnail fits within max_size and preserves dtype, aspect ratio, and
        never upscales."""
        # Act
        result = downsampler.thumbnail(image_array, max_size)

        # Assert
        assert result.shape[:2] == expected_shape
        assert result.dtype == dtype


@pytest.mark.unittest
class TestPillowDownsampler:
    def test_downsample_with_custom_resampling(self):
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
            (ResampleFilterOption.BILINEAR, True),
            (ResampleFilterOption.BICUBIC, False),
            (ResampleFilterOption.NEAREST, False),
            (ResampleFilterOption.LANCZOS, False),
        ],
    )
    def test_commutes_with_stitch(self, resample: ResampleFilterOption, expected: bool):
        """commutes_with_stitch is True only for BOX and BILINEAR."""
        # Arrange
        downsampler = PillowDownsampler(resample=resample)

        # Act & Assert
        assert downsampler.commutes_with_stitch is expected


@pytest.mark.unittest
@pytest.mark.skipif(not Cv2Downsampler.is_available(), reason="opencv not installed")
class TestCv2Downsampler:
    def test_commutes_with_stitch_true_for_inter_area(self):
        """commutes_with_stitch is True for the default INTER_AREA filter."""
        # Act & Assert
        assert Cv2Downsampler().commutes_with_stitch is True

    def test_commutes_with_stitch_false_for_inter_linear(self):
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
            (ResampleFilterOption.NEAREST, "INTER_NEAREST"),
            (ResampleFilterOption.BOX, "INTER_AREA"),
            (ResampleFilterOption.BILINEAR, "INTER_AREA"),
            (ResampleFilterOption.BICUBIC, "INTER_CUBIC"),
            (ResampleFilterOption.LANCZOS, "INTER_LANCZOS4"),
        ],
    )
    def test_maps_filter_to_cv2_equivalent(
        self, resample: ResampleFilterOption, cv2_attr: str
    ):
        # Arrange
        import cv2

        # Act
        downsampler = Cv2Downsampler.from_filter(resample)

        # Assert
        assert downsampler is not None
        assert downsampler._interpolation == getattr(cv2, cv2_attr)

    def test_filter_without_equivalent_returns_none(self):
        # HAMMING has no cv2 equivalent, so Pillow must handle it.
        # Act & Assert
        assert Cv2Downsampler.from_filter(ResampleFilterOption.HAMMING) is None


@pytest.mark.unittest
class TestDownsamplerCreate:
    """Downsampler.create selects the backend from the preference argument,
    falling back to settings.preferred_downsampler when it is None."""

    @pytest.fixture(autouse=True)
    def _restore_setting(self):
        original = settings.preferred_downsampler
        yield
        settings.preferred_downsampler = original

    def test_pillow_argument_forces_pillow(self):
        # Act & Assert
        assert isinstance(
            Downsampler.create(preferred=DownsamplerOption.PILLOW), PillowDownsampler
        )

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_opencv_argument_selects_cv2(self):
        # Act & Assert
        assert isinstance(
            Downsampler.create(preferred=DownsamplerOption.OPENCV), Cv2Downsampler
        )

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_argument_overrides_settings(self):
        # Arrange
        settings.preferred_downsampler = DownsamplerOption.PILLOW

        # Act & Assert — the argument wins over the setting
        assert isinstance(
            Downsampler.create(preferred=DownsamplerOption.OPENCV), Cv2Downsampler
        )

    def test_defaults_to_settings(self):
        # Arrange
        settings.preferred_downsampler = DownsamplerOption.PILLOW

        # Act & Assert — no argument falls back to the setting
        assert isinstance(Downsampler.create(), PillowDownsampler)

    @pytest.mark.skipif(
        not Cv2Downsampler.is_available(), reason="opencv not installed"
    )
    def test_none_setting_prefers_cv2_when_available(self):
        # Arrange
        settings.preferred_downsampler = None

        # Act & Assert
        assert isinstance(Downsampler.create(), Cv2Downsampler)

    def test_unknown_name_raises(self):
        # Act & Assert
        with pytest.raises(ValueError):
            settings.preferred_downsampler = "bogus"
