#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Tests for Downsampler."""

import pytest
from PIL import Image

from wsidicom.downsampler import PillowDownsampler
from wsidicom.geometry import Size


@pytest.fixture
def downsampler() -> PillowDownsampler:
    """Create a PillowDownsampler instance."""
    return PillowDownsampler()


@pytest.mark.unittest
class TestPillowDownsampler:
    def test_downsample_to_half_size(self, downsampler: PillowDownsampler):
        """Test downsampling a 512x512 image to 256x256."""
        # Arrange
        image = Image.new("RGB", (512, 512), color=(255, 0, 0))
        output_size = Size(256, 256)

        # Act
        result = downsampler.downsample(image, output_size)

        # Assert
        assert result.size == (256, 256)
        assert result.mode == "RGB"

    def test_downsample_non_square(self, downsampler: PillowDownsampler):
        """Test downsampling a non-square image."""
        # Arrange
        image = Image.new("RGB", (512, 256), color=(0, 255, 0))
        output_size = Size(256, 128)

        # Act
        result = downsampler.downsample(image, output_size)

        # Assert
        assert result.size == (256, 128)

    def test_downsample_preserves_mode_grayscale(self, downsampler: PillowDownsampler):
        """Test that downsampling preserves the image mode."""
        # Arrange
        image = Image.new("L", (256, 256), color=128)
        output_size = Size(128, 128)

        # Act
        result = downsampler.downsample(image, output_size)

        # Assert
        assert result.mode == "L"

    def test_downsample_odd_dimensions(self, downsampler: PillowDownsampler):
        """Test downsampling an image with odd dimensions."""
        # Arrange
        image = Image.new("RGB", (511, 255), color=(0, 0, 255))
        output_size = Size(255, 127)

        # Act
        result = downsampler.downsample(image, output_size)

        # Assert
        assert result.size == (255, 127)

    def test_downsample_with_custom_resampling(self):
        """The configured resampling filter is actually applied."""
        # Arrange — a non-uniform image so the filter choice changes the result
        image = Image.new("RGB", (512, 512), color=(0, 0, 0))
        image.paste((255, 255, 255), (0, 0, 256, 512))  # sharp vertical edge
        output_size = Size(256, 256)

        # Act — same image downsampled with two different filters
        nearest = PillowDownsampler(resample=Image.Resampling.NEAREST).downsample(
            image, output_size
        )
        bilinear = PillowDownsampler(resample=Image.Resampling.BILINEAR).downsample(
            image, output_size
        )

        # Assert — both yield the requested size, and the filter genuinely
        # matters (NEAREST differs from BILINEAR at the edge); a solid image
        # would pass even if the filter were ignored.
        assert nearest.size == (256, 256)
        assert bilinear.size == (256, 256)
        assert nearest.tobytes() != bilinear.tobytes()

    def test_thumbnail_preserves_aspect_ratio(self, downsampler: PillowDownsampler):
        """Thumbnail fits within max_size while preserving aspect ratio."""
        # Arrange
        image = Image.new("RGB", (512, 256), color=(0, 255, 0))
        max_size = Size(128, 128)

        # Act
        result = downsampler.thumbnail(image, max_size)

        # Assert — fits within the box, aspect ratio preserved (2:1)
        assert result.size == (128, 64)

    def test_thumbnail_does_not_upscale(self, downsampler: PillowDownsampler):
        """Thumbnail never upscales an image smaller than max_size."""
        # Arrange
        image = Image.new("RGB", (64, 64), color=(0, 0, 255))
        max_size = Size(256, 256)

        # Act
        result = downsampler.thumbnail(image, max_size)

        # Assert
        assert result.size == (64, 64)

    def test_commutes_with_stitch_true_for_box(self):
        """Test commutes_with_stitch is True for BOX filter."""
        # Arrange
        downsampler = PillowDownsampler(resample=Image.Resampling.BOX)

        # Act & Assert
        assert downsampler.commutes_with_stitch is True

    def test_commutes_with_stitch_true_for_bilinear(self):
        """Test commutes_with_stitch is True for BILINEAR filter."""
        # Arrange
        downsampler = PillowDownsampler(resample=Image.Resampling.BILINEAR)

        # Act & Assert
        assert downsampler.commutes_with_stitch is True

    def test_commutes_with_stitch_false_for_bicubic(self):
        """Test commutes_with_stitch is False for BICUBIC filter."""
        # Arrange
        downsampler = PillowDownsampler(resample=Image.Resampling.BICUBIC)

        # Act & Assert
        assert downsampler.commutes_with_stitch is False

    def test_commutes_with_stitch_false_for_nearest(self):
        """Test commutes_with_stitch is False for NEAREST filter."""
        # Arrange
        downsampler = PillowDownsampler(resample=Image.Resampling.NEAREST)

        # Act & Assert
        assert downsampler.commutes_with_stitch is False

    def test_commutes_with_stitch_false_for_lanczos(self):
        """Test commutes_with_stitch is False for LANCZOS filter."""
        # Arrange
        downsampler = PillowDownsampler(resample=Image.Resampling.LANCZOS)

        # Act & Assert
        assert downsampler.commutes_with_stitch is False
