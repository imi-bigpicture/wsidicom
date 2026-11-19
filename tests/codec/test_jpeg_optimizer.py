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

"""Tests for JPEG Huffman table optimization."""

import io

import numpy as np
import pytest
from PIL import Image as Pillow
from pydicom.uid import (
    JPEG2000,
    UID,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessSV1,
)
from wsidicom_data import TestData

from wsidicom.codec import Channels, JpegOptimizer, JpegSettings, Subsampling
from wsidicom.codec.encoder import JpegEncoder, PillowEncoder


@pytest.fixture
def image() -> np.ndarray:
    """Return an 8-bit RGB test image."""
    return np.asarray(TestData.image(8, 3))


def decode(encoded: bytes) -> np.ndarray:
    """Decode a JPEG image to pixels."""
    with Pillow.open(io.BytesIO(encoded)) as decoded:
        return np.asarray(decoded)


@pytest.mark.unittest
@pytest.mark.skipif(not JpegOptimizer.is_available(), reason="Requires imagecodecs")
class TestJpegOptimizer:
    def test_recode_keeps_pixels(self, image: np.ndarray) -> None:
        # Arrange
        encoded = JpegEncoder(JpegSettings()).encode(image)
        original = decode(encoded)
        optimizer = JpegOptimizer()

        # Act
        optimized_encoded = optimizer.recode(encoded)

        # Assert
        optimized = decode(optimized_encoded)
        assert np.array_equal(optimized, original)

    def test_recode_makes_image_smaller(self, image: np.ndarray) -> None:
        # Arrange
        encoded = JpegEncoder(JpegSettings()).encode(image)
        optimizer = JpegOptimizer()

        # Act
        optimized = optimizer.recode(encoded)

        # Assert
        assert len(optimized) < len(encoded)

    def test_recode_twice_changes_nothing(self, image: np.ndarray) -> None:
        # Arrange
        optimizer = JpegOptimizer()
        optimized = optimizer.recode(JpegEncoder(JpegSettings()).encode(image))

        # Act
        optimized_again = optimizer.recode(optimized)

        # Assert
        assert optimized_again == optimized

    def test_recode_makes_progressive_image_baseline(self, image: np.ndarray) -> None:
        # Arrange
        buffer = io.BytesIO()
        Pillow.fromarray(image).save(buffer, format="jpeg", progressive=True)
        progressive = buffer.getvalue()
        original = decode(progressive)
        optimizer = JpegOptimizer()

        # Act
        optimized_encoded = optimizer.recode(progressive)

        # Assert
        optimized = decode(optimized_encoded)
        # A marker byte 0xFF is never followed by a non-zero byte in entropy-coded
        # data, other than by a restart marker, so the frame markers can be found.
        assert b"\xff\xc2" in progressive
        assert b"\xff\xc2" not in optimized_encoded
        assert b"\xff\xc0" in optimized_encoded
        assert np.array_equal(optimized, original)

    @pytest.mark.parametrize(
        ["transfer_syntax", "expected"],
        [
            (JPEGBaseline8Bit, True),
            (JPEGExtended12Bit, False),
            (JPEGLosslessSV1, False),
            (JPEG2000, False),
            (ExplicitVRLittleEndian, False),
        ],
    )
    def test_supports_only_jpeg_baseline(
        self, transfer_syntax: UID, expected: bool
    ) -> None:
        # Arrange

        # Act
        supported = JpegOptimizer.supports_transfer_syntax(transfer_syntax)

        # Assert
        assert supported == expected


@pytest.mark.unittest
class TestJpegSettingsOptimize:
    def test_optimize_defaults_to_false(self) -> None:
        # Arrange

        # Act
        settings = JpegSettings()

        # Assert
        assert settings.optimize is False

    def test_with_optimize_keeps_other_settings(self) -> None:
        # Arrange
        settings = JpegSettings(
            quality=90, bits=8, channels=Channels.RGB, subsampling=Subsampling.R444
        )

        # Act
        optimized = settings.with_optimize(True)

        # Assert
        assert optimized.optimize is True
        assert optimized.quality == settings.quality
        assert optimized.bits == settings.bits
        assert optimized.channels == settings.channels
        assert optimized.subsampling == settings.subsampling
        assert settings.optimize is False


@pytest.mark.unittest
class TestOptimizingJpegEncoders:
    @pytest.mark.parametrize("encoder_type", [JpegEncoder, PillowEncoder])
    def test_optimized_encoding_keeps_pixels(
        self, encoder_type: type[JpegEncoder | PillowEncoder], image: np.ndarray
    ) -> None:
        # Arrange
        if not encoder_type.is_available():
            pytest.skip(f"{encoder_type.__name__} not available")
        settings = JpegSettings()
        default = decode(encoder_type(settings).encode(image))

        # Act
        optimized_encoded = encoder_type(settings.with_optimize(True)).encode(image)

        # Assert
        optimized = decode(optimized_encoded)
        assert np.array_equal(optimized, default)

    @pytest.mark.parametrize("encoder_type", [JpegEncoder, PillowEncoder])
    def test_optimized_encoding_is_smaller(
        self, encoder_type: type[JpegEncoder | PillowEncoder], image: np.ndarray
    ) -> None:
        # Arrange
        if not encoder_type.is_available():
            pytest.skip(f"{encoder_type.__name__} not available")
        settings = JpegSettings()
        default = encoder_type(settings).encode(image)

        # Act
        optimized = encoder_type(settings.with_optimize(True)).encode(image)

        # Assert
        assert len(optimized) < len(default)
