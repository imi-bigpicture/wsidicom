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

from wsidicom.codec import (
    Channels,
    Encoder,
    Jpeg2kSettings,
    JpegSettings,
    Recoder,
    Settings,
    Subsampling,
)
from wsidicom.codec.encoder import JpegEncoder, PillowEncoder
from wsidicom.codec.recoder import JpegOptimizer


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
        optimizer = JpegOptimizer(JpegSettings(optimize=True))

        # Act
        optimized_encoded = optimizer.recode(encoded)

        # Assert
        optimized = decode(optimized_encoded)
        assert np.array_equal(optimized, original)

    def test_recode_makes_image_smaller(self, image: np.ndarray) -> None:
        # Arrange
        encoded = JpegEncoder(JpegSettings()).encode(image)
        optimizer = JpegOptimizer(JpegSettings(optimize=True))

        # Act
        optimized = optimizer.recode(encoded)

        # Assert
        assert len(optimized) < len(encoded)

    def test_recode_twice_changes_nothing(self, image: np.ndarray) -> None:
        # Arrange
        optimizer = JpegOptimizer(JpegSettings(optimize=True))
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
        optimizer = JpegOptimizer(JpegSettings(optimize=True))

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
        optimizer = JpegOptimizer(JpegSettings(optimize=True))

        # Act
        supported = optimizer.supports_transfer_syntax(transfer_syntax)

        # Assert
        assert supported == expected

    def test_recoded_images_are_jpeg_baseline(self) -> None:
        # Arrange
        optimizer = JpegOptimizer(JpegSettings(optimize=True))

        # Act
        transfer_syntax = optimizer.transfer_syntax

        # Assert
        assert transfer_syntax == JPEGBaseline8Bit

    def test_matching_encoder_optimizes_jpeg_baseline_encoder(self) -> None:
        # Arrange
        optimizer = JpegOptimizer(JpegSettings(optimize=True))
        encoder = Encoder.create_for_settings(JpegSettings(quality=90))

        # Act
        matching = optimizer.matching_encoder(encoder)

        # Assert
        settings = matching.settings
        assert isinstance(settings, JpegSettings)
        assert settings.optimize
        assert settings.quality == 90

    @pytest.mark.parametrize(
        "settings",
        [JpegSettings(optimize=True), Jpeg2kSettings()],
    )
    def test_matching_encoder_keeps_other_encoders(self, settings: Settings) -> None:
        # Arrange
        optimizer = JpegOptimizer(JpegSettings(optimize=True))
        encoder = Encoder.create_for_settings(settings)

        # Act
        matching = optimizer.matching_encoder(encoder)

        # Assert
        assert matching is encoder


@pytest.mark.unittest
class TestRecoderCreateForSettings:
    @pytest.mark.skipif(not JpegOptimizer.is_available(), reason="Requires imagecodecs")
    def test_optimizing_jpeg_settings_give_jpeg_optimizer(self) -> None:
        # Arrange
        settings = JpegSettings(optimize=True)

        # Act
        recoder = Recoder.create_for_settings(settings)

        # Assert
        assert isinstance(recoder, JpegOptimizer)
        assert recoder.settings == settings

    @pytest.mark.parametrize(
        "settings",
        [JpegSettings(), JpegSettings(bits=12, optimize=True), Jpeg2kSettings()],
    )
    def test_other_settings_give_no_recoder(self, settings: Settings) -> None:
        # Arrange

        # Act
        recoder = Recoder.create_for_settings(settings)

        # Assert
        assert recoder is None


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
