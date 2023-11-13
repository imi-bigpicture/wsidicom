#    Copyright 2023 SECTRA AB
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

import pytest
from PIL import ImageChops, ImageStat
from PIL.Image import Image as PILImage
from pydicom.uid import (
    JPEG2000,
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
)

from wsidicom.codec import (
    Channels,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsLosslessSettings,
    JpegSettings,
    RleSettings,
    Subsampling,
)
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.codec.decoder import ImageCodecsDecoder, PillowDecoder, PydicomDecoder
from wsidicom.codec.settings import NumpySettings
from wsidicom.geometry import Size


@pytest.mark.unittest
class TestPillowDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (ImplicitVRLittleEndian, False),
            (ExplicitVRBigEndian, False),
            (ExplicitVRLittleEndian, False),
            (DeflatedExplicitVRLittleEndian, False),
            (RLELossless, False),
            (JPEGBaseline8Bit, True),
            (JPEGExtended12Bit, False),
            (JPEGLosslessP14, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, False),
            (JPEGLSNearLossless, False),
            (JPEG2000Lossless, True),
            (JPEG2000, True),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PillowDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        ["encoder_settings", "allowed_rms"],
        [
            (JpegSettings(95, 8, Channels.GRAYSCALE), 2),
            (JpegSettings(95, 8, Channels.YBR, Subsampling.R444), 2),
            (JpegSettings(95, 8, Channels.RGB, Subsampling.R444), 2),
            (JpegLosslessSettings(7, 8, Channels.GRAYSCALE), 0),
            (JpegLosslessSettings(7, 8, Channels.YBR), 0),
            (JpegLosslessSettings(7, 8, Channels.RGB), 0),
            (JpegLosslessSettings(1, 8, Channels.GRAYSCALE), 0),
            (JpegLosslessSettings(1, 8, Channels.YBR), 0),
            (JpegLosslessSettings(1, 8, Channels.RGB), 0),
            (Jpeg2kSettings(80, 8, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(80, 8, Channels.YBR), 1),
            (Jpeg2kSettings(80, 8, Channels.RGB), 1),
            (Jpeg2kSettings(80, 16, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(0, 8, Channels.GRAYSCALE), 0),
            (Jpeg2kSettings(0, 8, Channels.YBR), 0),
            (Jpeg2kSettings(0, 8, Channels.RGB), 0),
            (Jpeg2kSettings(0, 16, Channels.GRAYSCALE), 0),
        ],
    )
    def test_decode(
        self,
        image: PILImage,
        encoded: bytes,
        encoder_settings: EncoderSettings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = PillowDecoder()

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if encoder_settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms


@pytest.mark.unittest
class TestPydicomDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (ImplicitVRLittleEndian, True),
            (ExplicitVRBigEndian, True),
            (ExplicitVRLittleEndian, True),
            (DeflatedExplicitVRLittleEndian, True),
            (RLELossless, True),
            (JPEGBaseline8Bit, True),
            (JPEGExtended12Bit, True),
            (JPEGLosslessP14, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, False),
            (JPEGLSNearLossless, False),
            (JPEG2000Lossless, True),
            (JPEG2000, True),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PydicomDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        "encoder_settings",
        [
            RleSettings(8, Channels.GRAYSCALE),
            NumpySettings(8, Channels.GRAYSCALE, True, True),
            NumpySettings(8, Channels.GRAYSCALE, True, False),
            NumpySettings(8, Channels.GRAYSCALE, False, True),
            NumpySettings(16, Channels.GRAYSCALE, True, True),
            NumpySettings(16, Channels.GRAYSCALE, True, False),
            NumpySettings(16, Channels.GRAYSCALE, False, True),
            NumpySettings(8, Channels.RGB, True, True),
            NumpySettings(8, Channels.RGB, True, False),
            NumpySettings(8, Channels.RGB, False, True),
        ],
    )
    def test_decode(
        self, image: PILImage, encoded: bytes, encoder_settings: EncoderSettings
    ):
        # Arrange
        if isinstance(encoder_settings, NumpySettings):
            pixel_representation = encoder_settings.pixel_representation
        else:
            pixel_representation = 0
        decoder = PydicomDecoder(
            encoder_settings.transfer_syntax,
            Size(image.width, image.height),
            1 if encoder_settings.channels == Channels.GRAYSCALE else 3,
            encoder_settings.bits,
            encoder_settings.bits,
            encoder_settings.photometric_interpretation,
            pixel_representation,
            0,
        )

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if encoder_settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms == 0


@pytest.mark.unittest
class TestImageCodecsDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (ImplicitVRLittleEndian, False),
            (ExplicitVRBigEndian, False),
            (ExplicitVRLittleEndian, False),
            (DeflatedExplicitVRLittleEndian, False),
            (RLELossless, False),
            (JPEGBaseline8Bit, True),
            (JPEGExtended12Bit, True),
            (JPEGLosslessP14, True),
            (JPEGLosslessSV1, True),
            (JPEGLSLossless, True),
            (JPEGLSNearLossless, True),
            (JPEG2000Lossless, True),
            (JPEG2000, True),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = ImageCodecsDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        ["encoder_settings", "allowed_rms"],
        [
            (JpegSettings(95, 8, Channels.GRAYSCALE), 2),
            (JpegSettings(95, 8, Channels.YBR, Subsampling.R444), 2),
            (JpegSettings(95, 8, Channels.RGB, Subsampling.R444), 2),
            (JpegSettings(95, 12, Channels.GRAYSCALE), 2),
            (JpegLosslessSettings(7, 8, Channels.GRAYSCALE), 0),
            (JpegLosslessSettings(7, 8, Channels.YBR), 0),
            (JpegLosslessSettings(7, 8, Channels.RGB), 0),
            (JpegLosslessSettings(7, 16, Channels.GRAYSCALE), 0),
            (JpegLosslessSettings(1, 8, Channels.GRAYSCALE), 0),
            (JpegLosslessSettings(1, 8, Channels.YBR), 0),
            (JpegLosslessSettings(1, 8, Channels.RGB), 0),
            (JpegLosslessSettings(1, 16, Channels.GRAYSCALE), 0),
            (JpegLsLosslessSettings(0, 8), 0),
            (JpegLsLosslessSettings(0, 16), 0),
            (JpegLsLosslessSettings(1, 8), 1),
            (JpegLsLosslessSettings(1, 16), 1),
            (Jpeg2kSettings(80, 8, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(80, 8, Channels.YBR), 1),
            (Jpeg2kSettings(80, 8, Channels.RGB), 1),
            (Jpeg2kSettings(80, 16, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(0, 8, Channels.GRAYSCALE), 0),
            (Jpeg2kSettings(0, 8, Channels.YBR), 0),
            (Jpeg2kSettings(0, 8, Channels.RGB), 0),
            (Jpeg2kSettings(0, 16, Channels.GRAYSCALE), 0),
        ],
    )
    def test_decode(
        self,
        image: PILImage,
        encoded: bytes,
        encoder_settings: EncoderSettings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = ImageCodecsDecoder(encoder_settings.transfer_syntax)
        if not decoder.is_available():
            pytest.skip("ImageCodecs is not available")

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if encoder_settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms
