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
from PIL.Image import Image
from pydicom.uid import (
    HTJ2K,
    JPEG2000,
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    HTJ2KLossless,
    HTJ2KLosslessRPCL,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLossless,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
)

from wsidicom.codec import (
    Channels,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsSettings,
    JpegSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)
from wsidicom.codec.decoder import (
    ImageCodecsDecoder,
    ImageCodecsRleDecoder,
    PillowDecoder,
    PydicomDecoder,
    PyJpegLsDecoder,
    PyLibJpegOpenJpegDecoder,
    PylibjpegRleDecoder,
)
from wsidicom.codec.optionals import (
    IMAGE_CODECS_AVAILABLE,
    PYLIBJPEGLS_AVAILABLE,
    PYLIBJPEGOPENJPEG_AVAILABLE,
)
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
            (JPEGLossless, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, False),
            (JPEGLSNearLossless, False),
            (JPEG2000Lossless, True),
            (JPEG2000, True),
            (HTJ2KLossless, True),
            (HTJ2K, True),
            (HTJ2KLosslessRPCL, True),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PillowDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        ["settings", "allowed_rms"],
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
        image: Image,
        encoded: bytes,
        settings: Settings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = PillowDecoder()

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
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
            (JPEGLossless, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, PYLIBJPEGLS_AVAILABLE),
            (JPEGLSNearLossless, PYLIBJPEGLS_AVAILABLE),
            (JPEG2000Lossless, True),
            (JPEG2000, True),
            (HTJ2KLossless, False),
            (HTJ2K, False),
            (HTJ2KLosslessRPCL, False),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PydicomDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        "settings",
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
    def test_decode(self, image: Image, encoded: bytes, settings: Settings):
        # Arrange
        decoder = PydicomDecoder(
            settings.transfer_syntax,
            Size(image.width, image.height),
            1 if settings.channels == Channels.GRAYSCALE else 3,
            settings.bits,
            settings.bits,
            settings.photometric_interpretation,
        )

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
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
            (JPEGBaseline8Bit, IMAGE_CODECS_AVAILABLE),
            (JPEGExtended12Bit, IMAGE_CODECS_AVAILABLE),
            (JPEGLossless, IMAGE_CODECS_AVAILABLE),
            (JPEGLosslessSV1, IMAGE_CODECS_AVAILABLE),
            (JPEGLSLossless, IMAGE_CODECS_AVAILABLE),
            (JPEGLSNearLossless, IMAGE_CODECS_AVAILABLE),
            (JPEG2000Lossless, IMAGE_CODECS_AVAILABLE),
            (JPEG2000, IMAGE_CODECS_AVAILABLE),
            (HTJ2KLossless, IMAGE_CODECS_AVAILABLE),
            (HTJ2K, IMAGE_CODECS_AVAILABLE),
            (HTJ2KLosslessRPCL, IMAGE_CODECS_AVAILABLE),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = ImageCodecsDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.skipif(
        not ImageCodecsDecoder.is_available(), reason="Image codecs not available"
    )
    @pytest.mark.parametrize(
        ["settings", "allowed_rms"],
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
            (JpegLsSettings(0, 8, Channels.GRAYSCALE), 0),
            (JpegLsSettings(0, 8, Channels.RGB), 0),
            (JpegLsSettings(0, 16, Channels.GRAYSCALE), 0),
            (JpegLsSettings(1, 8, Channels.GRAYSCALE), 1),
            (JpegLsSettings(1, 8, Channels.RGB), 1),
            (JpegLsSettings(1, 16, Channels.GRAYSCALE), 1),
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
        image: Image,
        encoded: bytes,
        settings: Settings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = ImageCodecsDecoder(settings.transfer_syntax)
        if not decoder.is_available():
            pytest.skip("ImageCodecs is not available")

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms


@pytest.mark.unittest
class TestPylibjpegRleDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (ImplicitVRLittleEndian, False),
            (ExplicitVRBigEndian, False),
            (ExplicitVRLittleEndian, False),
            (DeflatedExplicitVRLittleEndian, False),
            (RLELossless, True),
            (JPEGBaseline8Bit, False),
            (JPEGExtended12Bit, False),
            (JPEGLossless, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, False),
            (JPEGLSNearLossless, False),
            (JPEG2000Lossless, False),
            (JPEG2000, False),
            (HTJ2KLossless, False),
            (HTJ2K, False),
            (HTJ2KLosslessRPCL, False),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PylibjpegRleDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.skipif(
        not PylibjpegRleDecoder.is_available(), reason="Pylibjpeg-rle not available"
    )
    @pytest.mark.parametrize(
        "settings",
        [
            RleSettings(8, Channels.GRAYSCALE),
            RleSettings(8, Channels.RGB),
            RleSettings(16, Channels.GRAYSCALE),
        ],
    )
    def test_decode(self, image: Image, encoded: bytes, settings: Settings):
        # Arrange
        decoder = PylibjpegRleDecoder(
            Size(image.width, image.height),
            1 if settings.channels == Channels.GRAYSCALE else 3,
            settings.bits,
        )

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms == 0


@pytest.mark.unittest
class TestImagecodecsRleDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (ImplicitVRLittleEndian, False),
            (ExplicitVRBigEndian, False),
            (ExplicitVRLittleEndian, False),
            (DeflatedExplicitVRLittleEndian, False),
            (RLELossless, True),
            (JPEGBaseline8Bit, False),
            (JPEGExtended12Bit, False),
            (JPEGLossless, False),
            (JPEGLosslessSV1, False),
            (JPEGLSLossless, False),
            (JPEGLSNearLossless, False),
            (JPEG2000Lossless, False),
            (JPEG2000, False),
            (HTJ2KLossless, False),
            (HTJ2K, False),
            (HTJ2KLosslessRPCL, False),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = ImageCodecsRleDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.parametrize(
        "settings",
        [
            RleSettings(8, Channels.GRAYSCALE),
            RleSettings(8, Channels.RGB),
            RleSettings(16, Channels.GRAYSCALE),
        ],
    )
    def test_decode(self, image: Image, encoded: bytes, settings: Settings):
        # Arrange
        decoder = ImageCodecsRleDecoder(
            Size(image.width, image.height),
            1 if settings.channels == Channels.GRAYSCALE else 3,
            settings.bits,
        )

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms == 0


@pytest.mark.unittest
class TestPyJpegLsDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (JPEGLSLossless, PyJpegLsDecoder.is_available()),
            (JPEGLSNearLossless, PyJpegLsDecoder.is_available()),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PyJpegLsDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.skipif(
        not PyJpegLsDecoder.is_available(), reason="PyJpegLs codecs not available"
    )
    @pytest.mark.parametrize(
        ["settings", "allowed_rms"],
        [
            (JpegLsSettings(0, 8, Channels.GRAYSCALE), 0),
            (JpegLsSettings(0, 8, Channels.RGB), 0),
            (JpegLsSettings(0, 16, Channels.GRAYSCALE), 0),
            (JpegLsSettings(1, 8, Channels.GRAYSCALE), 1),
            (JpegLsSettings(1, 8, Channels.RGB), 1),
            (JpegLsSettings(1, 16, Channels.GRAYSCALE), 1),
        ],
    )
    def test_decode(
        self,
        image: Image,
        encoded: bytes,
        settings: Settings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = PyJpegLsDecoder()
        if not decoder.is_available():
            pytest.skip("PypegLs is not available")

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms


@pytest.mark.unittest
class TestPyLibJpegOpenJpegDecoder:
    @pytest.mark.parametrize(
        ["transfer_syntax", "expected_result"],
        [
            (JPEG2000Lossless, PYLIBJPEGOPENJPEG_AVAILABLE),
            (JPEG2000, PYLIBJPEGOPENJPEG_AVAILABLE),
            (HTJ2KLossless, PYLIBJPEGOPENJPEG_AVAILABLE),
            (HTJ2K, PYLIBJPEGOPENJPEG_AVAILABLE),
            (HTJ2KLosslessRPCL, PYLIBJPEGOPENJPEG_AVAILABLE),
        ],
    )
    def test_is_supported(self, transfer_syntax: UID, expected_result: bool):
        # Arrange

        # Act
        is_supported = PyLibJpegOpenJpegDecoder.is_supported(transfer_syntax, 3, 8)

        # Assert
        assert is_supported == expected_result

    @pytest.mark.skipif(
        not PyLibJpegOpenJpegDecoder.is_available(),
        reason="OpenJpeg codecs not available",
    )
    @pytest.mark.parametrize(
        ["settings", "allowed_rms"],
        [
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
        image: Image,
        encoded: bytes,
        settings: Settings,
        allowed_rms: float,
    ):
        # Arrange
        decoder = PyLibJpegOpenJpegDecoder()
        if not decoder.is_available():
            pytest.skip("OpenJpeg is not available")

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms
