import pytest
from PIL import Image, ImageChops, ImageStat
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
    Encoder,
    Jpeg2kLosslessSettings,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsLosslessSettings,
    JpegLsNearLosslessSettings,
    JpegSettings,
    RleSettings,
    Subsampling,
)
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.codec.decoder import ImageCodecsDecoder, PillowDecoder, PydicomDecoder
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
        ],
    )
    def test_decode(self, encoder_settings: EncoderSettings):
        # Arrange
        image = Image.open("tests/testdata/test_tile.png")
        if encoder_settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
        encoder = Encoder.create(encoder_settings)
        encoded = encoder.encode(image)
        decoder = PydicomDecoder(
            encoder_settings.transfer_syntax,
            Size(image.width, image.height),
            1 if encoder_settings.channels == Channels.GRAYSCALE else 3,
            encoder_settings.bits,
            encoder_settings.bits,
            encoder_settings.photometric_interpretation,
            1,
            0,
        )

        # Act
        decoded = decoder.decode(encoded)

        # Assert

        if encoder_settings.channels == Channels.GRAYSCALE:
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms < 2


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
        "encoder_settings",
        [
            JpegSettings(8, Channels.GRAYSCALE, quality=95),
            JpegSettings(8, Channels.YBR, subsampling=Subsampling.R444, quality=95),
            JpegSettings(8, Channels.RGB, subsampling=Subsampling.R444, quality=95),
            JpegSettings(12, Channels.GRAYSCALE, quality=95),
            JpegLosslessSettings(8, Channels.GRAYSCALE, predictor=7),
            JpegLosslessSettings(8, Channels.YBR, predictor=7),
            JpegLosslessSettings(8, Channels.RGB, predictor=7),
            JpegLosslessSettings(16, Channels.GRAYSCALE, predictor=7),
            JpegLosslessSettings(8, Channels.GRAYSCALE, predictor=None),
            JpegLosslessSettings(8, Channels.YBR, predictor=None),
            JpegLosslessSettings(8, Channels.RGB, predictor=None),
            JpegLosslessSettings(16, Channels.GRAYSCALE, predictor=None),
            JpegLsLosslessSettings(8),
            JpegLsLosslessSettings(16),
            JpegLsNearLosslessSettings(8),
            JpegLsNearLosslessSettings(16),
            Jpeg2kSettings(8, Channels.GRAYSCALE),
            Jpeg2kSettings(8, Channels.YBR),
            Jpeg2kSettings(8, Channels.RGB),
            Jpeg2kSettings(16, Channels.GRAYSCALE),
            Jpeg2kLosslessSettings(8, Channels.GRAYSCALE),
            Jpeg2kLosslessSettings(8, Channels.YBR),
            Jpeg2kLosslessSettings(8, Channels.RGB),
            Jpeg2kLosslessSettings(16, Channels.GRAYSCALE),
        ],
    )
    def test_decode(self, encoder_settings: EncoderSettings):
        # Arrange
        image = Image.open("tests/testdata/test_tile.png")
        if encoder_settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
        encoder = Encoder.create(encoder_settings)
        encoded = encoder.encode(image)
        # with open(
        #     rf"c:\temp\{encoder_settings.transfer_syntax.name}-{encoder_settings.channels.name}-{encoder_settings.bits}.{encoder_settings.extension}",
        #     "wb",
        # ) as f:
        #     f.write(encoded)
        decoder = ImageCodecsDecoder(encoder_settings.transfer_syntax)

        # Act
        decoded = decoder.decode(encoded)

        # Assert
        # decoded.save(
        #     rf"c:\temp\{encoder_settings.transfer_syntax.name}-{encoder_settings.channels.name}-{encoder_settings.bits}.png"
        # )
        if encoder_settings.channels == Channels.GRAYSCALE:
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms < 2
