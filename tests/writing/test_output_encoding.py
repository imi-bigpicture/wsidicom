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

"""Tests for resolving how the tiles of a source end up encoded in the output."""

import pytest
from decoy import Decoy
from pydicom.uid import UID, JPEG2000Lossless, JPEGBaseline8Bit

from wsidicom.codec import (
    Channels,
    Encoder,
    Jpeg2kSettings,
    JpegSettings,
    Settings,
)
from wsidicom.codec.recoder import JpegOptimizer
from wsidicom.instance import ImageData
from wsidicom.options import TranscodingMode
from wsidicom.writing.output_encoding import (
    CopiedEncoding,
    EncodingResolver,
    RecodedEncoding,
    TranscodedEncoding,
)


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.fixture
def jpeg_encoder() -> Encoder:
    """Return a JPEG Baseline encoder that does not optimize its tables."""
    return Encoder.create_for_settings(JpegSettings())


@pytest.fixture
def jpeg_2000_encoder() -> Encoder:
    """Return a JPEG 2000 encoder."""
    return Encoder.create_for_settings(Jpeg2kSettings())


def create_image_data(
    decoy: Decoy,
    encoder: Encoder,
    transfer_syntax: UID,
    transcoder: Encoder | None = None,
    bits: int = 8,
    samples_per_pixel: int = 3,
) -> ImageData:
    """Return mocked image data with the encoding and pixel format given."""
    image_data = decoy.mock(cls=ImageData)
    decoy.when(image_data.encoder).then_return(encoder)
    decoy.when(image_data.transfer_syntax).then_return(transfer_syntax)
    decoy.when(image_data.transcoder).then_return(transcoder)
    decoy.when(image_data.bits).then_return(bits)
    decoy.when(image_data.samples_per_pixel).then_return(samples_per_pixel)
    return image_data


@pytest.mark.unittest
class TestEncodingResolver:
    def test_without_transcoder_tiles_are_copied(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        resolver = EncodingResolver(None, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == CopiedEncoding(jpeg_encoder, JPEGBaseline8Bit)
        assert encoding.transcoder is None

    def test_transcoder_in_same_transfer_syntax_copies_tiles(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(quality=90))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == CopiedEncoding(transcoder, JPEGBaseline8Bit)

    def test_generated_tiles_of_copied_source_are_encoded_with_transcoder(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(quality=95))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, CopiedEncoding)
        assert encoding.generated_tiles_encoder is transcoder

    def test_generated_tiles_fall_back_to_source_encoder_if_transcoder_can_not(
        self, decoy: Decoy, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        # The transcoder is in the transfer syntax of the source, so the tiles
        # are copied, but it encodes 16-bit pixels and the source has 8 bits.
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        transcoder = Encoder.create_for_settings(
            Jpeg2kSettings(levels=0, bits=16, channels=Channels.GRAYSCALE)
        )
        assert transcoder.transfer_syntax == JPEG2000Lossless
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, CopiedEncoding)
        assert encoding.generated_tiles_encoder is jpeg_2000_encoder

    def test_transcoder_in_other_transfer_syntax_transcodes_tiles(
        self, decoy: Decoy, jpeg_encoder: Encoder, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        resolver = EncodingResolver(jpeg_encoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == TranscodedEncoding(jpeg_encoder, JPEGBaseline8Bit)
        assert encoding.transcoder is jpeg_encoder

    def test_settings_are_transcoded_with_encoder_created_for_them(
        self, decoy: Decoy, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        settings = JpegSettings(quality=90)
        resolver = EncodingResolver(settings, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, TranscodedEncoding)
        assert encoding.generated_tiles_encoder.settings == settings
        assert encoding.transfer_syntax == JPEGBaseline8Bit

    def test_always_transcoding_transcodes_tiles_in_same_transfer_syntax(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(quality=90))
        resolver = EncodingResolver(transcoder, TranscodingMode.ALWAYS)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == TranscodedEncoding(transcoder, JPEGBaseline8Bit)

    def test_image_data_that_needs_transcoding_is_transcoded(
        self, decoy: Decoy, jpeg_encoder: Encoder, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(
            decoy, jpeg_2000_encoder, JPEG2000Lossless, transcoder=jpeg_encoder
        )
        resolver = EncodingResolver(None, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == TranscodedEncoding(jpeg_encoder, JPEGBaseline8Bit)

    def test_transcoding_when_required_copies_tiles_in_other_transfer_syntax(
        self, decoy: Decoy, jpeg_encoder: Encoder, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        resolver = EncodingResolver(jpeg_encoder, TranscodingMode.WHEN_REQUIRED)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == CopiedEncoding(jpeg_encoder, JPEG2000Lossless)

    def test_transcoding_when_required_transcodes_image_data_that_needs_it(
        self, decoy: Decoy, jpeg_encoder: Encoder, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(
            decoy, jpeg_2000_encoder, JPEG2000Lossless, transcoder=jpeg_2000_encoder
        )
        resolver = EncodingResolver(jpeg_encoder, TranscodingMode.WHEN_REQUIRED)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == TranscodedEncoding(jpeg_encoder, JPEGBaseline8Bit)

    def test_transcoder_with_other_bits_raises(
        self, decoy: Decoy, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        transcoder = Encoder.create_for_settings(JpegSettings(bits=12))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act & Assert
        with pytest.raises(ValueError, match="must match image data bits"):
            resolver.resolve(image_data)

    def test_settings_are_adapted_to_each_pixel_format(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        gray = create_image_data(
            decoy, jpeg_encoder, JPEGBaseline8Bit, bits=16, samples_per_pixel=1
        )
        color = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        resolver = EncodingResolver(Jpeg2kSettings(), TranscodingMode.MATCH)

        # Act
        gray_encoder = resolver.resolve(gray).transcoder
        color_encoder = resolver.resolve(color).transcoder

        # Assert
        assert gray_encoder is not None
        assert (gray_encoder.bits, gray_encoder.channels) == (16, Channels.GRAYSCALE)
        assert color_encoder is not None
        assert (color_encoder.bits, color_encoder.channels) == (8, Channels.YBR)

    def test_generated_tiles_encoder_is_adapted_to_the_pixel_format(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(
            decoy, jpeg_encoder, JPEGBaseline8Bit, samples_per_pixel=1
        )
        resolver = EncodingResolver(JpegSettings(), TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, CopiedEncoding)
        assert encoding.generated_tiles_encoder.channels == Channels.GRAYSCALE

    def test_encoder_is_not_adapted(self, decoy: Decoy, jpeg_encoder: Encoder) -> None:
        # Arrange
        image_data = create_image_data(
            decoy, jpeg_encoder, JPEGBaseline8Bit, samples_per_pixel=1
        )
        resolver = EncodingResolver(jpeg_encoder, TranscodingMode.ALWAYS)

        # Act & Assert
        with pytest.raises(ValueError, match="must match image data bits"):
            resolver.resolve(image_data)

    @pytest.mark.parametrize(
        "settings",
        [JpegSettings(), Jpeg2kSettings(bits=8), Jpeg2kSettings(channels=Channels.YBR)],
    )
    def test_settings_that_can_not_be_adapted_raise(
        self, decoy: Decoy, jpeg_encoder: Encoder, settings: Settings
    ) -> None:
        # Arrange
        image_data = create_image_data(
            decoy, jpeg_encoder, JPEGBaseline8Bit, bits=16, samples_per_pixel=1
        )
        resolver = EncodingResolver(settings, TranscodingMode.ALWAYS)

        # Act & Assert
        with pytest.raises(ValueError, match="must match image data bits"):
            resolver.resolve(image_data)


@pytest.mark.unittest
@pytest.mark.skipif(not JpegOptimizer.is_available(), reason="Requires imagecodecs")
class TestEncodingResolverOptimizingJpeg:
    def test_jpeg_baseline_tiles_are_recoded(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(optimize=True))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, RecodedEncoding)
        assert isinstance(encoding.recoder, JpegOptimizer)
        assert encoding.transfer_syntax == JPEGBaseline8Bit
        assert encoding.transcoder is None

    def test_transcoding_when_required_recodes_jpeg_baseline_tiles(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(optimize=True))
        resolver = EncodingResolver(transcoder, TranscodingMode.WHEN_REQUIRED)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, RecodedEncoding)
        assert encoding.transfer_syntax == JPEGBaseline8Bit

    def test_generated_tiles_of_recoded_source_are_optimized(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(optimize=True))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        settings = encoding.generated_tiles_encoder.settings
        assert isinstance(settings, JpegSettings)
        assert settings.optimize
        assert settings.quality == JpegSettings().quality

    def test_own_transcoder_of_image_data_is_optimized(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        # Image data in the transfer syntax of the transcoder, that can only give
        # decoded tiles, is transcoded with its own transcoder.
        image_data = create_image_data(
            decoy, jpeg_encoder, JPEGBaseline8Bit, transcoder=jpeg_encoder
        )
        transcoder = Encoder.create_for_settings(JpegSettings(optimize=True))
        resolver = EncodingResolver(transcoder, TranscodingMode.MATCH)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, TranscodedEncoding)
        settings = encoding.generated_tiles_encoder.settings
        assert isinstance(settings, JpegSettings)
        assert settings.optimize
