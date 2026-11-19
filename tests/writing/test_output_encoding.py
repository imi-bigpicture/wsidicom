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
    Encoder,
    Jpeg2kSettings,
    JpegSettings,
)
from wsidicom.codec.recoder import JpegOptimizer
from wsidicom.instance import ImageData
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
) -> ImageData:
    """Return mocked 8-bit, 3-sample image data with the encoding given."""
    image_data = decoy.mock(cls=ImageData)
    decoy.when(image_data.encoder).then_return(encoder)
    decoy.when(image_data.transfer_syntax).then_return(transfer_syntax)
    decoy.when(image_data.transcoder).then_return(transcoder)
    decoy.when(image_data.bits).then_return(8)
    decoy.when(image_data.samples_per_pixel).then_return(3)
    return image_data


@pytest.mark.unittest
class TestEncodingResolver:
    def test_without_transcoder_tiles_are_copied(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        resolver = EncodingResolver(None, False)

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
        resolver = EncodingResolver(transcoder, False)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert encoding == CopiedEncoding(jpeg_encoder, JPEGBaseline8Bit)

    def test_transcoder_in_other_transfer_syntax_transcodes_tiles(
        self, decoy: Decoy, jpeg_encoder: Encoder, jpeg_2000_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_2000_encoder, JPEG2000Lossless)
        resolver = EncodingResolver(jpeg_encoder, False)

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
        resolver = EncodingResolver(settings, False)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, TranscodedEncoding)
        assert encoding.generated_tiles_encoder.settings == settings
        assert encoding.transfer_syntax == JPEGBaseline8Bit

    def test_forced_transcoding_transcodes_tiles_in_same_transfer_syntax(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(quality=90))
        resolver = EncodingResolver(transcoder, True)

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
        resolver = EncodingResolver(None, False)

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
        resolver = EncodingResolver(transcoder, False)

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
        resolver = EncodingResolver(transcoder, False)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, RecodedEncoding)
        assert isinstance(encoding.recoder, JpegOptimizer)
        assert encoding.transfer_syntax == JPEGBaseline8Bit
        assert encoding.transcoder is None

    def test_generated_tiles_of_recoded_source_are_optimized(
        self, decoy: Decoy, jpeg_encoder: Encoder
    ) -> None:
        # Arrange
        image_data = create_image_data(decoy, jpeg_encoder, JPEGBaseline8Bit)
        transcoder = Encoder.create_for_settings(JpegSettings(optimize=True))
        resolver = EncodingResolver(transcoder, False)

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
        resolver = EncodingResolver(transcoder, False)

        # Act
        encoding = resolver.resolve(image_data)

        # Assert
        assert isinstance(encoding, TranscodedEncoding)
        settings = encoding.generated_tiles_encoder.settings
        assert isinstance(settings, JpegSettings)
        assert settings.optimize
