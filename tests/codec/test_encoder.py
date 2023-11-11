import pytest
from PIL import ImageChops, ImageStat
from PIL.Image import Image as PILImage
from pydicom import Dataset

from wsidicom.codec import (
    Channels,
    Decoder,
    Encoder,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsLosslessSettings,
    JpegSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)


@pytest.fixture
def dataset(image: PILImage, settings: Settings):
    dataset = Dataset()
    dataset.PhotometricInterpretation = settings.photometric_interpretation
    dataset.BitsStored = settings.bits
    dataset.BitsAllocated = settings.allocated_bits
    dataset.Rows = image.height
    dataset.Columns = image.width
    dataset.SamplesPerPixel = settings.samples_per_pixel
    dataset.PixelRepresentation = 0
    dataset.PlanarConfiguration = 0
    yield dataset


@pytest.fixture
def decoder(settings: Settings, dataset: Dataset):
    yield Decoder.create(
        settings.transfer_syntax,
        settings.samples_per_pixel,
        settings.bits,
        dataset,
    )


@pytest.mark.unittest
class TestEncoder:
    @pytest.mark.parametrize(
        ["settings", "allowed_rms"],
        [
            (JpegSettings(8, Channels.GRAYSCALE, quality=95), 2),
            (
                JpegSettings(8, Channels.YBR, subsampling=Subsampling.R444, quality=95),
                2,
            ),
            (
                JpegSettings(8, Channels.RGB, subsampling=Subsampling.R444, quality=95),
                2,
            ),
            (JpegSettings(12, Channels.GRAYSCALE, quality=95), 2),
            (JpegLosslessSettings(8, Channels.GRAYSCALE, predictor=7), 0),
            (JpegLosslessSettings(8, Channels.YBR, predictor=7), 0),
            (JpegLosslessSettings(8, Channels.RGB, predictor=7), 0),
            (JpegLosslessSettings(16, Channels.GRAYSCALE, predictor=7), 0),
            (JpegLosslessSettings(8, Channels.GRAYSCALE, predictor=None), 0),
            (JpegLosslessSettings(8, Channels.YBR, predictor=None), 0),
            (JpegLosslessSettings(8, Channels.RGB, predictor=None), 0),
            (JpegLosslessSettings(16, Channels.GRAYSCALE, predictor=None), 0),
            (JpegLsLosslessSettings(8), 0),
            (JpegLsLosslessSettings(16), 0),
            (JpegLsLosslessSettings(8, level=1), 1),
            (JpegLsLosslessSettings(16, level=1), 1),
            (Jpeg2kSettings(8, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(8, Channels.YBR), 1),
            (Jpeg2kSettings(8, Channels.RGB), 1),
            (Jpeg2kSettings(16, Channels.GRAYSCALE), 1),
            (Jpeg2kSettings(8, Channels.GRAYSCALE, level=None), 0),
            (Jpeg2kSettings(8, Channels.YBR, level=None), 0),
            (Jpeg2kSettings(8, Channels.RGB, level=None), 0),
            (Jpeg2kSettings(16, Channels.GRAYSCALE, level=None), 0),
            (RleSettings(8, Channels.GRAYSCALE), 0),
            (RleSettings(8, Channels.RGB), 0),
            (RleSettings(16, Channels.GRAYSCALE), 0),
            (NumpySettings(8, Channels.GRAYSCALE, True, True), 0),
            (NumpySettings(8, Channels.GRAYSCALE, True, False), 0),
            (NumpySettings(8, Channels.GRAYSCALE, False, True), 0),
            (NumpySettings(16, Channels.GRAYSCALE, True, True), 0),
            (NumpySettings(16, Channels.GRAYSCALE, True, False), 0),
            (NumpySettings(16, Channels.GRAYSCALE, False, True), 0),
            (NumpySettings(8, Channels.RGB, True, True), 0),
            (NumpySettings(8, Channels.RGB, True, False), 0),
            (NumpySettings(8, Channels.RGB, False, True), 0),
        ],
    )
    def test_encode(
        self, image: PILImage, decoder: Decoder, settings: Settings, allowed_rms: float
    ):
        # Arrange
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
        encoder = Encoder.create(settings)

        # Act
        encoded = encoder.encode(image)

        # Assert
        decoded = decoder.decode(encoded)
        if settings.channels == Channels.GRAYSCALE:
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms
