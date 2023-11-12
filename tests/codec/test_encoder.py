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
