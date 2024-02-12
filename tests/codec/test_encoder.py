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

from typing import Type

import pytest
from PIL import ImageChops, ImageStat
from PIL.Image import Image
from pydicom import Dataset

from wsidicom.codec import (
    Channels,
    Decoder,
    Encoder,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsSettings,
    JpegSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)
from wsidicom.codec.encoder import (
    ImageCodecsRleEncoder,
    Jpeg2kEncoder,
    JpegEncoder,
    JpegLsEncoder,
    NumpyEncoder,
    PillowEncoder,
    PyJpegLsEncoder,
    PyLibJpegOpenJpegEncoder,
    PylibjpegRleEncoder,
)
from wsidicom.instance.dataset import WsiDataset


@pytest.fixture
def dataset(image: Image, settings: Settings):
    dataset = Dataset()
    dataset.PhotometricInterpretation = settings.photometric_interpretation
    dataset.BitsStored = settings.bits
    dataset.BitsAllocated = settings.allocated_bits
    dataset.Rows = image.height
    dataset.Columns = image.width
    dataset.SamplesPerPixel = settings.samples_per_pixel
    dataset.PixelRepresentation = 0
    dataset.PlanarConfiguration = 0
    yield WsiDataset(dataset)


@pytest.fixture
def decoder(settings: Settings, dataset: WsiDataset):
    if (
        Decoder.select_decoder(
            settings.transfer_syntax, dataset.samples_per_pixel, dataset.bits
        )
        is None
    ):
        pytest.skip("Decoder not available")
    yield Decoder.create(
        settings.transfer_syntax,
        dataset.samples_per_pixel,
        dataset.bits,
        dataset.tile_size,
        dataset.photometric_interpretation,
    )


@pytest.mark.unittest
class TestEncoder:
    @pytest.mark.parametrize(
        ["encoder_type", "settings", "allowed_rms"],
        [
            (JpegEncoder, JpegSettings(95, 8, Channels.GRAYSCALE), 2),
            (JpegEncoder, JpegSettings(95, 8, Channels.YBR, Subsampling.R444), 2),
            (JpegEncoder, JpegSettings(95, 8, Channels.RGB, Subsampling.R444), 2),
            (JpegEncoder, JpegSettings(95, 12, Channels.GRAYSCALE), 2),
            (PillowEncoder, JpegSettings(95, 8, Channels.GRAYSCALE), 2),
            (PillowEncoder, JpegSettings(95, 8, Channels.YBR, Subsampling.R444), 2),
            (JpegEncoder, JpegLosslessSettings(7, 8, Channels.GRAYSCALE), 0),
            (JpegEncoder, JpegLosslessSettings(7, 8, Channels.YBR), 0),
            (JpegEncoder, JpegLosslessSettings(7, 8, Channels.RGB), 0),
            (JpegEncoder, JpegLosslessSettings(7, 16, Channels.GRAYSCALE), 0),
            (JpegEncoder, JpegLosslessSettings(1, 8, Channels.GRAYSCALE), 0),
            (JpegEncoder, JpegLosslessSettings(1, 8, Channels.YBR), 0),
            (JpegEncoder, JpegLosslessSettings(1, 8, Channels.RGB), 0),
            (JpegEncoder, JpegLosslessSettings(1, 16, Channels.GRAYSCALE), 0),
            (JpegLsEncoder, JpegLsSettings(0, 8, Channels.GRAYSCALE), 0),
            (JpegLsEncoder, JpegLsSettings(0, 8, Channels.RGB), 0),
            (JpegLsEncoder, JpegLsSettings(0, 16, Channels.GRAYSCALE), 0),
            (JpegLsEncoder, JpegLsSettings(1, 8, Channels.GRAYSCALE), 1),
            (JpegLsEncoder, JpegLsSettings(1, 8, Channels.RGB), 1),
            (JpegLsEncoder, JpegLsSettings(1, 16, Channels.GRAYSCALE), 1),
            (PyJpegLsEncoder, JpegLsSettings(0, 8, Channels.GRAYSCALE), 0),
            (PyJpegLsEncoder, JpegLsSettings(0, 8, Channels.RGB), 0),
            (PyJpegLsEncoder, JpegLsSettings(0, 16, Channels.GRAYSCALE), 0),
            (PyJpegLsEncoder, JpegLsSettings(1, 8, Channels.GRAYSCALE), 1),
            (PyJpegLsEncoder, JpegLsSettings(1, 8, Channels.RGB), 1),
            (PyJpegLsEncoder, JpegLsSettings(1, 16, Channels.GRAYSCALE), 1),
            (Jpeg2kEncoder, Jpeg2kSettings(80, 8, Channels.GRAYSCALE), 1),
            (Jpeg2kEncoder, Jpeg2kSettings(80, 8, Channels.YBR), 1),
            (Jpeg2kEncoder, Jpeg2kSettings(80, 8, Channels.RGB), 1),
            (Jpeg2kEncoder, Jpeg2kSettings(80, 16, Channels.GRAYSCALE), 1),
            (Jpeg2kEncoder, Jpeg2kSettings(0, 8, Channels.GRAYSCALE), 0),
            (Jpeg2kEncoder, Jpeg2kSettings(0, 8, Channels.YBR), 0),
            (Jpeg2kEncoder, Jpeg2kSettings(0, 8, Channels.RGB), 0),
            (Jpeg2kEncoder, Jpeg2kSettings(0, 16, Channels.GRAYSCALE), 0),
            (PillowEncoder, Jpeg2kSettings(80, 8, Channels.GRAYSCALE), 1),
            (PillowEncoder, Jpeg2kSettings(80, 8, Channels.YBR), 1),
            (PillowEncoder, Jpeg2kSettings(80, 8, Channels.RGB), 1),
            (PillowEncoder, Jpeg2kSettings(0, 8, Channels.GRAYSCALE), 0),
            (PillowEncoder, Jpeg2kSettings(0, 8, Channels.YBR), 0),
            (PillowEncoder, Jpeg2kSettings(0, 8, Channels.RGB), 0),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(80, 8, Channels.GRAYSCALE), 1),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(80, 8, Channels.YBR), 1),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(80, 8, Channels.RGB), 1),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(80, 16, Channels.GRAYSCALE), 1),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(0, 8, Channels.GRAYSCALE), 0),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(0, 8, Channels.YBR), 0),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(0, 8, Channels.RGB), 0),
            (PyLibJpegOpenJpegEncoder, Jpeg2kSettings(0, 16, Channels.GRAYSCALE), 0),
            (PylibjpegRleEncoder, RleSettings(8, Channels.GRAYSCALE), 0),
            (PylibjpegRleEncoder, RleSettings(8, Channels.RGB), 0),
            (PylibjpegRleEncoder, RleSettings(16, Channels.GRAYSCALE), 0),
            (ImageCodecsRleEncoder, RleSettings(8, Channels.GRAYSCALE), 0),
            (ImageCodecsRleEncoder, RleSettings(8, Channels.RGB), 0),
            (ImageCodecsRleEncoder, RleSettings(16, Channels.GRAYSCALE), 0),
            (NumpyEncoder, NumpySettings(8, Channels.GRAYSCALE, True, True), 0),
            (NumpyEncoder, NumpySettings(8, Channels.GRAYSCALE, True, False), 0),
            (NumpyEncoder, NumpySettings(8, Channels.GRAYSCALE, False, True), 0),
            (NumpyEncoder, NumpySettings(16, Channels.GRAYSCALE, True, True), 0),
            (NumpyEncoder, NumpySettings(16, Channels.GRAYSCALE, True, False), 0),
            (NumpyEncoder, NumpySettings(16, Channels.GRAYSCALE, False, True), 0),
            (NumpyEncoder, NumpySettings(8, Channels.RGB, True, True), 0),
            (NumpyEncoder, NumpySettings(8, Channels.RGB, True, False), 0),
            (NumpyEncoder, NumpySettings(8, Channels.RGB, False, True), 0),
        ],
    )
    def test_encode(
        self,
        image: Image,
        decoder: Decoder,
        encoder_type: Type[Encoder],
        settings: Settings,
        allowed_rms: float,
    ):
        # Arrange
        if not encoder_type.is_available():
            pytest.skip("Encoder not available")
        encoder = encoder_type(settings)

        # Act
        encoded = encoder.encode(image)

        # Assert
        decoded = decoder.decode(encoded)
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
            decoded = decoded.convert("L")
        diff = ImageChops.difference(decoded, image)
        for band_rms in ImageStat.Stat(diff).rms:
            assert band_rms <= allowed_rms
