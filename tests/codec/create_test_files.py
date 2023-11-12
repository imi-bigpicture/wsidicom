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

from pathlib import Path
from typing import List

from PIL import Image
from tests.codec.conftest import get_filepath_for_encoder_settings, get_test_tile_path

from wsidicom.codec import (
    Channels,
    Encoder,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsLosslessSettings,
    JpegSettings,
    RleSettings,
    Subsampling,
)
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.codec.settings import NumpySettings

encoder_settings: List[EncoderSettings] = [
    JpegSettings(95, 8, Channels.GRAYSCALE),
    JpegSettings(95, 8, Channels.YBR, Subsampling.R444),
    JpegSettings(95, 8, Channels.RGB, Subsampling.R444),
    JpegSettings(95, 12, Channels.GRAYSCALE),
    JpegLosslessSettings(7, 8, Channels.GRAYSCALE),
    JpegLosslessSettings(7, 8, Channels.YBR),
    JpegLosslessSettings(7, 8, Channels.RGB),
    JpegLosslessSettings(7, 16, Channels.GRAYSCALE),
    JpegLosslessSettings(1, 8, Channels.GRAYSCALE),
    JpegLosslessSettings(1, 8, Channels.YBR),
    JpegLosslessSettings(1, 8, Channels.RGB),
    JpegLosslessSettings(1, 16, Channels.GRAYSCALE),
    JpegLsLosslessSettings(0, 8),
    JpegLsLosslessSettings(0, 16),
    JpegLsLosslessSettings(1, 8),
    JpegLsLosslessSettings(1, 16),
    Jpeg2kSettings(80, 8, Channels.GRAYSCALE),
    Jpeg2kSettings(80, 8, Channels.YBR),
    Jpeg2kSettings(80, 8, Channels.RGB),
    Jpeg2kSettings(80, 16, Channels.GRAYSCALE),
    Jpeg2kSettings(0, 8, Channels.GRAYSCALE),
    Jpeg2kSettings(0, 8, Channels.YBR),
    Jpeg2kSettings(0, 8, Channels.RGB),
    Jpeg2kSettings(0, 16, Channels.GRAYSCALE),
    RleSettings(8, Channels.GRAYSCALE),
    RleSettings(8, Channels.RGB),
    RleSettings(16, Channels.GRAYSCALE),
    NumpySettings(8, Channels.GRAYSCALE, True, True),
    NumpySettings(8, Channels.GRAYSCALE, True, False),
    NumpySettings(8, Channels.GRAYSCALE, False, True),
    NumpySettings(16, Channels.GRAYSCALE, True, True),
    NumpySettings(16, Channels.GRAYSCALE, True, False),
    NumpySettings(16, Channels.GRAYSCALE, False, True),
    NumpySettings(8, Channels.RGB, True, True),
    NumpySettings(8, Channels.RGB, True, False),
    NumpySettings(8, Channels.RGB, False, True),
]


def create_encoded_test_files():
    for settings in encoder_settings:
        print("Creating test file for settings: ", settings)
        test_tile_path = get_test_tile_path(Path("tests/testdata/"))
        image = Image.open(test_tile_path)
        if settings.channels == Channels.GRAYSCALE:
            image = image.convert("L")
        encoder = Encoder.create(settings)
        encoded = encoder.encode(image)
        output_path = get_filepath_for_encoder_settings(
            Path("tests/testdata/encoded/"), settings
        )

        with open(output_path, "wb") as file:
            file.write(encoded)


create_encoded_test_files()
