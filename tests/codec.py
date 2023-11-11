from cgi import test
from pathlib import Path
from typing import List

from PIL import Image

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
from wsidicom.codec.settings import NumpySettings

encoder_settings: List[EncoderSettings] = [
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
]


def get_test_tile_path(test_data_path: Path):
    return test_data_path.joinpath("test_tile.png")


def get_filepath_for_encoder_settings(test_data_path: Path, settings: EncoderSettings):
    return test_data_path.joinpath(
        f"{settings.transfer_syntax.name}-{settings.channels.name}-{settings.bits}"
    ).with_suffix(settings.extension)


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


def get_encoded_test_file(settings: EncoderSettings):
    file_path = get_filepath_for_encoder_settings(
        Path("tests/testdata/encoded"), settings
    )
    with open(file_path, "rb") as file:
        return file.read()


create_encoded_test_files()
