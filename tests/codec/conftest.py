from pathlib import Path
import pytest
from PIL import Image

from wsidicom.codec import Settings as EncoderSettings


def get_test_tile_path(test_data_path: Path):
    return test_data_path.joinpath("test_tile.png")


def get_filepath_for_encoder_settings(test_data_path: Path, settings: EncoderSettings):
    return test_data_path.joinpath(
        f"{settings.transfer_syntax.name}-{settings.channels.name}-{settings.bits}"
    ).with_suffix(settings.extension)


@pytest.fixture
def image():
    path = get_test_tile_path(Path("tests/testdata"))
    yield Image.open(path)


@pytest.fixture
def encoded(encoder_settings: EncoderSettings):
    file_path = get_filepath_for_encoder_settings(
        Path("tests/testdata/encoded"), encoder_settings
    )
    with open(file_path, "rb") as file:
        return file.read()
