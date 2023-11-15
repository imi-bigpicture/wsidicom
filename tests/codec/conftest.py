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
    if not path.exists():
        pytest.skip(f"Test file {path} does not exist")
    yield Image.open(path)


@pytest.fixture
def encoded(encoder_settings: EncoderSettings):
    file_path = get_filepath_for_encoder_settings(
        Path("tests/testdata/encoded"), encoder_settings
    )
    if not file_path.exists():
        pytest.skip(f"Test file {file_path} does not exist")
    with open(file_path, "rb") as file:
        return file.read()
