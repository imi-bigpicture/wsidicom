#    Copyright 2022, 2023 SECTRA AB
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
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset

from tests.data_gen import (
    TESTFRAME,
    create_layer_file,
    create_main_dataset,
    create_meta_dataset,
)
from wsidicom.file.io import OffsetTableType, WsiDicomReader
from wsidicom.instance import ImageType, TileType, WsiDataset

FILE_SETTINGS = {
    "sparse_no_bot": {
        "name": "sparse_no_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.EMPTY,
    },
    "sparse_with_bot": {
        "name": "sparse_with_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.BASIC,
    },
    "full_no_bot_": {
        "name": "full_no_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.EMPTY,
    },
    "full_with_bot": {
        "name": "full_with_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.BASIC,
    },
}


@pytest.fixture()
def meta_dataset():
    yield create_meta_dataset()


@pytest.fixture()
def padded_test_frame():
    yield TESTFRAME + b"\x00" * (len(TESTFRAME) % 2)


@pytest.fixture()
def dataset(name: str):
    file_setting = FILE_SETTINGS[name]
    dataset = create_main_dataset(file_setting["tile_type"], file_setting["bot_type"])
    yield dataset


@pytest.fixture()
def test_file(name: str, dataset: Dataset, meta_dataset: FileMetaDataset):
    file_setting = FILE_SETTINGS[name]
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath(file_setting["name"])
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader.open(path)
        yield reader
        reader.close()


@pytest.mark.unittest
class TestWWsiDicomReader:
    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_offset_table_type_property(
        self, test_file: WsiDicomReader, settings: Dict[str, Any]
    ):
        # Arrange

        # Act
        offset_table_type = test_file.offset_table_type

        # Assert
        assert offset_table_type == settings["bot_type"]

    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_tile_type_property(
        self, test_file: WsiDicomReader, settings: Dict[str, Any]
    ):
        # Arrange

        # Act
        tile_type = test_file.dataset.tile_type

        # Assert
        assert tile_type == settings["tile_type"]

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_dataset_property(self, test_file: WsiDicomReader):
        # Arrange
        path = test_file.filepath
        assert path is not None

        # Act
        dataset = WsiDataset(dcmread(path, stop_before_pixels=True))

        # Assert
        assert test_file.dataset == dataset

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_image_type_property(
        self,
        test_file: WsiDicomReader,
    ):
        # Arrange

        # Act
        image_type = test_file.image_type

        # Assert
        assert image_type == ImageType.VOLUME

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_uids_property(self, test_file: WsiDicomReader, dataset: Dataset):
        # Arrange

        # Act
        uids = test_file.uids

        # Assert
        assert uids.instance == dataset.SOPInstanceUID
        assert uids.concatenation == getattr(
            dataset, "SOPInstanceUIDOfConcatenationSource", None
        )
        assert uids.slide.frame_of_reference == dataset.FrameOfReferenceUID
        assert uids.slide.study_instance == dataset.StudyInstanceUID
        assert uids.slide.series_instance == dataset.SeriesInstanceUID

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_transfer_syntax_property(
        self, test_file: WsiDicomReader, meta_dataset: FileMetaDataset
    ):
        # Arrange

        # Act
        transfer_syntax = test_file.transfer_syntax

        # Assert
        assert transfer_syntax == meta_dataset.TransferSyntaxUID

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_frame_offset_property(self, test_file: WsiDicomReader):
        # Arrange

        # Act
        frame_offset = test_file.frame_offset

        # Assert
        assert frame_offset == 0

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_frame_count_property(self, test_file: WsiDicomReader):
        # Arrange

        # Act
        frame_count = test_file.frame_count

        # Assert
        assert frame_count == 1

    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_get_offset_table_type(
        self, test_file: WsiDicomReader, settings: Dict[str, Any]
    ):
        # Arrange

        # Act
        offset_type = test_file.offset_table_type

        # Assert
        assert offset_type == settings["bot_type"]

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_read_frame(self, test_file: WsiDicomReader, padded_test_frame: bytes):
        # Arrange

        # Act
        frame = test_file.read_frame(0)

        # Assert
        assert frame == padded_test_frame
