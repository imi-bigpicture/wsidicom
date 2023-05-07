#    Copyright 2022 SECTRA AB
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

import math
from typing import Any, Dict, Tuple
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydicom import Dataset, dcmread

from wsidicom.file.wsidicom_file import WsiDicomFile
from wsidicom.file.wsidicom_file_base import OffsetTableType
from wsidicom.instance import ImageType, TileType, WsiDataset
from parameterized import parameterized

from tests.data_gen import (
    TESTFRAME,
    create_layer_file,
    create_main_dataset,
    create_meta_dataset,
)


@dataclass
class WsiDicomFileTestFile:
    path: Path
    tile_type: TileType
    bot_type: OffsetTableType
    ds: Dataset


FILE_SETTINGS = {
    "sparse_no_bot": {
        "name": "sparse_no_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.NONE,
    },
    "sparse_with_bot": {
        "name": "sparse_with_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.BASIC,
    },
    "full_no_bot_": {
        "name": "full_no_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.NONE,
    },
    "full_with_bot": {
        "name": "full_with_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.BASIC,
    },
}


@pytest.mark.unittest
class WsiDicomFileTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tempdir: TemporaryDirectory
        self.file: WsiDicomFile

    @classmethod
    def setUpClass(cls):
        cls.tempdir = TemporaryDirectory()
        cls.meta_dataset = create_meta_dataset()
        cls.padded_test_frame = TESTFRAME + b"\x00" * (len(TESTFRAME) % 2)
        cls._files: Dict[str, Tuple[WsiDicomFile, Dataset]] = {}

    @classmethod
    def tearDownClass(cls):
        [file.close() for (file, _) in cls._files.values()]
        cls.tempdir.cleanup()

    def get_file(self, name: str) -> Tuple[WsiDicomFile, Dataset]:
        if name in self._files:
            return self._files[name]
        file_setting = FILE_SETTINGS[name]
        dataset = create_main_dataset(
            file_setting["tile_type"], file_setting["bot_type"]
        )
        test_file = WsiDicomFileTestFile(
            Path(self.tempdir.name).joinpath(file_setting["name"]),
            file_setting["tile_type"],
            file_setting["bot_type"],
            dataset,
        )
        create_layer_file(test_file.path, test_file.ds, self.meta_dataset)
        file = WsiDicomFile.open(test_file.path)
        self._files[name] = file, dataset
        return file, dataset

    @parameterized.expand((FILE_SETTINGS.items))
    def test_open(self, name: str, settings: Dict[str, Any]):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        offset_table_type = test_file.offset_table_type
        tile_type = test_file.dataset.tile_type

        # Assert
        self.assertEqual(offset_table_type, settings["bot_type"])
        self.assertEqual(tile_type, settings["tile_type"])

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_dataset_property(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)
        path = test_file.filepath
        assert path is not None

        # Act
        ds = WsiDataset(dcmread(path, stop_before_pixels=True))

        # Assert
        self.assertEqual(test_file.dataset, ds)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_image_type_property(self, name: str):
        # Arrage
        test_file, _ = self.get_file(name)

        # Act
        image_type = test_file.image_type

        # Assert
        self.assertEqual(image_type, ImageType.VOLUME)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_uids_property(self, name: str):
        # Arrange
        test_file, dataset = self.get_file(name)

        # Act
        uids = test_file.uids
        # Assert
        self.assertEqual(uids.instance, dataset.SOPInstanceUID)
        self.assertEqual(
            uids.concatenation,
            getattr(dataset, "SOPInstanceUIDOfConcatenationSource", None),
        )
        self.assertEqual(uids.slide.frame_of_reference, dataset.FrameOfReferenceUID)
        self.assertEqual(uids.slide.study_instance, dataset.StudyInstanceUID)
        self.assertEqual(uids.slide.series_instance, dataset.SeriesInstanceUID)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_transfer_syntax_property(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        transfer_syntax = test_file.transfer_syntax

        # Assert
        self.assertEqual(transfer_syntax, self.meta_dataset.TransferSyntaxUID)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_frame_offset_property(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        frame_offset = test_file.frame_offset

        # Assert
        self.assertEqual(frame_offset, 0)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_frame_count_property(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        frame_count = test_file.frame_count

        # Assert
        self.assertEqual(frame_count, 1)

    @parameterized.expand((FILE_SETTINGS.items))
    def test_get_offset_table_type(self, name: str, settings: Dict[str, Any]):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        offset_type = test_file._get_offset_table_type()

        # Assert
        self.assertEqual(offset_type, settings["bot_type"])

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_validate_pixel_data_start(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)
        test_file._file.seek(test_file._pixel_data_position)

        # Act
        tag = test_file._file.read_tag()

        # Assert
        test_file._validate_pixel_data_start(tag)

    @parameterized.expand((FILE_SETTINGS.items))
    def test_read_bot_length(self, name: str, settings: Dict[str, Any]):
        # Arrange
        test_file, _ = self.get_file(name)
        test_file._file.seek(test_file._pixel_data_position)
        tag = test_file._file.read_tag()
        test_file._validate_pixel_data_start(tag)
        if settings["bot_type"] == OffsetTableType.BASIC:
            expected_bot_length = 4
        else:
            expected_bot_length = None

        # Act
        length = test_file._read_bot_length()

        # Assert
        self.assertEqual(length, expected_bot_length)

    @parameterized.expand((FILE_SETTINGS.items))
    def test_read_bot(self, name: str, settings: Dict[str, Any]):
        # Arrange
        test_file, _ = self.get_file(name)
        test_file._file.seek(test_file._pixel_data_position)
        tag = test_file._file.read_tag()
        test_file._validate_pixel_data_start(tag)
        if settings["bot_type"] == OffsetTableType.BASIC:
            first_bot_entry = b"\x00\x00\x00\x00"
        else:
            first_bot_entry = None

        # Act
        bot = test_file._read_bot()

        # Assert
        self.assertEqual(bot, first_bot_entry)

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_parse_bot_table(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        test_file._file.seek(test_file._pixel_data_position)
        tag = test_file._file.read_tag()
        test_file._validate_pixel_data_start(tag)
        bot = test_file._read_bot()
        first_frame_item_position = test_file._file.tell()

        # Act
        if bot is not None:
            positions = test_file._parse_table(
                bot, OffsetTableType.BASIC, first_frame_item_position
            )

            # Assert
            self.assertEqual(
                positions,
                [
                    (
                        (first_frame_item_position + TAG_BYTES + LENGTH_BYTES),
                        math.ceil(len(TESTFRAME) / 2) * 2,
                    )
                ],
            )

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_read_positions_from_pixeldata(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        test_file._file.seek(test_file._pixel_data_position)
        tag = test_file._file.read_tag()
        test_file._validate_pixel_data_start(tag)
        bot = test_file._read_bot()
        first_frame_item = test_file._file.tell()

        # Act
        positions = test_file._read_positions_from_pixeldata()

        # Assert
        self.assertEqual(
            positions,
            [
                (
                    (first_frame_item + TAG_BYTES + LENGTH_BYTES),
                    len(self.padded_test_frame),
                )
            ],
        )

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_read_sequence_delimiter(self, name: str):
        test_file, _ = self.get_file(name)
        (last_item_position, last_item_length) = test_file.frame_positions[-1]
        last_item_end = last_item_position + last_item_length
        test_file._file.seek(last_item_end)
        test_file._file.read_tag()
        test_file._read_sequence_delimiter()

    @parameterized.expand((FILE_SETTINGS.keys))
    def test_read_frame(self, name: str):
        # Arrange
        test_file, _ = self.get_file(name)

        # Act
        frame = test_file.read_frame(0)

        # Assert
        self.assertEqual(frame, self.padded_test_frame)
