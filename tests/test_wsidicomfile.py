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
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from pydicom import Dataset, dcmread
from wsidicom.dataset import ImageType, TileType
from wsidicom.file.file import OffsetTableType
from wsidicom.instance import WsiDataset, WsiDicomFile

from .data_gen import (
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


@pytest.mark.unittest
class WsiDicomFileTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tempdir: TemporaryDirectory
        self.file: WsiDicomFile

    @classmethod
    def setUpClass(cls):
        cls.tempdir = TemporaryDirectory()
        dirpath = Path(cls.tempdir.name)
        cls.meta_dataset = create_meta_dataset()
        file_settings = [
            {
                "name": "sparse_no_bot.dcm",
                "tile_type": TileType.SPARSE,
                "bot_type": OffsetTableType.NONE,
            },
            {
                "name": "sparse_with_bot.dcm",
                "tile_type": TileType.SPARSE,
                "bot_type": OffsetTableType.BASIC,
            },
            {
                "name": "full_no_bot_path.dcm",
                "tile_type": TileType.FULL,
                "bot_type": OffsetTableType.BASIC,
            },
            {
                "name": "full_with_bot_path.dcm",
                "tile_type": TileType.FULL,
                "bot_type": OffsetTableType.BASIC,
            },
        ]

        cls.test_files = [
            WsiDicomFileTestFile(
                dirpath.joinpath(file_setting["name"]),
                file_setting["tile_type"],
                file_setting["bot_type"],
                create_main_dataset(
                    file_setting["tile_type"], file_setting["bot_type"]
                ),
            )
            for file_setting in file_settings
        ]
        for test_file in cls.test_files:
            create_layer_file(test_file.path, test_file.ds, cls.meta_dataset)

        cls.opened_files = {
            WsiDicomFile(test_file.path): test_file for test_file in cls.test_files
        }
        cls.padded_test_frame = TESTFRAME + b"\x00" * (len(TESTFRAME) % 2)

    @classmethod
    def tearDownClass(cls):
        [file.close() for file in cls.opened_files]
        cls.tempdir.cleanup()

    def test_open_with_parse(self):
        for test_file in self.test_files:
            print(test_file.path, test_file.tile_type, test_file.bot_type)
            with WsiDicomFile(test_file.path, True) as file:
                self.assertIsNotNone(file._frame_positions)
                self.assertEqual(file.offset_table_type, test_file.bot_type)

                self.assertEqual(file.dataset.tile_type, test_file.tile_type)

    def test_open_without_parse(self):
        for test_file in self.test_files:
            with WsiDicomFile(test_file.path, False) as file:
                self.assertIsNone(file._offset_table_type)
                self.assertIsNone(file._frame_positions)
                self.assertEqual(file.offset_table_type, test_file.bot_type)

                self.assertEqual(file.dataset.tile_type, test_file.tile_type)

    def test_dataset_property(self):
        for test_file in self.opened_files:
            path = test_file.filepath
            ds = WsiDataset(dcmread(path, stop_before_pixels=True))
            self.assertEqual(test_file.dataset, ds)

    def test_image_type_property(self):
        for test_file in self.opened_files:
            self.assertEqual(test_file.image_type, ImageType.VOLUME)

    def test_uids_property(self):
        for test_file, settings in self.opened_files.items():
            self.assertEqual(test_file.uids.instance, settings.ds.SOPInstanceUID)
            self.assertEqual(
                test_file.uids.concatenation,
                getattr(settings.ds, "SOPInstanceUIDOfConcatenationSource", None),
            )
            self.assertEqual(
                test_file.uids.slide.frame_of_reference, settings.ds.FrameOfReferenceUID
            )
            self.assertEqual(
                test_file.uids.slide.study_instance, settings.ds.StudyInstanceUID
            )
            self.assertEqual(
                test_file.uids.slide.series_instance, settings.ds.SeriesInstanceUID
            )

    def test_transfer_syntax_property(self):
        for test_file in self.opened_files:
            self.assertEqual(
                test_file.transfer_syntax, self.meta_dataset.TransferSyntaxUID
            )

    def test_frame_offset_property(self):
        for test_file in self.opened_files:
            self.assertEqual(test_file.frame_offset, 0)

    def test_frame_count_property(self):
        for test_file in self.opened_files:
            self.assertEqual(test_file.frame_count, 1)

    def test_get_offset_table_type(self):
        for test_file, setting in self.opened_files.items():
            self.assertEqual(test_file._get_offset_table_type(), setting.bot_type)

    def test_validate_pixel_data_start(self):
        for test_file in self.opened_files:
            test_file._fp.seek(test_file._pixel_data_position)
            tag = test_file._fp.read_tag()
            test_file._validate_pixel_data_start(tag)

    def test_read_bot_length(self):
        for test_file, setting in self.opened_files.items():
            test_file._fp.seek(test_file._pixel_data_position)
            tag = test_file._fp.read_tag()
            test_file._validate_pixel_data_start(tag)
            length = test_file._read_bot_length()
            self.assertEqual(
                length, (4 if setting.bot_type == OffsetTableType.BASIC else None)
            )

    def test_read_bot(self):
        for test_file, setting in self.opened_files.items():
            test_file._fp.seek(test_file._pixel_data_position)
            tag = test_file._fp.read_tag()
            test_file._validate_pixel_data_start(tag)
            bot = test_file._read_bot()
            first_bot_entry = b"\x00\x00\x00\x00"
            self.assertEqual(
                bot,
                (
                    first_bot_entry
                    if setting.bot_type == OffsetTableType.BASIC
                    else None
                ),
            )

    def test_parse_bot_table(self):
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        for test_file, setting in self.opened_files.items():
            test_file._fp.seek(test_file._pixel_data_position)
            tag = test_file._fp.read_tag()
            test_file._validate_pixel_data_start(tag)
            bot = test_file._read_bot()
            first_frame_item = test_file._fp.tell()
            if bot is None:
                continue
            positions = test_file._parse_table(
                bot, OffsetTableType.BASIC, first_frame_item
            )
            self.assertEqual(
                positions,
                [
                    (
                        (first_frame_item + TAG_BYTES + LENGTH_BYTES),
                        math.ceil(len(TESTFRAME) / 2) * 2,
                    )
                ],
            )

    def test_read_positions_from_pixeldata(self):
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        for test_file, setting in self.opened_files.items():
            test_file._fp.seek(test_file._pixel_data_position)
            tag = test_file._fp.read_tag()
            test_file._validate_pixel_data_start(tag)
            bot = test_file._read_bot()
            first_frame_item = test_file._fp.tell()
            positions = test_file._read_positions_from_pixeldata()
            self.assertEqual(
                positions,
                [
                    (
                        (first_frame_item + TAG_BYTES + LENGTH_BYTES),
                        len(self.padded_test_frame),
                    )
                ],
            )

    def test_read_sequence_delimiter(self):
        for test_file in self.opened_files:
            (last_item_position, last_item_length) = test_file.frame_positions[-1]
            last_item_end = last_item_position + last_item_length
            test_file._fp.seek(last_item_end)
            test_file._fp.read_tag()
            test_file._read_sequence_delimiter()

    def test_read_frame(self):
        for test_file in self.opened_files:
            frame = test_file.read_frame(0)
            self.assertEqual(frame, self.padded_test_frame)
