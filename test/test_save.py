#    Copyright 2021 SECTRA AB
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

import os
from random import randint
from struct import unpack
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple
from pydicom.encaps import itemize_frame
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.tag import ItemTag, SequenceDelimiterTag, Tag

import pytest
from PIL import Image
from pydicom.filebase import DicomFile
from pydicom.uid import UID, JPEGBaseline8Bit, generate_uid
from wsidicom import WsiDicom
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.instance import (ImageData, WsiDicomFile,
                               WsiDicomFileWriter)
from wsidicom.uid import WSI_SOP_CLASS_UID
from os import urandom

wsidicom_test_data_dir = os.environ.get("WSIDICOM_TESTDIR", "C:/temp/wsidicom")
sub_data_dir = "interface"
data_dir = wsidicom_test_data_dir + '/' + sub_data_dir


class WsiDicomTestFile(WsiDicomFile):
    """Test version of WsiDicomFile that overrides __init__."""
    def __init__(self, filepath: Path, transfer_syntax: UID, frame_count: int):
        self._filepath = filepath
        self._fp = DicomFile(filepath, mode='rb')
        self._fp.is_little_endian = transfer_syntax.is_little_endian
        self._fp.is_implicit_VR = transfer_syntax.is_implicit_VR
        self._frame_count = frame_count
        self.__enter__()


class WsiDicomTestImageData(ImageData):
    def __init__(self, data: List[bytes]) -> None:
        self._data = data

    @property
    def files(self) -> List[Path]:
        return []

    @property
    def transfer_syntax(self) -> UID:
        return JPEGBaseline8Bit

    @property
    def image_size(self) -> Size:
        return Size(100, 100)

    @property
    def tile_size(self) -> Size:
        return Size(10, 10)

    @property
    def pixel_spacing(self) -> SizeMm:
        return SizeMm(1.0, 1.0)

    @property
    def samples_per_pixel(self) -> int:
        return 3

    @property
    def photometric_interpretation(self) -> str:
        return 'YBR'

    def _get_decoded_tile(
        self,
        tile_point:
        Point,
        z: float,
        path: str
    ) -> Image.Image:
        raise NotImplementedError

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self._data[tile.x + tile.y * self.tiled_size.width]

    def close(self) -> None:
        return super().close()


@pytest.mark.save
class WsiDicomFileSaveTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_folders: Dict[Path, WsiDicom]

    @classmethod
    def setUpClass(cls):
        cls.frame_count = 100
        MIN_FRAME_LENGTH = 2
        MAX_FRAME_LENGTH = 100
        # Generate test data by itemizing random bytes of random length
        # from MIN_FRAME_LENGTH to MAX_FRAME_LENGTH.
        cls.test_data = [
            next(itemize_frame(urandom(randint(
                MIN_FRAME_LENGTH,
                MAX_FRAME_LENGTH
            ))))
            for i in range(cls.frame_count)
        ]

    def test_write_preamble(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._write_preamble()
            self.assertTrue(is_dicom(filepath))

    def test_write_meta(self):
        transfer_syntax = JPEGBaseline8Bit
        instance_uid = generate_uid()
        class_uid = UID(WSI_SOP_CLASS_UID)
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._write_preamble()
                write_file._write_file_meta(instance_uid, transfer_syntax)
            file_meta = read_file_meta_info(filepath)
        self.assertEqual(file_meta.TransferSyntaxUID, transfer_syntax)
        self.assertEqual(file_meta.MediaStorageSOPInstanceUID, instance_uid)
        self.assertEqual(file_meta.MediaStorageSOPClassUID, class_uid)

    def write_table(
        self,
        file_path: Path,
        offset_table: Optional[str]
    ) -> List[Tuple[int, int]]:
        with WsiDicomFileWriter(file_path) as write_file:
            table_start, pixel_data_start = write_file._write_pixel_data_start(
                number_of_frames=self.frame_count,
                offset_table=offset_table
            )
            positions = []
            for frame in self.test_data:
                positions.append(write_file._fp.tell())
                write_file._fp.write(frame)
            pixel_data_end = write_file._fp.tell()
            write_file._write_pixel_data_end()
            if offset_table is not None:
                if table_start is None:
                    raise ValueError('Table start should not be None')
                if offset_table == 'eot':
                    write_file._write_eot(
                        table_start,
                        pixel_data_start,
                        positions,
                        pixel_data_end
                    )
                elif offset_table == 'bot':
                    write_file._write_bot(
                        table_start,
                        pixel_data_start,
                        positions
                    )

        TAG_BYTES = 4
        LENGHT_BYTES = 4
        frame_offsets = []
        for position in positions:  # Positions are from frame data start
            frame_offsets.append(position + TAG_BYTES + LENGHT_BYTES)
        frame_lengths = [  # Lengths are without tag and length parts
            len(frame) - TAG_BYTES - LENGHT_BYTES for frame in self.test_data
        ]
        expected_frame_index = [
            (offset, length)
            for offset, length in zip(frame_offsets, frame_lengths)
        ]
        return expected_frame_index

    def read_table(
        self,
        file_path: Path
    ) -> List[Tuple[int, int]]:
        with WsiDicomTestFile(
            file_path,
            JPEGBaseline8Bit,
            self.frame_count
        ) as read_file:
            frame_index = read_file._parse_pixel_data()
            return frame_index

    def assert_end_of_file(self, file: WsiDicomTestFile):
        with self.assertRaises(EOFError):
            file._fp.read(1, need_exact_length=True)

    def test_write_and_read_bot(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            expected_frame_index = self.write_table(filepath, 'bot')
            frame_index = self.read_table(filepath)
            self.assertEqual(expected_frame_index, frame_index)

    def test_write_and_read_eot(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            expected_frame_index = self.write_table(filepath, 'eot')
            frame_index = self.read_table(filepath)
            self.assertEqual(expected_frame_index, frame_index)

    def test_write_and_read_no_table(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            expected_frame_index = self.write_table(filepath, None)
            frame_index = self.read_table(filepath)
            self.assertEqual(expected_frame_index, frame_index)

    def test_reserve_bot(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._reserve_bot(self.frame_count)
            with WsiDicomTestFile(
                filepath,
                JPEGBaseline8Bit,
                self.frame_count
            ) as read_file:
                tag = read_file._fp.read_tag()
                self.assertEqual(tag, ItemTag)
                length = read_file._read_tag_length(False)
                BOT_ITEM_LENGTH = 4
                self.assertEqual(length, BOT_ITEM_LENGTH*self.frame_count)
                for frame in range(self.frame_count):
                    self.assertEqual(read_file._fp.read_UL(), 0)
                self.assert_end_of_file(read_file)

    def test_reserve_eot(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._reserve_eot(self.frame_count)
            with WsiDicomTestFile(
                filepath,
                JPEGBaseline8Bit,
                self.frame_count
            ) as read_file:
                tag = read_file._fp.read_tag()
                self.assertEqual(tag, Tag('ExtendedOffsetTable'))
                length = read_file._read_tag_length(True)
                EOT_ITEM_LENGTH = 8
                self.assertEqual(length, EOT_ITEM_LENGTH*self.frame_count)
                for frame in range(self.frame_count):
                    self.assertEqual(
                        unpack('<Q', read_file._fp.read(EOT_ITEM_LENGTH))[0],
                        0
                    )

                tag = read_file._fp.read_tag()
                self.assertEqual(tag, Tag('ExtendedOffsetTableLengths'))
                length = read_file._read_tag_length(True)
                EOT_ITEM_LENGTH = 8
                self.assertEqual(length, EOT_ITEM_LENGTH*self.frame_count)
                for frame in range(self.frame_count):
                    self.assertEqual(
                        unpack('<Q', read_file._fp.read(EOT_ITEM_LENGTH))[0],
                        0
                    )
                self.assert_end_of_file(read_file)

    def test_write_pixel_end(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._write_pixel_data_end()
            with WsiDicomTestFile(
                filepath,
                JPEGBaseline8Bit,
                self.frame_count
            ) as read_file:
                tag = read_file._fp.read_tag()
                self.assertEqual(tag, SequenceDelimiterTag)
                length = read_file._read_tag_length(False)
                self.assertEqual(length, 0)

    def test_write_pixel_data(self):
        image_data = WsiDicomTestImageData(self.test_data)
        print(image_data.tiled_size)
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._write_pixel_data(
                    image_data=image_data,
                    z=image_data.default_z,
                    path=image_data.default_path,
                    workers=1,
                    chunk_size=10
                )



