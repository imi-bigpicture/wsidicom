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

import math
import os
import random
import sys
import unittest
from pathlib import Path
from struct import unpack
from tempfile import TemporaryDirectory
from typing import List, Optional, OrderedDict, Sequence, Tuple, cast

import pytest
from PIL import ImageChops, ImageFilter, ImageStat
from PIL.Image import Image as PILImage
from pydicom import Sequence as DicomSequence
from pydicom.dataset import Dataset
from pydicom.filebase import DicomFile
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.tag import ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID, JPEGBaseline8Bit, generate_uid
from wsidicom import WsiDicom
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.image_data import ImageData, ImageOrigin
from wsidicom.file import WsiDicomFile, WsiDicomFileWriter
from wsidicom.uid import WSI_SOP_CLASS_UID
from wsidicom.wsidicom import WsiDicomLevel

SLIDE_FOLDER = Path(os.environ.get(
    "WSIDICOM_TESTDIR",
    "tests/testdata/slides")
)


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
    def __init__(self, data: Sequence[bytes], tiled_size: Size) -> None:
        if len(data) != tiled_size.area:
            raise ValueError('Number of frames and tiled size area differ')
        TILE_SIZE = Size(10, 10)
        self._data = data
        self._tile_size = TILE_SIZE
        self._image_size = tiled_size * TILE_SIZE

    @property
    def files(self) -> List[Path]:
        return []

    @property
    def transfer_syntax(self) -> UID:
        return JPEGBaseline8Bit

    @property
    def image_size(self) -> Size:
        return self._image_size

    @property
    def tile_size(self) -> Size:
        return self._tile_size

    @property
    def pixel_spacing(self) -> SizeMm:
        return SizeMm(1.0, 1.0)

    @property
    def samples_per_pixel(self) -> int:
        return 3

    @property
    def photometric_interpretation(self) -> str:
        return 'YBR'

    @property
    def image_origin(self) -> ImageOrigin:
        return ImageOrigin()

    def _get_decoded_tile(
        self,
        tile_point:
        Point,
        z: float,
        path: str
    ) -> PILImage:
        raise NotImplementedError()

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self._data[tile.x + tile.y * self.tiled_size.width]

    def close(self) -> None:
        return super().close()


@pytest.mark.save
class WsiDicomFileSaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiled_size = Size(10, 10)
        cls.frame_count = cls.tiled_size.area
        SEED = 0
        MIN_FRAME_LENGTH = 2
        MAX_FRAME_LENGTH = 100
        # Generate test data by itemizing random bytes of random length
        # from MIN_FRAME_LENGTH to MAX_FRAME_LENGTH.
        rng = random.Random(SEED)
        lengths = [
            rng.randint(
                MIN_FRAME_LENGTH,
                MAX_FRAME_LENGTH
            )
            for i in range(cls.frame_count)
        ]
        cls.test_data = [
            rng.getrandbits(length*8).to_bytes(length, sys.byteorder)
            for length in lengths
        ]
        cls.image_data = WsiDicomTestImageData(cls.test_data, cls.tiled_size)
        cls.test_dataset = cls.create_test_dataset(
            cls.frame_count,
            cls.image_data
        )

        folders = cls._get_folders(SLIDE_FOLDER)
        cls.test_folders = {}
        for folder in folders:
            relative_path = cls._get_relative_path(folder)
            cls.test_folders[relative_path] = cls.open(folder)

        if len(cls.test_folders) == 0:
            raise unittest.SkipTest(
                f'no test slide files found for {SLIDE_FOLDER}, '
                'skipping'
            )

    @staticmethod
    def open(folder: Path) -> WsiDicom:
        while next(folder.iterdir()).is_dir():
            folder = next(folder.iterdir())
        return WsiDicom.open(folder)

    @staticmethod
    def _get_folders(slide_folder: Path) -> List[Path]:
        if not slide_folder.exists():
            print("slide folder does not exist")
            return []
        return [
            item for item in slide_folder.iterdir()
            if item.is_dir
        ]

    @staticmethod
    def _get_relative_path(slide_path: Path) -> Path:
        parts = slide_path.parts
        return Path(parts[-1])

    @staticmethod
    def create_test_dataset(
        frame_count: int,
        image_data: WsiDicomTestImageData
    ):
        dataset = Dataset()
        dataset.SOPClassUID = WSI_SOP_CLASS_UID
        dataset.ImageType = [
            'ORGINAL', 'PRIMARY', 'VOLUME', 'NONE'
        ]
        dataset.NumberOfFrames = frame_count
        dataset.SOPInstanceUID = generate_uid()
        dataset.StudyInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = generate_uid()
        dataset.FrameOfReferenceUID = generate_uid()
        dataset.DimensionOrganizationType = 'TILED_FULL'
        dataset.Rows = image_data.tile_size.width
        dataset.Columns = image_data.tile_size.height
        dataset.SamplesPerPixel = image_data.samples_per_pixel
        dataset.PhotometricInterpretation = (
            image_data.photometric_interpretation
        )
        dataset.TotalPixelMatrixColumns = image_data.image_size.width
        dataset.TotalPixelMatrixRows = image_data.image_size.height
        dataset.OpticalPathSequence = DicomSequence([])
        dataset.ImagedVolumeWidth = 1.0
        dataset.ImagedVolumeHeight = 1.0
        dataset.ImagedVolumeDepth = 1.0
        dataset.InstanceNumber = 0
        pixel_measure = Dataset()
        pixel_measure.PixelSpacing = [
            image_data.pixel_spacing.width,
            image_data.pixel_spacing.height
        ]
        pixel_measure.SpacingBetweenSlices = 1.0
        pixel_measure.SliceThickness = 1.0
        shared_functional_group = Dataset()
        shared_functional_group.PixelMeasuresSequence = (
            DicomSequence([pixel_measure])
        )
        dataset.SharedFunctionalGroupsSequence = DicomSequence(
            [shared_functional_group]
        )
        dataset.TotalPixelMatrixFocalPlanes = 1
        dataset.NumberOfOpticalPaths = 1
        dataset.ExtendedDepthOfField = 'NO'
        dataset.FocusMethod = 'AUTO'
        return dataset

    def test_write_preamble(self):
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                write_file._write_preamble()
            self.assertTrue(is_dicom(filepath))

    def test_write_meta(self):
        transfer_syntax = JPEGBaseline8Bit
        instance_uid = generate_uid()
        class_uid = WSI_SOP_CLASS_UID
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
            positions = write_file._write_pixel_data(
                self.image_data,
                self.image_data.default_z,
                self.image_data.default_path,
                1,
                100
            )
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
        LENGTH_BYTES = 4
        frame_offsets = []
        for position in positions:  # Positions are from frame data start
            frame_offsets.append(position + TAG_BYTES + LENGTH_BYTES)
        frame_lengths = [  # Lengths are disiable with 2
            2*math.ceil(len(frame) / 2) for frame in self.test_data
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
            frame_index, table_type = read_file._parse_pixel_data()
            return frame_index

    def assertEndOfFile(self, file: WsiDicomTestFile):
        with self.assertRaises(EOFError):
            file._fp.read(1, need_exact_length=True)

    def test_write_and_read_table(self):
        with TemporaryDirectory() as tempdir:
            for table in [None, 'bot', 'eot']:
                filepath = Path(tempdir + '/' + str(table))
                expected_frame_index = self.write_table(filepath, table)
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
                self.assertEndOfFile(read_file)

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
                self.assertEndOfFile(read_file)

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
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                positions = write_file._write_pixel_data(
                    image_data=self.image_data,
                    z=self.image_data.default_z,
                    path=self.image_data.default_path,
                    workers=1,
                    chunk_size=10
                )
            with WsiDicomTestFile(
                filepath,
                JPEGBaseline8Bit,
                self.frame_count
            ) as read_file:
                for position in positions:
                    read_file._fp.seek(position)
                    tag = read_file._fp.read_tag()
                    self.assertEqual(tag, ItemTag)

    def test_write_unsigned_long_long(self):
        values = [0, 4294967295]
        MODE = '<Q'
        BYTES_PER_ITEM = 8
        with TemporaryDirectory() as tempdir:
            filepath = Path(tempdir + '/1.dcm')
            with WsiDicomFileWriter(filepath) as write_file:
                for value in values:
                    write_file._write_unsigned_long_long(value)

            with WsiDicomTestFile(
                filepath,
                JPEGBaseline8Bit,
                self.frame_count
            ) as read_file:
                for value in values:
                    read_value = unpack(
                        MODE,
                        read_file._fp.read(BYTES_PER_ITEM)
                    )[0]
                    self.assertEqual(read_value, value)

    def test_write(self):
        with TemporaryDirectory() as tempdir:
            for table in [None, 'bot', 'eot']:
                filepath = Path(tempdir + '/' + str(table) + '.dcm')
                with WsiDicomFileWriter(filepath) as write_file:
                    write_file.write(
                        generate_uid(),
                        JPEGBaseline8Bit,
                        self.test_dataset,
                        OrderedDict({
                            (
                                self.image_data.default_path,
                                self.image_data.default_z
                            ):
                            self.image_data
                        }),
                        1,
                        100,
                        table,
                        0
                    )

                with WsiDicomFile(filepath) as read_file:
                    for index, frame in enumerate(self.test_data):
                        read_frame = read_file.read_frame(index)
                        # Stored frame can be up to one byte longer
                        self.assertTrue(
                            0 <=
                            len(read_frame) - len(frame)
                            <= 1
                        )
                        self.assertEqual(read_frame[:len(frame)], frame)

    def test_create_child(self):
        for _, wsi in self.test_folders.items():
            with TemporaryDirectory() as tempdir:
                print(f"running test create child on {wsi.files}")
                target = cast(WsiDicomLevel, wsi.levels[-2])
                source = cast(WsiDicomLevel, wsi.levels[-3])
                new_level = source.create_child(
                    2,
                    Path(tempdir),
                    generate_uid,
                    1,
                    100,
                    'bot',
                    0
                )
                new_level.close()
                with WsiDicom.open(tempdir) as created_wsi:
                    created_size = created_wsi.levels[0].size.to_tuple()
                    target_size = target.size.to_tuple()
                    self.assertEqual(created_size, target_size)

                    created = created_wsi.read_region(
                        (0, 0),
                        0,
                        created_size
                    )
                    original = wsi.read_region(
                        (0, 0),
                        target.level,
                        target_size
                    )
                    blur = ImageFilter.GaussianBlur(2)
                    diff = ImageChops.difference(
                        created.filter(blur),
                        original.filter(blur)
                    )
                    for band_rms in ImageStat.Stat(diff).rms:
                        self.assertLess(band_rms, 2)  # type: ignore
