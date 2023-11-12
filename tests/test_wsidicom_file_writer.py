#    Copyright 2021, 2023 SECTRA AB
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
from pathlib import Path
from struct import unpack
from typing import Callable, List, Optional, OrderedDict, Sequence, cast

import pytest
from PIL import ImageChops, ImageFilter, ImageStat
from PIL.Image import Image as PILImage
from pydicom import Sequence as DicomSequence
from pydicom.dataset import Dataset
from pydicom.filebase import DicomFile
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.tag import ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import (
    JPEG2000,
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
    generate_uid,
    UncompressedTransferSyntaxes,
)

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.file.wsidicom_file import WsiDicomFile
from wsidicom.file.wsidicom_file_base import OffsetTableType
from wsidicom.file.wsidicom_file_target import WsiDicomFileTarget
from wsidicom.file.wsidicom_file_writer import WsiDicomFileWriter
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.group.level import Level
from wsidicom.instance import ImageCoordinateSystem, ImageData
from wsidicom.instance.dataset import WsiDataset
from wsidicom.uid import WSI_SOP_CLASS_UID

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))


class WsiDicomTestFile(WsiDicomFile):
    """Test version of WsiDicomFile that overrides __init__."""

    def __init__(
        self,
        filepath: Path,
        transfer_syntax: UID,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        self._filepath = filepath
        self._file = DicomFile(filepath, mode="rb")
        self._file.is_little_endian = transfer_syntax.is_little_endian
        self._file.is_implicit_VR = transfer_syntax.is_implicit_VR
        self._frame_count = frame_count
        self._pixel_data_position = 0
        self._owned = True
        self._transfer_syntax_uid = transfer_syntax
        dataset = Dataset()
        dataset.BitsAllocated = (bits // 8) * 8
        dataset.BitsStored = bits
        dataset.Columns = tile_size.width
        dataset.Rows = tile_size.height
        dataset.SamplesPerPixel = samples_per_pixel

        self._dataset = WsiDataset(dataset)
        self.__enter__()

    @property
    def frame_count(self) -> int:
        return self._frame_count


class WsiDicomTestImageData(ImageData):
    def __init__(self, data: Sequence[bytes], tiled_size: Size) -> None:
        if len(data) != tiled_size.area:
            raise ValueError("Number of frames and tiled size area differ")
        TILE_SIZE = Size(10, 10)
        self._data = data
        self._tile_size = TILE_SIZE
        self._image_size = tiled_size * TILE_SIZE

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
    def bits(self) -> int:
        return 8

    @property
    def photometric_interpretation(self) -> str:
        return "YBR"

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return None

    @property
    def lossy_compressed(self) -> bool:
        return False

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> PILImage:
        raise NotImplementedError()

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self._data[tile.x + tile.y * self.tiled_size.width]


@pytest.fixture()
def tiled_size():
    yield Size(2, 2)


@pytest.fixture()
def frame_count(tiled_size: Size):
    yield tiled_size.area


@pytest.fixture()
def rng():
    SEED = 0
    yield random.Random(SEED)


@pytest.fixture()
def bits():
    yield 8


@pytest.fixture
def samples_per_pixel():
    yield 3


@pytest.fixture
def tile_size():
    yield Size(10, 10)


@pytest.fixture()
def frames(
    rng: random.Random,
    transfer_syntax: UID,
    frame_count: int,
    bits: int,
    samples_per_pixel: int,
    tile_size: Size,
):
    if transfer_syntax in UncompressedTransferSyntaxes:
        min_frame_length = bits * tile_size.area * samples_per_pixel // 8
        max_frame_length = min_frame_length
    else:
        min_frame_length = 2
        max_frame_length = 100
    lengths = [
        rng.randint(min_frame_length, max_frame_length) for i in range(frame_count)
    ]
    yield [
        rng.getrandbits(length * 8).to_bytes(length, sys.byteorder)
        for length in lengths
    ]


@pytest.fixture()
def image_data(frames: List[bytes], tiled_size: Size):
    yield WsiDicomTestImageData(frames, tiled_size)


@pytest.fixture()
def dataset(image_data: ImageData, frame_count: int):
    assert image_data.pixel_spacing is not None
    dataset = Dataset()
    dataset.SOPClassUID = WSI_SOP_CLASS_UID
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
    dataset.NumberOfFrames = frame_count
    dataset.SOPInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.DimensionOrganizationType = "TILED_FULL"
    dataset.Rows = image_data.tile_size.width
    dataset.Columns = image_data.tile_size.height
    dataset.SamplesPerPixel = image_data.samples_per_pixel
    dataset.BitsStored = image_data.bits
    dataset.BitsAllocated = image_data.bits // 8 * 8
    dataset.PhotometricInterpretation = image_data.photometric_interpretation
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
        image_data.pixel_spacing.height,
    ]
    pixel_measure.SpacingBetweenSlices = 1.0
    pixel_measure.SliceThickness = 1.0
    shared_functional_group = Dataset()
    shared_functional_group.PixelMeasuresSequence = DicomSequence([pixel_measure])
    dataset.SharedFunctionalGroupsSequence = DicomSequence([shared_functional_group])
    dataset.TotalPixelMatrixFocalPlanes = 1
    dataset.NumberOfOpticalPaths = 1
    dataset.ExtendedDepthOfField = "NO"
    dataset.FocusMethod = "AUTO"
    yield dataset


@pytest.mark.save
class TestWsiDicomFileWriter:
    @staticmethod
    def assertEndOfFile(file: WsiDicomTestFile):
        with pytest.raises(EOFError):
            file._file.read(1, need_exact_length=True)

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
        ],
    )
    def test_write_preamble(self, tmp_path: Path, transfer_syntax: UID):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, transfer_syntax) as write_file:
            write_file._write_preamble()

        # Assert
        assert is_dicom(filepath)

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            DeflatedExplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
            JPEG2000,
            JPEG2000Lossless,
            JPEGBaseline8Bit,
            JPEGExtended12Bit,
            JPEGLosslessP14,
            JPEGLosslessSV1,
            JPEGLSLossless,
            JPEGLSNearLossless,
            RLELossless,
        ],
    )
    def test_write_meta(self, tmp_path: Path, transfer_syntax: UID):
        # Arrange
        instance_uid = generate_uid()
        class_uid = WSI_SOP_CLASS_UID
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, JPEGBaseline8Bit) as write_file:
            write_file._write_preamble()
            write_file._write_file_meta(instance_uid, transfer_syntax)
        file_meta = read_file_meta_info(filepath)

        # Assert
        assert file_meta.TransferSyntaxUID == transfer_syntax
        assert file_meta.MediaStorageSOPInstanceUID == instance_uid
        assert file_meta.MediaStorageSOPClassUID == class_uid

    @pytest.mark.parametrize(
        "writen_table_type",
        [
            OffsetTableType.EMPTY,
            OffsetTableType.BASIC,
            OffsetTableType.EXTENDED,
        ],
    )
    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
        ],
    )
    def test_write_encapsulated_pixel_data(
        self,
        dataset: Dataset,
        image_data: ImageData,
        frames: List[bytes],
        frame_count: int,
        writen_table_type: OffsetTableType,
        transfer_syntax: UID,
        tmp_path: Path,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath(str(writen_table_type))

        # Act
        with WsiDicomFileWriter.open(filepath, transfer_syntax) as write_file:
            writen_frame_positions = write_file._write_encapsulated_pixel_data(
                dataset,
                {(image_data.default_path, image_data.default_z): image_data},
                1,
                100,
                writen_table_type,
            )

        # Assert
        with WsiDicomTestFile(
            filepath, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            read_frame_positions, read_table_type = read_file._parse_pixel_data()
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        frame_offsets = [
            position + TAG_BYTES + LENGTH_BYTES for position in writen_frame_positions
        ]
        frame_lengths = [  # Lengths are divisable with 2
            2 * math.ceil(len(frame) / 2) for frame in frames
        ]
        expected_frame_positons = [
            (offset, length) for offset, length in zip(frame_offsets, frame_lengths)
        ]
        assert expected_frame_positons == read_frame_positions
        assert writen_table_type == read_table_type

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
        ],
    )
    def test_write_unencapsulated_pixel_data(
        self,
        dataset: Dataset,
        image_data: ImageData,
        frames: List[bytes],
        frame_count: int,
        transfer_syntax: UID,
        tmp_path: Path,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath(str(transfer_syntax))

        # Act
        with WsiDicomFileWriter.open(filepath, transfer_syntax) as write_file:
            write_file._write_unencapsulated_pixel_data(
                dataset,
                {(image_data.default_path, image_data.default_z): image_data},
                1,
                100,
            )

        # Assert
        with WsiDicomTestFile(
            filepath, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            read_frame_positions, read_table_type = read_file._parse_pixel_data()
        assert read_table_type == OffsetTableType.NONE
        if transfer_syntax == ImplicitVRLittleEndian:
            offset = 8
        else:
            offset = 12
        for index, frame_position in enumerate(read_frame_positions):
            assert (
                frame_position[0]
                == offset + index * bits * tile_size.area * samples_per_pixel // 8
            )
            assert frame_position[1] == bits * tile_size.area * samples_per_pixel // 8

    def test_reserve_bot(
        self,
        tmp_path: Path,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, JPEGBaseline8Bit) as write_file:
            write_file._reserve_bot(frame_count)

        # Assert
        with WsiDicomTestFile(
            filepath, JPEGBaseline8Bit, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            tag = read_file._file.read_tag()
            assert tag == ItemTag
            BOT_ITEM_LENGTH = 4
            length = read_file._read_tag_length()
            assert length == BOT_ITEM_LENGTH * frame_count
            for frame in range(frame_count):
                assert read_file._file.read_UL() == 0
            self.assertEndOfFile(read_file)

    def test_reserve_eot(
        self,
        tmp_path: Path,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, JPEGBaseline8Bit) as write_file:
            write_file._reserve_eot(frame_count)

        # Assert
        with WsiDicomTestFile(
            filepath, JPEGBaseline8Bit, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            tag = read_file._file.read_tag()
            assert tag == Tag("ExtendedOffsetTable")
            read_file._read_tag_vr()
            EOT_ITEM_LENGTH = 8
            length = read_file._read_tag_length()
            assert length == EOT_ITEM_LENGTH * frame_count
            for frame in range(frame_count):
                assert unpack("<Q", read_file._file.read(EOT_ITEM_LENGTH))[0] == 0

            tag = read_file._file.read_tag()
            read_file._read_tag_vr()
            assert tag == Tag("ExtendedOffsetTableLengths")
            length = read_file._read_tag_length()
            EOT_ITEM_LENGTH = 8
            assert length == EOT_ITEM_LENGTH * frame_count
            for frame in range(frame_count):
                assert unpack("<Q", read_file._file.read(EOT_ITEM_LENGTH))[0] == 0
            self.assertEndOfFile(read_file)

    def test_write_pixel_end(
        self,
        tmp_path: Path,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, JPEGBaseline8Bit) as write_file:
            write_file._write_pixel_data_end()

        # Assert
        with WsiDicomTestFile(
            filepath, JPEGBaseline8Bit, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            tag = read_file._file.read_le_tag()
            assert tag == SequenceDelimiterTag
            length = read_file._read_tag_length()
            assert length == 0

    @pytest.mark.parametrize("transfer_syntax", [JPEGBaseline8Bit])
    def test_write_pixel_data(
        self,
        image_data: ImageData,
        tmp_path: Path,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomFileWriter.open(filepath, JPEGBaseline8Bit) as write_file:
            positions = write_file._write_pixel_data(
                image_data=image_data,
                encapsulate=True,
                z=image_data.default_z,
                path=image_data.default_path,
                workers=1,
                chunk_size=10,
            )

        # Assert
        with WsiDicomTestFile(
            filepath, JPEGBaseline8Bit, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            for position in positions:
                read_file._file.seek(position)
                tag = read_file._file.read_le_tag()
                assert tag == ItemTag

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
        ],
    )
    def test_write_unsigned_long_long(
        self,
        tmp_path: Path,
        frame_count: int,
        transfer_syntax: UID,
        bits,
        tile_size,
        samples_per_pixel,
    ):
        # Arrange
        values = [0, 4294967295]
        MODE = "<Q"
        BYTES_PER_ITEM = 8

        # Act
        filepath = tmp_path.joinpath("1.dcm")
        with WsiDicomFileWriter.open(filepath, transfer_syntax) as write_file:
            for value in values:
                write_file._write_unsigned_long_long(value)

        # Assert
        with WsiDicomTestFile(
            filepath, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            for value in values:
                read_value = unpack(MODE, read_file._file.read(BYTES_PER_ITEM))[0]
                assert read_value == value

    @pytest.mark.parametrize(
        "table_type",
        [
            OffsetTableType.EMPTY,
            OffsetTableType.BASIC,
            OffsetTableType.EXTENDED,
        ],
    )
    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
        ],
    )
    def test_write(
        self,
        image_data: ImageData,
        dataset: Dataset,
        frames: List[bytes],
        tmp_path: Path,
        table_type: OffsetTableType,
        transfer_syntax: UID,
    ):
        # Arrange
        filepath = tmp_path.joinpath(str(table_type))

        # Act
        with WsiDicomFileWriter.open(filepath, transfer_syntax) as write_file:
            write_file.write(
                generate_uid(),
                transfer_syntax,
                dataset,
                OrderedDict(
                    {
                        (
                            image_data.default_path,
                            image_data.default_z,
                        ): image_data
                    }
                ),
                1,
                100,
                table_type,
                0,
            )

        # Assert
        with WsiDicomFile.open(filepath) as read_file:
            for index, frame in enumerate(frames):
                read_frame = read_file.read_frame(index)
                # Stored frame can be up to one byte longer
                assert 0 <= len(read_frame) - len(frame) <= 1
                assert read_frame[: len(frame)] == frame

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_create_child(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        target_level = cast(Level, wsi.levels[-2])
        source_level = cast(Level, wsi.levels[-3])

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            generate_uid,
            1,
            100,
            "bot",
        ) as target:
            target._save_and_open_level(source_level, wsi.pixel_spacing, 2)

        # Assert
        with WsiDicom.open(tmp_path) as created_wsi:
            created_size = created_wsi.levels[0].size.to_tuple()
            target_size = target_level.size.to_tuple()
            assert created_size == target_size

            created = created_wsi.read_region((0, 0), 0, created_size)
            original = wsi.read_region((0, 0), target_level.level, target_size)
            blur = ImageFilter.GaussianBlur(2)
            diff = ImageChops.difference(created.filter(blur), original.filter(blur))
            for band_rms in ImageStat.Stat(diff).rms:
                assert band_rms < 2
