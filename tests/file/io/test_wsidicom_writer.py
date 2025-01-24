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
import threading
from pathlib import Path
from typing import List, Optional, OrderedDict, Sequence, Tuple

import pytest
from PIL.Image import Image
from pydicom import Sequence as DicomSequence
from pydicom.dataset import Dataset
from pydicom.tag import ItemTag, SequenceDelimiterTag
from pydicom.uid import (
    UID,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEGBaseline8Bit,
    generate_uid,
)

from wsidicom.file.io import (
    OffsetTableType,
    WsiDicomIO,
    WsiDicomReader,
    WsiDicomWriter,
)
from wsidicom.file.io.frame_index.parser import FrameIndexParser
from wsidicom.file.io.wsidicom_writer import (
    WsiDicomEncapsulatedWriter,
    WsiDicomNativeWriter,
)
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.instance import ImageData
from wsidicom.instance.dataset import WsiDataset
from wsidicom.metadata import ImageCoordinateSystem
from wsidicom.uid import WSI_SOP_CLASS_UID

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))


class WsiDicomTestReader(WsiDicomReader):
    """Test version of WsiDicomFile that overrides __init__."""

    def __init__(
        self,
        stream: WsiDicomIO,
        transfer_syntax: UID,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ):
        self._stream = stream
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
        dataset.NumberOfFrames = frame_count
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
        self._dataset = WsiDataset(dataset)
        self._frame_index_parser: Optional[FrameIndexParser] = None
        self._frame_index: Optional[List[Tuple[int, int]]] = None
        self._lock = threading.Lock()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @classmethod
    def open(
        cls,
        filepath: Path,
        transfer_syntax: UID,
        frame_count: int,
        bits: int,
        tile_size: Size,
        samples_per_pixel: int,
    ) -> "WsiDicomTestReader":
        stream = WsiDicomIO(
            open(filepath, "rb"),
            transfer_syntax=transfer_syntax,
            filepath=filepath,
            owned=True,
        )
        return cls(
            stream, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        )


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
        return "RGB"

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return None

    @property
    def lossy_compressed(self) -> bool:
        return False

    @property
    def thread_safe(self) -> bool:
        return True

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> Image:
        raise NotImplementedError()

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self._data[tile.x + tile.y * self.tiled_size.width]


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
    dataset.PixelRepresentation = 0
    dataset.PlanarConfiguration = 0
    dataset.TotalPixelMatrixColumns = image_data.image_size.width
    dataset.TotalPixelMatrixRows = image_data.image_size.height
    dataset.OpticalPathSequence = DicomSequence([])
    dataset.ImagedVolumeWidth = 1.0
    dataset.ImagedVolumeHeight = 1.0
    dataset.ImagedVolumeDepth = 1.0
    dataset.InstanceNumber = 0
    pixel_measure = Dataset()
    pixel_measure.PixelSpacing = [
        image_data.pixel_spacing.height,
        image_data.pixel_spacing.width,
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


@pytest.mark.unittest
class TestWsiDicomWriter:
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
        with WsiDicomEncapsulatedWriter.open(
            filepath, transfer_syntax, writen_table_type
        ) as writer:
            writen_frame_positions = writer._write_pixel_data(
                {(image_data.default_path, image_data.default_z): image_data},
                WsiDataset(dataset),
                (0, 0),
                1,
                100,
                1,
                None,
            )

        # Assert
        with WsiDicomTestReader.open(
            filepath, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        ) as reader:
            read_table_type = reader.offset_table_type
            read_frame_positions = reader.frame_index
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        frame_offsets = [
            position + TAG_BYTES + LENGTH_BYTES for position in writen_frame_positions
        ]
        frame_lengths = [  # Lengths are divisible with 2
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
        with WsiDicomNativeWriter.open(filepath, transfer_syntax) as writer:
            writer._write_pixel_data(
                {(image_data.default_path, image_data.default_z): image_data},
                WsiDataset(dataset),
                (0, 0),
                1,
                100,
                1,
                None,
            )

        # Assert
        with WsiDicomTestReader.open(
            filepath, transfer_syntax, frame_count, bits, tile_size, samples_per_pixel
        ) as read_file:
            read_table_type = read_file.offset_table_type
            read_frame_positions = read_file.frame_index
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

    def test_write_pixel_end(self, tmp_path: Path):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomEncapsulatedWriter.open(
            filepath, JPEGBaseline8Bit, OffsetTableType.BASIC
        ) as writer:
            writer._write_pixel_data_end_tag()

        # Assert
        with WsiDicomIO(
            open(filepath, "rb"), JPEGBaseline8Bit, filepath, True
        ) as read_file:
            tag = read_file.read_tag()
            assert tag == SequenceDelimiterTag
            length = read_file.read_tag_length(True)
            assert length == 0

    @pytest.mark.parametrize("transfer_syntax", [JPEGBaseline8Bit])
    def test_write_pixel_data(
        self, image_data: ImageData, tmp_path: Path, transfer_syntax: UID
    ):
        # Arrange
        filepath = tmp_path.joinpath("1.dcm")

        # Act
        with WsiDicomWriter.open(
            filepath, transfer_syntax, OffsetTableType.EMPTY
        ) as writer:
            positions = writer._write_tiles(
                image_data=image_data,
                z=image_data.default_z,
                path=image_data.default_path,
                workers=1,
                chunk_size=10,
            )

        with WsiDicomIO(
            open(filepath, "rb"), JPEGBaseline8Bit, filepath, True
        ) as read_file:
            for position in positions:
                read_file.seek(position)
                tag = read_file.read_tag()
                assert tag == ItemTag

    @pytest.mark.parametrize(
        ["transfer_syntax", "table_type"],
        [
            (JPEGBaseline8Bit, OffsetTableType.EMPTY),
            (JPEGBaseline8Bit, OffsetTableType.BASIC),
            (JPEGBaseline8Bit, OffsetTableType.EXTENDED),
            (ExplicitVRBigEndian, OffsetTableType.NONE),
            (ExplicitVRLittleEndian, OffsetTableType.NONE),
            (ImplicitVRLittleEndian, OffsetTableType.NONE),
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
        with WsiDicomWriter.open(filepath, transfer_syntax, table_type) as writer:
            writer.write(
                generate_uid(),
                WsiDataset(dataset),
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
                0,
            )

        # Assert
        with WsiDicomReader(
            WsiDicomIO(open(filepath, "rb"), transfer_syntax, filepath, True)
        ) as read_file:
            for index, frame in enumerate(frames):
                read_frame = read_file.read_frame(index)
                # Stored frame can be up to one byte longer
                assert 0 <= len(read_frame) - len(frame) <= 1
                assert read_frame[: len(frame)] == frame
