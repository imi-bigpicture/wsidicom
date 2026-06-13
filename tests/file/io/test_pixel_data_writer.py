#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Unit tests for EncapsulatedPixelDataWriter and NativePixelDataWriter."""

import struct
from pathlib import Path

import pytest
from pydicom import Sequence as DicomSequence
from pydicom.dataset import Dataset
from pydicom.tag import ItemTag, SequenceDelimiterTag
from pydicom.uid import (
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    VLWholeSlideMicroscopyImageStorage,
    generate_uid,
)

from wsidicom.file.io import OffsetTableType, WsiDicomIO
from wsidicom.file.io.wsidicom_writer import (
    EncapsulatedPixelDataWriter,
    NativePixelDataWriter,
)
from wsidicom.file.wsidicom_stream_opener import WsiDicomStreamOpener
from wsidicom.geometry import Size
from wsidicom.instance.dataset import WsiDataset

ITEM_HEADER_SIZE = 8  # 4-byte tag + 4-byte length


def _open_stream(filepath: Path, transfer_syntax) -> WsiDicomIO:
    return WsiDicomStreamOpener().open_for_writing(filepath, "w+b", transfer_syntax)


def _make_dataset(
    frame_count: int, tile_size: Size | None = None
) -> WsiDataset:
    """Create a minimal WsiDataset for testing."""
    if tile_size is None:
        tile_size = Size(10, 10)
    ds = Dataset()
    ds.SOPClassUID = VLWholeSlideMicroscopyImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
    ds.DimensionOrganizationType = "TILED_FULL"
    ds.NumberOfFrames = frame_count
    ds.Rows = tile_size.height
    ds.Columns = tile_size.width
    ds.SamplesPerPixel = 3
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.PhotometricInterpretation = "RGB"
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.TotalPixelMatrixColumns = tile_size.width
    ds.TotalPixelMatrixRows = tile_size.height
    ds.OpticalPathSequence = DicomSequence([])
    ds.ImagedVolumeWidth = 1.0
    ds.ImagedVolumeHeight = 1.0
    ds.ImagedVolumeDepth = 1.0
    ds.InstanceNumber = 0
    pixel_measure = Dataset()
    pixel_measure.PixelSpacing = [1.0, 1.0]
    pixel_measure.SpacingBetweenSlices = 1.0
    pixel_measure.SliceThickness = 1.0
    shared = Dataset()
    shared.PixelMeasuresSequence = DicomSequence([pixel_measure])
    ds.SharedFunctionalGroupsSequence = DicomSequence([shared])
    ds.TotalPixelMatrixFocalPlanes = 1
    ds.NumberOfOpticalPaths = 1
    ds.ExtendedDepthOfField = "NO"
    ds.FocusMethod = "AUTO"
    return WsiDataset(ds)


@pytest.mark.unittest
class TestEncapsulatedPixelDataWriter:
    """Tests for EncapsulatedPixelDataWriter."""

    def test_write_tile_returns_position_before_item_header(self, tmp_path: Path):
        """write_tile returns the file position of the item header start."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        tile = b"\xff\xd8\xff\xe0test_data"

        # Act
        position = writer.write_tile(tile)
        stream.seek(position)
        tag = stream.read_tag()

        # Assert
        assert tag == ItemTag

    def test_write_tile_item_length_matches(self, tmp_path: Path):
        """Item length field matches the tile data (with padding)."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        tile = b"\x01\x02\x03\x04\x05"  # 5 bytes, odd

        # Act
        position = writer.write_tile(tile)
        stream.seek(position + 4)  # skip item tag
        length = stream.read_UL()

        # Assert
        assert length == 6  # padded to even

    def test_write_tile_data_readable(self, tmp_path: Path):
        """Written tile data matches input (ignoring padding)."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        tile = b"\x01\x02\x03\x04\x05\x06"

        # Act
        position = writer.write_tile(tile)
        stream.seek(position + ITEM_HEADER_SIZE)
        data = stream.read(len(tile))

        # Assert
        assert data == tile

    def test_consecutive_tiles_positions(self, tmp_path: Path):
        """Consecutive tiles have correct spacing (data + 8-byte header)."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        tile1 = b"\x01\x02\x03\x04"  # 4 bytes, even
        tile2 = b"\x05\x06\x07\x08"

        # Act
        pos1 = writer.write_tile(tile1)
        pos2 = writer.write_tile(tile2)

        # Assert
        assert pos2 - pos1 == ITEM_HEADER_SIZE + len(tile1)

    def test_write_pixel_data_start_empty_bot(self, tmp_path: Path):
        """EMPTY offset table writes an empty BOT item."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        dataset = _make_dataset(frame_count=2)

        # Act
        pixels_start, table_writer = writer.write_pixel_data_start(dataset)
        stream.seek(pixels_start - ITEM_HEADER_SIZE)
        tag = stream.read_tag()
        length = stream.read_UL()

        # Assert
        assert table_writer is None
        assert tag == ItemTag
        assert length == 0

    def test_write_pixel_data_start_basic_bot(self, tmp_path: Path):
        """BASIC offset table reserves space for BOT entries."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.BASIC, JPEGBaseline8Bit
        )
        frame_count = 3
        dataset = _make_dataset(frame_count=frame_count)
        expected_bot_size = 4 * frame_count  # BOT reserves 4 bytes per frame

        # Act
        pixels_start, table_writer = writer.write_pixel_data_start(dataset)
        stream.seek(pixels_start - expected_bot_size - ITEM_HEADER_SIZE)
        tag = stream.read_tag()
        length = stream.read_UL()

        # Assert
        assert table_writer is not None
        assert tag == ItemTag
        assert length == expected_bot_size

    def test_write_pixel_data_end_writes_sequence_delimiter(self, tmp_path: Path):
        """write_pixel_data_end writes a sequence delimiter tag."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        dataset = _make_dataset(frame_count=1)
        pixels_start, table_writer = writer.write_pixel_data_start(dataset)
        pos = writer.write_tile(b"\x01\x02\x03\x04")

        # Act
        writer.write_pixel_data_end(
            table_writer,
            pixels_start,
            0,
            0,
            dataset,
            [pos],
            None,
        )
        delimiter_pos = stream.tell() - ITEM_HEADER_SIZE
        stream.seek(delimiter_pos)
        tag = stream.read_tag()

        # Assert
        assert tag == SequenceDelimiterTag

    def test_calculate_size_single_frame(self):
        """_calculate_size returns data bytes excluding item headers."""
        # Arrange
        tile_data_size = 100
        frame_positions = [0]
        last_frame_end = ITEM_HEADER_SIZE + tile_data_size

        # Act
        result = EncapsulatedPixelDataWriter._calculate_size(
            frame_positions, last_frame_end
        )

        # Assert
        assert result == tile_data_size

    def test_calculate_size_multiple_frames(self):
        """_calculate_size correctly subtracts 8 bytes per frame."""
        # Arrange
        tile_sizes = [50, 60, 70]
        positions = []
        offset = 0
        for size in tile_sizes:
            positions.append(offset)
            offset += ITEM_HEADER_SIZE + size
        last_frame_end = offset

        # Act
        result = EncapsulatedPixelDataWriter._calculate_size(positions, last_frame_end)

        # Assert
        assert result == sum(tile_sizes)

    def test_odd_tile_padded_to_even(self, tmp_path: Path):
        """Odd-length tiles are padded to even length in the file."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", JPEGBaseline8Bit)
        writer = EncapsulatedPixelDataWriter(
            stream, OffsetTableType.EMPTY, JPEGBaseline8Bit
        )
        odd_tile = b"\x01\x02\x03"  # 3 bytes

        # Act
        pos1 = writer.write_tile(odd_tile)
        pos2 = writer.write_tile(b"\x04\x05\x06\x07")

        # Assert - padded to 4 bytes + 8 header = 12 offset
        assert pos2 - pos1 == ITEM_HEADER_SIZE + 4


@pytest.mark.unittest
class TestNativePixelDataWriter:
    """Tests for NativePixelDataWriter."""

    def test_write_tile_returns_position(self, tmp_path: Path):
        """write_tile returns the position where data starts."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", ExplicitVRLittleEndian)
        writer = NativePixelDataWriter(stream)
        tile = b"\x01\x02\x03\x04"

        # Act
        position = writer.write_tile(tile)
        stream.seek(position)
        data = stream.read(len(tile))

        # Assert
        assert data == tile

    def test_write_tile_no_item_header(self, tmp_path: Path):
        """Native tiles have no item headers — data is written directly."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", ExplicitVRLittleEndian)
        writer = NativePixelDataWriter(stream)
        tile = b"\x01\x02\x03\x04"

        # Act
        pos1 = writer.write_tile(tile)
        pos2 = writer.write_tile(tile)

        # Assert - no 8-byte header between tiles
        assert pos2 - pos1 == len(tile)

    def test_write_pixel_data_start_length(self, tmp_path: Path):
        """Native pixel data start writes the correct total PixelData length."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", ExplicitVRLittleEndian)
        writer = NativePixelDataWriter(stream)
        tile_size = Size(10, 10)
        frame_count = 4
        dataset = _make_dataset(frame_count=frame_count, tile_size=tile_size)
        # 10*10 px * 3 samples (RGB) * 1 byte (8-bit) * 4 frames
        expected_length = 10 * 10 * 3 * 1 * 4

        # Act
        pixels_start, table_writer = writer.write_pixel_data_start(dataset)

        # Assert - the 4-byte length field (last 4 bytes of the OW element
        # header) holds the full pixel-data length.
        assert table_writer is None
        stream.seek(pixels_start - 4)
        (written_length,) = struct.unpack("<I", stream.read(4))
        assert written_length == expected_length

    def test_write_pixel_data_end_is_noop(self, tmp_path: Path):
        """Native write_pixel_data_end does nothing."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", ExplicitVRLittleEndian)
        writer = NativePixelDataWriter(stream)
        dataset = _make_dataset(frame_count=1)
        pos_before = stream.tell()

        # Act
        writer.write_pixel_data_end(None, 0, 0, 0, dataset, [0], None)

        # Assert
        assert stream.tell() == pos_before

    def test_consecutive_tiles_contiguous(self, tmp_path: Path):
        """Multiple tiles are written contiguously."""
        # Arrange
        stream = _open_stream(tmp_path / "test.dcm", ExplicitVRLittleEndian)
        writer = NativePixelDataWriter(stream)
        tiles = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08", b"\x09\x0a\x0b\x0c"]

        # Act
        positions = [writer.write_tile(t) for t in tiles]
        stream.seek(positions[0])
        all_data = stream.read(sum(len(t) for t in tiles))

        # Assert
        for i in range(len(tiles) - 1):
            assert positions[i + 1] == positions[i] + len(tiles[i])
        assert all_data == b"".join(tiles)
