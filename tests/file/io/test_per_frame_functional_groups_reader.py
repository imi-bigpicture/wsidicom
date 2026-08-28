#    Copyright 2026 SECTRA AB
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

from collections.abc import Sequence as TypingSequence
from io import BytesIO

import pytest
from pydicom.charset import convert_encodings
from pydicom.dataset import Dataset
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import write_dataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    UID,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEGBaseline8Bit,
)
from upath import UPath

from wsidicom.file.io.per_frame_functional_groups_reader import (
    PerFrameFunctionalGroupsReader,
    ShortStringElement,
    UnscannablePerFrameGroupsException,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.tags import (
    OpticalPathIdentifierTag,
    PerFrameFunctionalGroupsSequenceTag,
)

PIXEL_DATA_HEADER = b"\xe0\x7f\x10\x00OB\x00\x00\xff\xff\xff\xff"
"""Header of an encapsulated (7FE0,0010), the element a sequence is followed by."""


def create_frame(
    column: int,
    row: int,
    z_offset: str | None = "0.0",
    optical_path: str | None = None,
) -> Dataset:
    """Create one per frame functional group item."""
    position = Dataset()
    position.XOffsetInSlideCoordinateSystem = str(round(column * 0.128, 5))
    position.YOffsetInSlideCoordinateSystem = str(round(row * 0.128, 5))
    if z_offset is not None:
        position.ZOffsetInSlideCoordinateSystem = z_offset
    position.ColumnPositionInTotalImagePixelMatrix = column
    position.RowPositionInTotalImagePixelMatrix = row
    frame = Dataset()
    frame.PlanePositionSlideSequence = Sequence([position])
    if optical_path is not None:
        identification = Dataset()
        identification.OpticalPathIdentifier = optical_path
        frame.OpticalPathIdentificationSequence = Sequence([identification])
    return frame


def create_file(
    frames: TypingSequence[Dataset], undefined_length: bool = True
) -> WsiDicomIO:
    """Write frames as a per frame functional groups sequence in a minimal file.

    The sequence starts at offset zero and is followed by a pixel data header, as in a
    real instance. Written by pydicom, so the bytes are what a writer produces.

    Parameters
    ----------
    frames: TypingSequence[Dataset]
        Per frame functional group items to write.
    undefined_length: bool = True
        Write the sequence delimited rather than with a declared length. Both occur in
        files from real scanners.

    Returns
    -------
    WsiDicomIO
        File holding the sequence followed by a pixel data header.
    """
    dataset = Dataset()
    dataset.PerFrameFunctionalGroupsSequence = Sequence(frames)
    dataset[PerFrameFunctionalGroupsSequenceTag].is_undefined_length = undefined_length
    buffer = DicomBytesIO()
    buffer.is_little_endian = True
    buffer.is_implicit_VR = False
    write_dataset(buffer, dataset)
    return WsiDicomIO(
        BytesIO(buffer.getvalue() + PIXEL_DATA_HEADER),
        filepath=UPath("per_frame.dcm"),
        transfer_syntax=JPEGBaseline8Bit,
    )


def create_reader(
    frames: TypingSequence[Dataset],
    frame_count: int | None = None,
    undefined_length: bool = True,
    transfer_syntax: UID = JPEGBaseline8Bit,
    chunk_size: int | None = None,
) -> PerFrameFunctionalGroupsReader:
    """Create a reader for a file holding `frames`."""
    return PerFrameFunctionalGroupsReader(
        create_file(frames, undefined_length),
        0,
        len(frames) if frame_count is None else frame_count,
        transfer_syntax,
        chunk_size=chunk_size,
    )


def create_file_without_sequence() -> WsiDicomIO:
    """Create a file holding a pixel data header and nothing else."""
    return WsiDicomIO(
        BytesIO(PIXEL_DATA_HEADER),
        filepath=UPath("per_frame.dcm"),
        transfer_syntax=JPEGBaseline8Bit,
    )


@pytest.mark.unittest
class TestPerFrameFunctionalGroupsReader:
    @pytest.mark.parametrize("undefined_length", [True, False])
    @pytest.mark.parametrize("frame_count", [1, 2, 17])
    def test_read_positions(self, frame_count: int, undefined_length: bool):
        # Arrange
        frames = [
            create_frame(column=index + 1, row=index + 2)
            for index in range(frame_count)
        ]
        reader = create_reader(frames, undefined_length=undefined_length)

        # Act
        positions = reader.read_positions()

        # Assert
        assert list(positions.columns) == [index + 1 for index in range(frame_count)]
        assert list(positions.rows) == [index + 2 for index in range(frame_count)]

    @pytest.mark.parametrize("chunk_size", [None, 256])
    def test_read_positions_of_values_of_differing_length(self, chunk_size: int | None):
        """Values of one element are taken in one pass when they are all of a length.
        These are not, as a decimal string is only padded to an even length, so they
        have to be found one at a time instead."""
        # Arrange - "0.5" is padded to four bytes and "0.125" takes six.
        frames = [
            create_frame(1, 2, z_offset="0.5"),
            create_frame(3, 4, z_offset="0.125"),
            create_frame(5, 6, z_offset="0.5"),
        ]
        reader = create_reader(frames, chunk_size=chunk_size)

        # Act
        positions = reader.read_positions()

        # Assert
        assert list(positions.columns) == [1, 3, 5]
        assert positions.z_offsets is not None
        assert list(positions.z_offsets) == [0.5, 0.125, 0.5]

    def test_read_positions_of_every_element(self):
        # Arrange
        reader = create_reader(
            [
                create_frame(1, 2, z_offset="0.5", optical_path="1"),
                create_frame(3, 4, z_offset="1.5", optical_path="2"),
            ]
        )

        # Act
        positions = reader.read_positions()

        # Assert
        assert list(positions.columns) == [1, 3]
        assert list(positions.rows) == [2, 4]
        assert positions.z_offsets is not None
        assert list(positions.z_offsets) == [0.5, 1.5]
        assert positions.optical_path_identifiers is not None
        assert list(positions.optical_path_identifiers) == ["1", "2"]

    def test_read_positions_ends_at_next_element(self):
        # Arrange
        file = create_file([create_frame(1, 1), create_frame(2, 1)])
        reader = PerFrameFunctionalGroupsReader(file, 0, 2, JPEGBaseline8Bit)

        # Act
        reader.read_positions()

        # Assert
        file.seek(reader.end_of_sequence)
        assert file.read(4) == b"\xe0\x7f\x10\x00"  # PixelData follows the sequence

    def test_end_of_sequence_before_reading(self):
        # Arrange
        reader = create_reader([create_frame(1, 1)])

        # Act & Assert
        with pytest.raises(ValueError):
            assert reader.end_of_sequence

    def test_read_positions_without_z_offsets(self):
        # Arrange
        reader = create_reader(
            [create_frame(1, 1, z_offset=None), create_frame(2, 1, z_offset=None)]
        )

        # Act
        positions = reader.read_positions()

        # Assert
        assert positions.z_offsets is None

    def test_read_positions_without_optical_path_identifiers(self):
        # Arrange
        reader = create_reader([create_frame(1, 1), create_frame(2, 1)])

        # Act
        positions = reader.read_positions()

        # Assert
        assert positions.optical_path_identifiers is None

    def test_raises_when_element_is_in_some_frames_only(self):
        """A value present in some frames cannot be matched to frames by position."""
        # Arrange
        reader = create_reader(
            [create_frame(1, 1, z_offset="0.5"), create_frame(2, 1, z_offset=None)]
        )

        # Act & Assert
        with pytest.raises(UnscannablePerFrameGroupsException):
            reader.read_positions()

    def test_raises_when_frame_count_does_not_match(self):
        # Arrange
        reader = create_reader([create_frame(1, 1), create_frame(2, 1)], frame_count=3)

        # Act & Assert
        with pytest.raises(UnscannablePerFrameGroupsException):
            reader.read_positions()

    def test_raises_when_not_positioned_at_sequence(self):
        # Arrange
        reader = PerFrameFunctionalGroupsReader(
            create_file_without_sequence(), 0, 1, JPEGBaseline8Bit
        )

        # Act & Assert
        with pytest.raises(UnscannablePerFrameGroupsException):
            reader.read_positions()

    def test_raises_for_implicit_vr(self):
        # Arrange
        reader = create_reader(
            [create_frame(1, 1)], transfer_syntax=ImplicitVRLittleEndian
        )

        # Act & Assert
        with pytest.raises(UnscannablePerFrameGroupsException):
            reader.read_positions()

    @pytest.mark.parametrize("frame_count", [0, -1])
    def test_raises_without_frames(self, frame_count: int):
        # Arrange
        reader = create_reader([create_frame(1, 1)], frame_count=frame_count)

        # Act & Assert
        with pytest.raises(UnscannablePerFrameGroupsException):
            reader.read_positions()

    @pytest.mark.parametrize(
        ["transfer_syntax", "expected"],
        [
            (JPEGBaseline8Bit, True),
            (ExplicitVRLittleEndian, True),
            (ImplicitVRLittleEndian, False),
        ],
    )
    def test_is_scannable(self, transfer_syntax: UID, expected: bool):
        # Arrange

        # Act
        is_scannable = PerFrameFunctionalGroupsReader.is_scannable(transfer_syntax)

        # Assert
        assert is_scannable == expected

    @pytest.mark.parametrize("undefined_length", [True, False])
    @pytest.mark.parametrize("chunk_size", [16, 64, 256])
    def test_read_positions_across_chunk_boundaries(
        self, chunk_size: int, undefined_length: bool
    ):
        """A hit split across two reads has to be found exactly once."""
        # Arrange
        frames = [create_frame(column=index + 1, row=1) for index in range(20)]
        reader = create_reader(
            frames, undefined_length=undefined_length, chunk_size=chunk_size
        )

        # Act
        positions = reader.read_positions()

        # Assert
        assert list(positions.columns) == [index + 1 for index in range(20)]

    @pytest.mark.parametrize("undefined_length", [True, False])
    def test_read_positions_of_sequence_longer_than_the_overlap(
        self, undefined_length: bool
    ):
        """Buffered bytes are dropped once they are behind the overlap kept between
        reads, so a sequence longer than that is where a value can be taken twice or
        lost. Every frame still has to be found exactly once."""
        # Arrange
        frame_count = 1500
        frames = [
            create_frame(column=index + 1, row=index + 2, optical_path="1")
            for index in range(frame_count)
        ]
        reader = create_reader(
            frames, undefined_length=undefined_length, chunk_size=16 * 1024
        )

        # Act
        positions = reader.read_positions()

        # Assert
        assert list(positions.columns) == [index + 1 for index in range(frame_count)]
        assert list(positions.rows) == [index + 2 for index in range(frame_count)]
        assert positions.optical_path_identifiers is not None
        assert list(positions.optical_path_identifiers) == ["1"] * frame_count


@pytest.mark.unittest
class TestShortStringElement:
    """A text value is decoded with the character set the instance states, which is
    read before the sequence and so is known before the search starts."""

    @pytest.mark.parametrize(
        ["specific_character_set", "value", "expected"],
        [
            (None, b"0 ", "0"),  # the default repertoire
            ("ISO_IR 100", bytes([0xE9]), "é"),  # latin 1
            (["ISO_IR 192"], bytes([0xC3, 0xA9]), "é"),  # utf 8
        ],
    )
    def test_decodes_with_the_stated_character_set(
        self,
        specific_character_set: str | list[str] | None,
        value: bytes,
        expected: str,
    ):
        # Arrange
        element = ShortStringElement(
            OpticalPathIdentifierTag, convert_encodings(specific_character_set)
        )

        # Act
        values = element.decode_values([value])

        # Assert
        assert values.tolist() == [expected]

    def test_refuses_a_value_that_switches_character_set(self):
        # Arrange - an escape sequence needs more than the one codec values are read
        # with, so the sequence is left to be read into datasets instead.
        element = ShortStringElement(
            OpticalPathIdentifierTag, convert_encodings(["", "ISO 2022 IR 87"])
        )
        value = bytes([0x1B, 0x24, 0x42]) + b"abc"

        # Act & Assert
        with pytest.raises(ValueError):
            element.decode_values([value])
