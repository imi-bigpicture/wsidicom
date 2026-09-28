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

import struct
from collections.abc import Callable
from io import BytesIO

import pytest
from pydicom import Dataset
from pydicom.filereader import read_dataset as read_elements
from pydicom.sequence import Sequence
from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID, ExplicitVRLittleEndian, ImplicitVRLittleEndian

from wsidicom.file.io.sequence_walker import (
    SequenceWalker,
    UnwalkableSequenceException,
)

SPECIMEN_DESCRIPTION = Tag(0x0040, 0x0560)
PRIVATE_SEQUENCE = Tag(0x7FDF, 0x1001)
UNDEFINED = b"\xff\xff\xff\xff"
ITEM = b"\xfe\xff\x00\xe0"
ITEM_END = b"\xfe\xff\x0d\xe0" + b"\x00\x00\x00\x00"
SEQUENCE_END = b"\xfe\xff\xdd\xe0" + b"\x00\x00\x00\x00"


class Written:
    """Explicit VR little endian elements as bytes.

    Written by hand to produce framing pydicom does not write, such as fragments in
    a sequence and fragments whose content looks like a delimiter.
    """

    @staticmethod
    def tag(tag: BaseTag) -> bytes:
        return struct.pack("<HH", tag.group, tag.element)

    @classmethod
    def short(cls, tag: BaseTag, value_representation: bytes, value: bytes) -> bytes:
        """An element with a short form header."""
        return (
            cls.tag(tag) + value_representation + struct.pack("<H", len(value)) + value
        )

    @classmethod
    def long(cls, tag: BaseTag, value_representation: bytes, value: bytes) -> bytes:
        """An element with a long form header."""
        return (
            cls.tag(tag)
            + value_representation
            + b"\x00\x00"
            + struct.pack("<I", len(value))
            + value
        )

    @classmethod
    def opens_sequence(cls, tag: BaseTag) -> bytes:
        """The header of an undefined length sequence."""
        return cls.tag(tag) + b"SQ" + b"\x00\x00" + UNDEFINED

    @classmethod
    def item(cls, content: bytes) -> bytes:
        """A defined length item."""
        return ITEM + struct.pack("<I", len(content)) + content

    @classmethod
    def open_item(cls, content: bytes) -> bytes:
        """An undefined length item, ending with an item delimiter."""
        return ITEM + UNDEFINED + content + ITEM_END

    @classmethod
    def fragmented(cls, tag: BaseTag, fragments: list[bytes]) -> bytes:
        """A value encoded in fragments, ending with a sequence delimiter."""
        written = cls.tag(tag) + b"OB" + b"\x00\x00" + UNDEFINED
        for fragment in fragments:
            written += ITEM + struct.pack("<I", len(fragment)) + fragment
        return written + SEQUENCE_END


@pytest.fixture()
def walker_over() -> Callable[..., SequenceWalker]:
    """Make a walker over given bytes."""

    def make(
        written: bytes, transfer_syntax: UID = ExplicitVRLittleEndian
    ) -> SequenceWalker:
        return SequenceWalker(BytesIO(written), transfer_syntax)

    return make


@pytest.mark.unittest
class TestSequenceWalker:
    def test_walks_past_a_sequence_holding_one_item(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.item(Written.short(Tag(0x0040, 0x0551), b"LO", b"S1"))
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)

    def test_walks_past_a_sequence_holding_none(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        written = Written.opens_sequence(SPECIMEN_DESCRIPTION) + SEQUENCE_END
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)

    def test_walks_past_a_sequence_nested_in_an_item(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        nested = (
            Written.opens_sequence(Tag(0x0040, 0x059A))
            + Written.item(Written.short(Tag(0x0008, 0x0100), b"SH", b"C1"))
            + SEQUENCE_END
        )
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.open_item(nested)
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)

    def test_walks_past_a_value_sent_in_fragments(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # The fragment content looks like a sequence delimiter, which must not be
        # taken for the end of the sequence.
        written = (
            Written.opens_sequence(PRIVATE_SEQUENCE)
            + Written.open_item(
                Written.fragmented(Tag(0x7FE0, 0x0010), [SEQUENCE_END, b"\x01\x02"])
            )
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)

    def test_walks_past_a_long_form_value_that_is_not_a_sequence(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.item(Written.long(Tag(0x0028, 0x2000), b"OB", b"\xff" * 40))
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)

    def test_leaves_the_stream_past_the_sequence(self):
        # Arrange
        # An element after the sequence, so that the end of the sequence is not the
        # end of the file, where the stream is left after reading a block.
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.item(Written.short(Tag(0x0040, 0x0551), b"LO", b"S1"))
            + SEQUENCE_END
            + Written.short(Tag(0x0048, 0x0006), b"UL", b"\x00\x10\x00\x00")
        )
        stream = BytesIO(written)
        walker = SequenceWalker(stream, ExplicitVRLittleEndian)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert stream.tell() == end

    def test_lands_where_the_parser_lands(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        item = Dataset()
        item.SpecimenIdentifier = "S1"
        item.SpecimenUID = "1.2.3.4"
        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = Sequence([item, item])
        dataset[SPECIMEN_DESCRIPTION].is_undefined_length = True
        for held in dataset.SpecimenDescriptionSequence:
            held.is_undefined_length = True
        dataset.TotalPixelMatrixColumns = 4096
        buffer = BytesIO()
        dataset.save_as(buffer, implicit_vr=False, little_endian=True)
        written = buffer.getvalue()
        start = written.index(Written.tag(SPECIMEN_DESCRIPTION))
        stream = BytesIO(written)
        stream.seek(start)
        read_elements(
            stream,
            False,
            True,
            stop_when=lambda tag, vr, length: tag > SPECIMEN_DESCRIPTION,
        )
        parser_landed = stream.tell()
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(start + 12)

        # Assert
        assert end == parser_landed

    def test_walks_past_a_sequence_stating_no_value_representation(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # With implicit VR, lengths are four bytes and an undefined length element
        # is a sequence.
        written = (
            Written.tag(SPECIMEN_DESCRIPTION)
            + UNDEFINED
            + Written.item(
                Written.tag(Tag(0x0040, 0x0551)) + struct.pack("<I", 2) + b"S1"
            )
            + SEQUENCE_END
        )
        walker = walker_over(written, ImplicitVRLittleEndian)

        # Act
        end = walker.seek_past_sequence(8)

        # Assert
        assert end == len(written)

    def test_raises_on_an_unexpected_element_of_the_delimiter_group(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # Not an item or delimiter, though in the delimiter group.
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + b"\xfe\xff\x99\xe9"
            + b"\x00\x00\x00\x00"
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException):
            walker.seek_past_sequence(12)

    def test_raises_on_an_unexpected_element_among_fragments(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # An element where only fragments and a delimiter are allowed.
        fragments = (
            Written.tag(Tag(0x7FE0, 0x0010))
            + b"OB"
            + b"\x00\x00"
            + UNDEFINED
            + Written.short(Tag(0x0040, 0x0551), b"LO", b"S1")
            + SEQUENCE_END
        )
        written = (
            Written.opens_sequence(PRIVATE_SEQUENCE)
            + Written.open_item(fragments)
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException):
            walker.seek_past_sequence(12)

    def test_raises_on_an_item_written_with_implicit_vr(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # An implicit VR element in an explicit VR file, which pydicom detects per
        # item. Its length is where the VR belongs.
        implicit_element = (
            Written.tag(Tag(0x0040, 0x0551)) + struct.pack("<I", 2) + b"S1"
        )
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.open_item(implicit_element)
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException, match="value representation"):
            walker.seek_past_sequence(12)

    def test_raises_on_a_fragment_of_undefined_length(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # An undefined length UN element is a sequence with implicit VR content, and
        # its items can have undefined length, which a fragment never has.
        unknown = (
            Written.tag(PRIVATE_SEQUENCE)
            + b"UN"
            + b"\x00\x00"
            + UNDEFINED
            + ITEM
            + UNDEFINED
            + ITEM_END
            + SEQUENCE_END
        )
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + Written.open_item(unknown)
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException, match="undefined length"):
            walker.seek_past_sequence(12)

    @pytest.mark.parametrize(
        ["name", "body"],
        [
            (
                "attribute where an item belongs",
                Written.short(Tag(0x0040, 0x0551), b"LO", b"S1") + SEQUENCE_END,
            ),
            (
                "item where an attribute belongs",
                ITEM + UNDEFINED + Written.item(b"") + ITEM_END + SEQUENCE_END,
            ),
            ("end of item where an item belongs", ITEM_END + SEQUENCE_END),
            (
                "end of sequence where an attribute belongs",
                ITEM + UNDEFINED + SEQUENCE_END,
            ),
        ],
    )
    def test_raises_on_framing_that_does_not_alternate(
        self, walker_over: Callable[..., SequenceWalker], name: str, body: bytes
    ):
        """Items and attributes out of turn cannot be walked."""
        # Arrange
        walker = walker_over(Written.opens_sequence(SPECIMEN_DESCRIPTION) + body)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException):
            walker.seek_past_sequence(12)

    def test_raises_on_a_sequence_nested_too_deeply(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # Well formed, but nested deeper than the interpreter allows recursing.
        levels = 2000
        opens_level = ITEM + UNDEFINED + Written.opens_sequence(SPECIMEN_DESCRIPTION)
        closes_level = SEQUENCE_END + ITEM_END
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + opens_level * levels
            + closes_level * levels
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException, match="nested too deeply"):
            walker.seek_past_sequence(12)

    def test_raises_when_the_file_ends_within_the_sequence(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        written = Written.opens_sequence(SPECIMEN_DESCRIPTION) + ITEM
        walker = walker_over(written)

        # Act & Assert
        with pytest.raises(UnwalkableSequenceException):
            walker.seek_past_sequence(12)

    def test_walks_past_a_sequence_longer_than_a_block(
        self, walker_over: Callable[..., SequenceWalker]
    ):
        # Arrange
        # More items than one block holds, so the walker reads several blocks, and
        # of a size that does not divide the block, so that headers straddle them.
        item = Written.item(Written.short(Tag(0x0040, 0x0551), b"LO", b"S1" * 9))
        assert SequenceWalker.BLOCK_SIZE % len(item) != 0
        written = (
            Written.opens_sequence(SPECIMEN_DESCRIPTION)
            + item * (4 * SequenceWalker.BLOCK_SIZE // len(item))
            + SEQUENCE_END
        )
        walker = walker_over(written)

        # Act
        end = walker.seek_past_sequence(12)

        # Assert
        assert end == len(written)
