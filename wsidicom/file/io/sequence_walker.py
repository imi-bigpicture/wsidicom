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

"""Seek past a sequence by following its framing, without parsing its content."""

import struct
from collections.abc import Callable
from typing import BinaryIO, ClassVar, Final

from pydicom.uid import UID

from wsidicom.file.io.constants import (
    DELIMITER_GROUP_NUMBER,
    ITEM_DELIMITER_ELEMENT_NUMBER,
    ITEM_ELEMENT_NUMBER,
    LONG_FORM_HEADER_SIZE,
    LONG_FORM_VALUE_REPRESENTATIONS,
    SEQUENCE_DELIMITER_ELEMENT_NUMBER,
    SEQUENCE_VALUE_REPRESENTATION,
    SHORT_FORM_HEADER_SIZE,
    TAG_AND_LENGTH_SIZE,
    TAG_AND_VALUE_REPRESENTATION_SIZE,
    TAG_SIZE,
    TAG_VALUE_REPRESENTATION_AND_RESERVED_SIZE,
    UNDEFINED_LENGTH,
)


class UnwalkableSequenceException(Exception):
    """Raised when the framing of a sequence cannot be followed.

    Either the file is malformed or the walk has lost track of the framing. In both
    cases the sequence should be read instead of stepped over.
    """


class SequenceWalker:
    """Seeks past a sequence without parsing its content.

    Elements with defined length are skipped. Elements with undefined length are
    either nested sequences or values encoded in fragments. Both end with a sequence
    delimiter, so fragments are walked separately to not mistake their delimiter
    for the end of the sequence.
    """

    BLOCK_SIZE: ClassVar[int] = 8 * 1024
    """Number of bytes read at a time, to avoid a read per element header."""

    def __init__(self, stream: BinaryIO, transfer_syntax: UID):
        """Create a walker over `stream`.

        Parameters
        ----------
        stream: BinaryIO
            Stream to walk. Walking moves the stream position.
        transfer_syntax: UID
            Transfer syntax the stream is written in.
        """
        self._stream: Final = stream
        self._is_implicit_value_representation: Final = transfer_syntax.is_implicit_VR
        byte_order = "<" if transfer_syntax.is_little_endian else ">"
        # Annotated, as unpack_from is typed to return a tuple of Any.
        self._unpack_tag: Final[Callable[[bytes], tuple[int, int]]] = struct.Struct(
            f"{byte_order}HH"
        ).unpack_from
        self._unpack_short_length: Final[Callable[[bytes, int], tuple[int]]] = (
            struct.Struct(f"{byte_order}H").unpack_from
        )
        self._unpack_long_length: Final[Callable[[bytes, int], tuple[int]]] = (
            struct.Struct(f"{byte_order}I").unpack_from
        )
        self._block = b""
        self._block_position = 0

    def seek_past_sequence(self, position: int) -> int:
        """Seek past the sequence whose items start at `position`.

        Follows the framing to the end of the sequence without reading any values,
        and leaves the stream there.

        Parameters
        ----------
        position: int
            Offset of the first item of the sequence, after the sequence header.

        Returns
        -------
        int
            Offset just past the sequence delimiter.

        Raises
        ------
        UnwalkableSequenceException
            If the framing is malformed or the file ends before the sequence.
        """
        try:
            offset = self._seek_past_items(position)
        except RecursionError:
            # Nested deeper than the interpreter allows, which no real file is.
            raise UnwalkableSequenceException(
                f"Sequence at {position} is nested too deeply to step over."
            ) from None
        self._stream.seek(offset)
        return offset

    def _seek_past_items(self, position: int) -> int:
        """Return the offset past the sequence delimiter ending the items at
        `position`.

        Items of defined length are skipped, and those of undefined length walked
        into.

        Parameters
        ----------
        position: int
            Offset of the first item, or of the sequence delimiter if there is none.

        Returns
        -------
        int
            Offset just past the sequence delimiter.

        Raises
        ------
        UnwalkableSequenceException
            If something other than an item or a sequence delimiter is found.
        """
        offset = position
        while True:
            header = self._read_header(offset)
            group_number, element_number = self._unpack_tag(header)
            if group_number != DELIMITER_GROUP_NUMBER or element_number not in (
                ITEM_ELEMENT_NUMBER,
                SEQUENCE_DELIMITER_ELEMENT_NUMBER,
            ):
                raise UnwalkableSequenceException(
                    f"Found ({group_number:04X},{element_number:04X}) at {offset}, "
                    f"where an item or the end of a sequence belongs."
                )
            offset += TAG_AND_LENGTH_SIZE
            if element_number == SEQUENCE_DELIMITER_ELEMENT_NUMBER:
                return offset
            (length,) = self._unpack_long_length(header, TAG_SIZE)
            if length != UNDEFINED_LENGTH:
                offset += length
            else:
                offset = self._seek_past_attributes(offset)

    def _seek_past_attributes(self, position: int) -> int:
        """Return the offset past the item delimiter ending the attributes at
        `position`.

        Values of defined length are skipped, nested sequences walked and values
        encoded in fragments walked fragment by fragment.

        Parameters
        ----------
        position: int
            Offset of the first attribute of an undefined length item, or of the
            item delimiter if it has none.

        Returns
        -------
        int
            Offset just past the item delimiter.

        Raises
        ------
        UnwalkableSequenceException
            If an item or a sequence delimiter is found, or an attribute cannot be
            read.
        """
        offset = position
        while True:
            header = self._read_header(offset)
            group_number, element_number = self._unpack_tag(header)
            if group_number == DELIMITER_GROUP_NUMBER:
                if element_number != ITEM_DELIMITER_ELEMENT_NUMBER:
                    raise UnwalkableSequenceException(
                        f"Found ({group_number:04X},{element_number:04X}) at "
                        f"{offset}, where an attribute or the end of an item belongs."
                    )
                return offset + TAG_AND_LENGTH_SIZE
            length, offset, is_sequence = self._unpack_element_header(header, offset)
            if length != UNDEFINED_LENGTH:
                offset += length
            elif is_sequence:
                offset = self._seek_past_items(offset)
            else:
                offset = self._seek_past_fragments(offset)

    def _unpack_element_header(
        self, header: bytes, position: int
    ) -> tuple[int, int, bool]:
        """Unpack an element's length, value position and whether it is a sequence.

        Parameters
        ----------
        header: bytes
            Header of the element.
        position: int
            Offset the element starts at.

        Returns
        -------
        tuple[int, int, bool]
            Stated length, offset of the value, and whether the element is a
            sequence. With implicit VR an undefined length element is assumed to be
            a sequence, as only pixel data, which is not in a sequence, can
            otherwise have undefined length.

        Raises
        ------
        UnwalkableSequenceException
            If the header has no value representation where one is expected, or the
            file ends within the header.
        """
        if self._is_implicit_value_representation:
            (length,) = self._unpack_long_length(header, TAG_SIZE)
            return (
                length,
                position + TAG_AND_LENGTH_SIZE,
                length == UNDEFINED_LENGTH,
            )
        value_representation = header[TAG_SIZE:TAG_AND_VALUE_REPRESENTATION_SIZE]
        if not (value_representation.isalpha() and value_representation.isupper()):
            # E.g. an item written with implicit VR, which has its length here.
            raise UnwalkableSequenceException(
                f"Expected a value representation at {position + TAG_SIZE}, found "
                f"{value_representation!r}."
            )
        if value_representation in LONG_FORM_VALUE_REPRESENTATIONS:
            if len(header) < LONG_FORM_HEADER_SIZE:
                raise UnwalkableSequenceException(
                    f"File ended within the header at {position} while stepping "
                    f"over a sequence."
                )
            (length,) = self._unpack_long_length(
                header, TAG_VALUE_REPRESENTATION_AND_RESERVED_SIZE
            )
            return (
                length,
                position + LONG_FORM_HEADER_SIZE,
                value_representation == SEQUENCE_VALUE_REPRESENTATION,
            )
        (length,) = self._unpack_short_length(header, TAG_AND_VALUE_REPRESENTATION_SIZE)
        return length, position + SHORT_FORM_HEADER_SIZE, False

    def _seek_past_fragments(self, position: int) -> int:
        """Return the offset past the value encoded in fragments at `position`.

        Fragments are items holding bytes rather than datasets, so they do not nest
        and the first sequence delimiter ends the value.

        Parameters
        ----------
        position: int
            Offset of the first fragment.

        Returns
        -------
        int
            Offset just past the sequence delimiter ending the value.

        Raises
        ------
        UnwalkableSequenceException
            If an element other than a fragment or a sequence delimiter is found,
            or a fragment has undefined length.
        """
        offset = position
        while True:
            header = self._read_header(offset)
            group_number, element_number = self._unpack_tag(header)
            (length,) = self._unpack_long_length(header, TAG_SIZE)
            offset += TAG_AND_LENGTH_SIZE
            if group_number != DELIMITER_GROUP_NUMBER or element_number not in (
                ITEM_ELEMENT_NUMBER,
                SEQUENCE_DELIMITER_ELEMENT_NUMBER,
            ):
                # Anything else means the framing is malformed.
                raise UnwalkableSequenceException(
                    f"Expected a fragment or a delimiter at "
                    f"{offset - TAG_AND_LENGTH_SIZE} while stepping over a value "
                    f"sent in fragments, found "
                    f"({group_number:04X},{element_number:04X}).",
                )
            if element_number == SEQUENCE_DELIMITER_ELEMENT_NUMBER:
                return offset
            if length == UNDEFINED_LENGTH:
                # Not a fragment, e.g. an item of an undefined length UN element.
                raise UnwalkableSequenceException(
                    f"Found an item of undefined length at "
                    f"{offset - TAG_AND_LENGTH_SIZE}, where a fragment belongs."
                )
            offset += length

    def _read_header(self, position: int) -> bytes:
        """Read the header at `position`, reading a new block if needed.

        Parameters
        ----------
        position: int
            Offset to read from.

        Returns
        -------
        bytes
            `LONG_FORM_HEADER_SIZE` bytes from `position`, or at least
            `TAG_AND_LENGTH_SIZE` bytes at the end of the file.

        Raises
        ------
        UnwalkableSequenceException
            If fewer than `TAG_AND_LENGTH_SIZE` bytes are left in the file.
        """
        position_in_block = position - self._block_position
        outside_block = (
            position_in_block < 0
            or position_in_block + LONG_FORM_HEADER_SIZE > len(self._block)
        )
        if outside_block:
            self._block = self._read_block(position)
            self._block_position = position
            position_in_block = 0
            if len(self._block) < TAG_AND_LENGTH_SIZE:
                raise UnwalkableSequenceException(
                    f"File ended at {position} while stepping over a sequence."
                )
        return self._block[
            position_in_block : position_in_block + LONG_FORM_HEADER_SIZE
        ]

    def _read_block(self, position: int) -> bytes:
        """Read a block from `position`.

        Reads again if a read returns fewer bytes than asked for, which can happen
        for non-local files.

        Parameters
        ----------
        position: int
            Offset to read from.

        Returns
        -------
        bytes
            `BLOCK_SIZE` bytes, or fewer at the end of the file.
        """
        self._stream.seek(position)
        block = self._stream.read(self.BLOCK_SIZE)
        while len(block) < self.BLOCK_SIZE:
            further_bytes = self._stream.read(self.BLOCK_SIZE - len(block))
            if not further_bytes:
                break
            block += further_bytes
        return block
