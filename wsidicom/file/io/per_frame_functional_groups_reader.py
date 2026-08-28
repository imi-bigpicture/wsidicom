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

"""Reader for tile positions in the Per Frame Functional Groups Sequence."""

import re
import struct
from abc import ABCMeta, abstractmethod
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydicom.charset import convert_encodings
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag
from pydicom.uid import UID

from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.tags import (
    ColumnPositionInTotalImagePixelMatrixTag,
    OpticalPathIdentifierTag,
    PerFrameFunctionalGroupsSequenceTag,
    RowPositionInTotalImagePixelMatrixTag,
    ZOffsetInSlideCoordinateSystemTag,
)


class UnscannablePerFrameGroupsException(Exception):
    """Exception raised when the values could not be found in the bytes of the
    sequence, leaving no way to get them but to build a dataset for every item."""

    pass


@dataclass(frozen=True)
class SearchBuffer:
    """A block of the sequence to search, and where it sits in the file."""

    buffer: bytes
    """The bytes to search."""

    file_position: int
    """Offset in the file that `buffer` starts at."""

    search_to: int
    """Offset in `buffer` to stop searching at, which need not be its end."""


@dataclass(frozen=True)
class FoundValues:
    """What searching for one element in one buffer found.

    Parameters
    ----------
    raw: list[bytes]
        The bytes of each value found, in the order they were found.
    found_to_file_position: int
        Offset in the file just past the last value found, or what was searched from
        if there were none.
    """

    raw: list[bytes]
    found_to_file_position: int


class SearchElement(metaclass=ABCMeta):
    """An element to find in the sequence by the bytes that introduce it.

    Those bytes are the tag and the value representation, and the length too when the
    value representation fixes it. They are the same in every item, so the element can
    be found by searching for them. The value always starts eight bytes in.

    Subclassed per value representation, into FixedLengthElement and
    VariableLengthElement.
    """

    VALUE_OFFSET: ClassVar[int] = 8
    """Bytes from the start of an element to the start of its value.

    A tag, a value representation and a length, of four, two and two bytes. That is the
    short form, which every value representation searched for is written in.
    """

    VR: ClassVar[bytes]
    """Value representation of the element."""

    def __init__(self, tag: BaseTag):
        """Create an element to search for.

        Parameters
        ----------
        tag: BaseTag
            Tag of the element.
        """
        self.tag = tag
        """Tag of the element."""

        self.pattern = struct.pack("<HH", tag.group, tag.element) + self.VR
        """Bytes that introduce the element and never vary."""

    @property
    @abstractmethod
    def max_bytes(self) -> int:
        """Most bytes the element can occupy, which sizes the carry between chunks.

        Returns
        -------
        int
            Bytes of the element with the longest value it could hold.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_values(
        self, search_buffer: SearchBuffer, found_to_file_position: int
    ) -> FoundValues:
        """Find the raw bytes of every value of this element in the buffer.

        How they are found depends on whether the value representation fixes the
        length, so each kind of element finds its own.

        Parameters
        ----------
        search_buffer: SearchBuffer
            Block of the sequence to search.
        found_to_file_position: int
            Offset in the file that values have already been found up to, which is
            where the search starts and what is returned if none are found.

        Returns
        -------
        FoundValues
            The bytes of every value found, and how far that got.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_values(self, raw: list[bytes]) -> NDArray:
        """Return the values decoded from the bytes found for them.

        Decodes every value at once rather than one at a time, as there is one of them
        for every frame of the image. They come back as an array of whatever type the
        value representation decodes to, in the order they were found, which is the
        order of the frames.

        Parameters
        ----------
        raw: list[bytes]
            The bytes of every value found, in the order they were found.

        Returns
        -------
        NDArray
            The values.
        """
        raise NotImplementedError()


class FixedLengthElement(SearchElement):
    """Element whose value representation fixes the length: its header never varies.

    The whole element is therefore the same size wherever it is, so a regular
    expression can take the value out of every one of them in a single pass.
    """

    LENGTH: ClassVar[int]
    """Length of the value, which the value representation fixes."""

    def __init__(self, tag: BaseTag):
        super().__init__(tag)
        self.pattern += struct.pack("<H", self.LENGTH)

        self.matcher = re.compile(
            re.escape(self.pattern) + b"(" + b"." * self.LENGTH + b")", re.DOTALL
        )
        """The element, capturing its value. Matches only where the whole element is
        within the searched range, which is what the search needs at a boundary."""

    @property
    def max_bytes(self) -> int:
        return len(self.pattern) + self.VALUE_OFFSET + self.LENGTH

    def find_values(
        self, search_buffer: SearchBuffer, found_to_file_position: int
    ) -> FoundValues:
        """Find the values of this element in the buffer.

        The whole element is the same size wherever it is, so one pass of the matcher
        finds every one of them and returns their values together. A match has to fit
        before where the search stops, so an element lying across the end of the buffer
        or of the sequence is left for the next chunk rather than half taken.

        Parameters
        ----------
        search_buffer: SearchBuffer
            Block of the sequence to search.
        found_to_file_position: int
            Offset in the file that values have already been found up to, which is
            where the search starts and what is returned if none are found.

        Returns
        -------
        FoundValues
            The bytes of every value found, and how far that got.
        """
        buffer = search_buffer.buffer
        search_to = search_buffer.search_to
        search_from = max(found_to_file_position - search_buffer.file_position, 0)
        found = self.matcher.findall(buffer, search_from, search_to)
        if len(found) == 0:
            return FoundValues([], found_to_file_position)
        # Where the last of them ended, which findall does not report. Bounded so that
        # it cannot land on an element running past where the search stops, which
        # findall would not have taken and which is left for the next chunk.
        whole = self.VALUE_OFFSET + self.LENGTH
        last = buffer.rfind(
            self.pattern, search_from, search_to - whole + len(self.pattern)
        )
        return FoundValues(found, search_buffer.file_position + last + whole)


class VariableLengthElement(SearchElement):
    """Element whose length varies. Each item states the length before its value."""

    MAX_VALUE_LENGTH: ClassVar[int]
    """Longest a value of this value representation can be."""

    @property
    def max_bytes(self) -> int:
        return len(self.pattern) + self.VALUE_OFFSET + self.MAX_VALUE_LENGTH

    def find_values(
        self, search_buffer: SearchBuffer, found_to_file_position: int
    ) -> FoundValues:
        """Find the values of this element.

        In one pass if they are all of the length the of the first item, and one at a
        time when they are not.

        Parameters
        ----------
        search_buffer: SearchBuffer
            Block of the sequence to search.
        found_to_file_position: int
            Offset in the file that values have already been found up to, which is
            where the search starts and what is returned if none are found.

        Returns
        -------
        FoundValues
            The bytes of every value found, and how far that got.
        """
        buffer = search_buffer.buffer
        search_from = max(found_to_file_position - search_buffer.file_position, 0)
        first = buffer.find(self.pattern, search_from, search_buffer.search_to)
        if first == -1:
            # No value of this element in what is left of the buffer, so there is no
            # length to go by and nothing to find.
            return FoundValues([], found_to_file_position)
        if first + self.VALUE_OFFSET > len(buffer):
            # The length has not been read yet, so it is left for the next chunk.
            return FoundValues([], found_to_file_position)
        assumed_length = struct.unpack_from("<H", buffer, first + len(self.pattern))[0]
        values = self._find_values_of_same_length(
            search_buffer, search_from, assumed_length
        )
        if values is not None:
            return values
        return self._find_values_one_at_a_time(
            search_buffer, search_from, found_to_file_position
        )

    def _find_values_of_same_length(
        self, search_buffer: SearchBuffer, search_from: int, assumed_length: int
    ) -> FoundValues | None:
        """Find the values if every one of them is `assumed_length` bytes.

        Counting what the pattern alone finds says whether the matcher, which only
        matches this one length, found all of them. The count is over the same range
        the matcher can match in, so that an element reaching past where the search
        stops is left for the next chunk by both of them rather than only by one.
        Getting that range wrong costs the pass and not the values: the two disagree,
        and the values are found one at a time instead.

        Parameters
        ----------
        search_buffer: SearchBuffer
            Block of the sequence to search.
        search_from: int
            Offset in the buffer to start searching at.
        assumed_length: int
            Length every value is assumed to be.

        Returns
        -------
        FoundValues | None
            The bytes of every value found and how far that got, or None if the values
            are not all of that length and have to be found one at a time.
        """
        buffer = search_buffer.buffer
        search_to = search_buffer.search_to
        whole = self.VALUE_OFFSET + assumed_length
        fits_to = search_to - whole + len(self.pattern)
        found = self._matcher_for(assumed_length).findall(
            buffer, search_from, search_to
        )
        if len(found) == 0:
            return None
        if len(found) != buffer.count(self.pattern, search_from, fits_to):
            return None
        last = buffer.rfind(self.pattern, search_from, fits_to)
        return FoundValues(found, search_buffer.file_position + last + whole)

    def _find_values_one_at_a_time(
        self, search_buffer: SearchBuffer, search_from: int, found_to_file_position: int
    ) -> FoundValues:
        """Find the values regardless of length, by reading the length each states.

        Parameters
        ----------
        search_buffer: SearchBuffer
            Block of the sequence to search.
        search_from: int
            Offset in the buffer to start searching at.
        found_to_file_position: int
            Offset in the file that values have already been found up to.

        Returns
        -------
        FoundValues
            The bytes of every value found, and how far that got.
        """
        buffer = search_buffer.buffer
        search_to = search_buffer.search_to
        value_offset = self.VALUE_OFFSET
        pattern = self.pattern
        raw: list[bytes] = []
        at = buffer.find(pattern, search_from, search_to)
        while at != -1:
            if at + value_offset > len(buffer):
                break  # the length has not been read yet
            # The element states its own length, in the two bytes before its value.
            length = struct.unpack_from("<H", buffer, at + len(pattern))[0]
            end = at + value_offset + length
            if end > search_to:
                break
            raw.append(buffer[at + value_offset : end])
            found_to_file_position = search_buffer.file_position + end
            at = buffer.find(pattern, end, search_to)
        return FoundValues(raw, found_to_file_position)

    def _matcher_for(self, length: int) -> re.Pattern[bytes]:
        """Return a matcher for the element with a value of `length` bytes.

        Not kept, as re compiles a pattern once and hands back the same matcher for it
        after that.

        Parameters
        ----------
        length: int
            Length of the value.

        Returns
        -------
        re.Pattern[bytes]
            The element of that length, capturing its value.
        """
        return re.compile(
            re.escape(self.pattern + struct.pack("<H", length))
            + b"("
            + b"." * length
            + b")",
            re.DOTALL,
        )


class SignedLongElement(FixedLengthElement):
    """Element of value representation SL, a 32 bit signed integer."""

    VR = b"SL"
    LENGTH = 4

    def decode_values(self, raw: list[bytes]) -> NDArray[np.int64]:
        return np.frombuffer(b"".join(raw), dtype="<i4").astype(np.int64)


class DecimalStringElement(VariableLengthElement):
    """Element of value representation DS, a decimal written out as text."""

    VR = b"DS"
    MAX_VALUE_LENGTH = 16
    """A decimal string is at most 16 bytes, by PS3.5 Table 6.2-1."""

    def decode_values(self, raw: list[bytes]) -> NDArray[np.float64]:
        return np.asarray([float(value) for value in raw], dtype=np.float64)


class ShortStringElement(VariableLengthElement):
    """Element of value representation SH, a short text.

    Decoded with the character set the instance states in Specific Character Set
    (0008,0005), which is read before the sequence and so is known by the time the
    search starts. It is turned into a Python codec once, leaving a value to cost no
    more than decoding its bytes.

    A value that switches character set within itself, which ISO 2022 allows and an
    escape byte marks, needs more than one codec to read. Rather than read it wrongly
    and disagree with what pydicom reads elsewhere in the dataset, the search gives up
    and leaves the sequence to be read into pydicom datasets.
    """

    VR = b"SH"
    MAX_VALUE_LENGTH = 16
    """A short string is at most 16 bytes, by PS3.5 Table 6.2-1."""

    ESCAPE: ClassVar[int] = 0x1B
    """Byte introducing a character set escape sequence."""

    def __init__(self, tag: BaseTag, encodings: Sequence[str]):
        """Create an element to search for.

        Parameters
        ----------
        tag: BaseTag
            Tag of the element.
        encodings: Sequence[str]
            Python codecs the instance states, converted from Specific Character Set.
            The first is the default repertoire and any others are code extensions.
        """
        super().__init__(tag)
        self.encoding = encodings[0]
        """Python codec the values are decoded with."""

        self.has_code_extensions = len(encodings) > 1
        """Whether the instance states a character set a value can escape into.

        Only then can a value switch character set within itself, so only then is
        there any reason to look for an escape byte.
        """

    def decode_values(self, raw: list[bytes]) -> NDArray[np.str_]:
        """Return the values, decoded with the character set the instance states.

        The padding a short text is written with is not part of a value.

        Raises
        ------
        ValueError
            If a value switches character set within itself, which one codec cannot
            read.
        """
        if self.has_code_extensions and any(self.ESCAPE in value for value in raw):
            raise ValueError(
                "A value switches character set within itself, which takes more than "
                "the one codec the values are decoded with."
            )
        return np.asarray(
            [value.decode(self.encoding).rstrip("\x00 ") for value in raw],
            dtype=np.str_,
        )


class PerFrameFunctionalGroupsReader:
    """Reads tile positions out of the Per Frame Functional Groups Sequence.

    A sparse image carries one item per frame in the Per Frame Functional Groups
    Sequence, and reading them with pydicom creates an object for every frame and for
    every sequence nested in it. We only need to reach four values per frame that a
    sparse tile index wants: the column and row position, the focal plane and the
    optical path. Read only these by searching the sequence for the pattern that
    introduces each of those values.

    What is searched for is one :class:`SearchElement` per value, which knows the
    bytes that introduce it and how to take its values out of a buffer. This reads the
    sequence a chunk at a time, hands each chunk to each element, and checks that as
    many values came back as the instance declares frames.

    Item lengths differ between frames, as the slide coordinates are `DS` and a decimal
    string is only padded to an even length, but a search does not care: a
    varying-length neighbour only moves where the next hit is found.

    What the search assumes
    -----------------------
    Searching is not parsing. It reads no structure, so it cannot see which item or
    nested sequence a value was in. Most of what that would tell it the standard
    settles anyway: the items are in frame order, so the nth value found belongs to the
    nth frame; a functional group macro in the per frame groups is in every one of
    them, so an element is stated by every frame or by none; and the sequences the
    values are found in hold a single item each, so an element is stated once per
    frame. The transfer syntax is checked outright, by :func:`is_scannable`, before
    anything is read.

    What is left is assumed rather than known or checked:

    - The tags within an item are in groups below 5200. That is what tells the
      delimiter ending the sequence from the ones ending the sequences nested in the
      items, and a private tag in a higher group would end the search early. Nothing in
      the standard forbids one.
    - The bytes that introduce an element appear nowhere but at the start of one, so
      every hit is a real element and not something inside another value.

    A file that breaks what the standard settles is refused rather than read wrongly,
    as the count of values then does not match the frame count.
    """

    CHUNK_SIZE: ClassVar[int] = 8 * 1024 * 1024
    """Bytes read at a time while searching."""

    SEQUENCE_DELIMITER: ClassVar[bytes] = struct.pack(
        "<HHI", SequenceDelimiterTag.group, SequenceDelimiterTag.element, 0
    )
    """The sequence delimiter, tag and zero length, ending a delimited sequence."""

    DELIMITER_CANDIDATE: ClassVar[re.Pattern[bytes]] = re.compile(
        re.escape(SEQUENCE_DELIMITER) + rb"[\s\S][\x52-\xff][\s\S][\s\S][A-Z][A-Z]"
    )
    """A delimiter that could be the end of the sequence, with what follows it.

    Every nested sequence in every item ends in the same delimiter, so most of them are
    not the one being looked for. The one that is has the element following the sequence
    after it, whose tag is ordered after the tag of the sequence and so is in a group of
    at least 5200, while the tags inside an item are all in far lower groups. Only a
    delimiter followed by a high group and by two letters naming a value representation
    is therefore worth looking at.
    """

    UNDEFINED_LENGTH: ClassVar[int] = 0xFFFFFFFF
    """Length a sequence states when it is delimited rather than of a stated length."""

    SEQUENCE_HEADER_BYTES: ClassVar[int] = 12
    """Bytes of tag, value representation and length introducing a sequence."""

    TAG_AND_VR_BYTES: ClassVar[int] = 6
    """Bytes of tag and value representation introducing an element, which is as much
    of the element after the sequence as is needed to recognise it."""

    def __init__(
        self,
        file: WsiDicomIO,
        sequence_file_position: int,
        frame_count: int,
        transfer_syntax: UID,
        specific_character_set: str | MutableSequence[str] | None = None,
        chunk_size: int | None = None,
    ):
        """Create a reader for the sequence at `sequence_file_position`.

        Parameters
        ----------
        file: WsiDicomIO
            File to read from.
        sequence_file_position: int
            Offset of the Per Frame Functional Groups Sequence tag.
        frame_count: int
            Number of frames the instance declares, used to check the result.
        transfer_syntax: UID
            Transfer syntax of the instance.
        specific_character_set: str | MutableSequence[str] | None = None
            Specific Character Set (0008,0005) of the instance, which text values are
            decoded with. The default repertoire if not given.
        chunk_size: int | None = None
            Bytes to read at a time, `CHUNK_SIZE` if not given.
        """
        self._file = file
        self._sequence_file_position = sequence_file_position
        self._frame_count = frame_count
        self._transfer_syntax = transfer_syntax
        self._chunk_size = chunk_size if chunk_size is not None else self.CHUNK_SIZE
        self._end_of_sequence: int | None = None
        encodings = convert_encodings(specific_character_set)
        self._search_elements: tuple[SearchElement, ...] = (
            SignedLongElement(ColumnPositionInTotalImagePixelMatrixTag),
            SignedLongElement(RowPositionInTotalImagePixelMatrixTag),
            DecimalStringElement(ZOffsetInSlideCoordinateSystemTag),
            ShortStringElement(OpticalPathIdentifierTag, encodings),
        )
        self._search_elements_by_tag = {
            element.tag: element for element in self._search_elements
        }

    @staticmethod
    def is_scannable(transfer_syntax: UID) -> bool:
        """Return True if a data set of this transfer syntax can be searched.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax of the instance.

        Returns
        -------
        bool
            True if the data set is explicit VR little endian and not deflated.
        """
        return (
            not transfer_syntax.is_implicit_VR
            and transfer_syntax.is_little_endian
            and not transfer_syntax.is_deflated
        )

    @property
    def end_of_sequence(self) -> int:
        """Offset of the first byte after the sequence, known once the positions have
        been read. Whatever element is ordered next starts there, which is usually but
        not necessarily the extended offset table or the pixel data.

        Returns
        -------
        int
            Offset in the file of the first byte after the sequence.

        Raises
        ------
        ValueError
            If the positions have not been read yet, as the end is only found by
            reading up to it.
        """
        if self._end_of_sequence is None:
            raise ValueError("Positions have not been read yet.")
        return self._end_of_sequence

    def read_positions(self) -> PerFrameGroupPositions:
        """Find the tile positions of all frames in the bytes of the sequence.

        Returns
        -------
        PerFrameGroupPositions
            Positions for every frame.

        Raises
        ------
        UnscannablePerFrameGroupsException
            If the values could not be found with confidence, in which case the caller
            has to build a dataset for every item instead.
        """
        if self._frame_count < 1 or not self.is_scannable(self._transfer_syntax):
            raise UnscannablePerFrameGroupsException(
                f"Cannot search a data set of transfer syntax {self._transfer_syntax} "
                f"for {self._frame_count} frames."
            )
        sequence_length = self._read_sequence_length()
        found_values, end_of_sequence = self._search(sequence_length)
        if end_of_sequence is None:
            raise UnscannablePerFrameGroupsException(
                "Sequence did not end in a delimiter followed by an element."
            )

        positions = PerFrameGroupPositions(
            columns=self._decode_required_values(
                ColumnPositionInTotalImagePixelMatrixTag, found_values
            ),
            rows=self._decode_required_values(
                RowPositionInTotalImagePixelMatrixTag, found_values
            ),
            z_offsets=self._decode_optional_values(
                ZOffsetInSlideCoordinateSystemTag, found_values
            ),
            optical_path_identifiers=self._decode_optional_values(
                OpticalPathIdentifierTag, found_values
            ),
        )
        self._end_of_sequence = end_of_sequence
        return positions

    def _decode_required_values(
        self, tag: BaseTag, found_values: dict[BaseTag, list[bytes]]
    ) -> NDArray:
        """Return the values found for an element every frame carries, decoded.

        Parameters
        ----------
        tag: BaseTag
            Tag of the element.
        found_values: dict[BaseTag, list[bytes]]
            The bytes of every value found, per element.

        Returns
        -------
        NDArray
            The values, one per frame.

        Raises
        ------
        UnscannablePerFrameGroupsException
            If there is not one value per frame, or if the bytes found do not decode.
        """
        raw = found_values[tag]
        if len(raw) != self._frame_count:
            # Which way it is wrong says what is wrong with the file, and only one of
            # the two is a frame going without.
            too_many = len(raw) > self._frame_count
            raise UnscannablePerFrameGroupsException(
                f"Found {len(raw)} values of {tag} for {self._frame_count} frames, so "
                + (
                    "the element is stated more than once by a frame."
                    if too_many
                    else "the element is not stated by every frame."
                )
            )
        element = self._search_elements_by_tag[tag]
        try:
            return element.decode_values(raw)
        except ValueError as exception:
            raise UnscannablePerFrameGroupsException(
                f"Could not decode the values found for {tag}: {exception}"
            ) from exception

    def _decode_optional_values(
        self, tag: BaseTag, found_values: dict[BaseTag, list[bytes]]
    ) -> NDArray | None:
        """Return the values found for an element frames need not carry, or None.

        Parameters
        ----------
        tag: BaseTag
            Tag of the element.
        found_values: dict[BaseTag, list[bytes]]
            The bytes of every value found, per element.

        Returns
        -------
        NDArray | None
            The values, or None if no frame carries the element.

        Raises
        ------
        UnscannablePerFrameGroupsException
            If only some frames carry the element, as the values cannot then be matched
            to frames by position.
        """
        if len(found_values[tag]) == 0:
            return None
        return self._decode_required_values(tag, found_values)

    def _search(
        self, sequence_length: int | None
    ) -> tuple[dict[BaseTag, list[bytes]], int | None]:
        """Search the sequence for every element, a chunk at a time.

        A sequence that states a length is read to that length, a delimited one until
        its delimiter is found. Nothing beyond the delimiter is searched, so reading
        ahead into the pixel data cannot contribute values.

        Parameters
        ----------
        sequence_length: int | None
            Length of the sequence, or None if it is delimited instead of stating one.

        Returns
        -------
        tuple[dict[BaseTag, list[bytes]], int | None]
            Bytes of the values found per element, and the offset just after the
            sequence, or None if the sequence did not end where it should have.
        """
        found_values: dict[BaseTag, list[bytes]] = {
            element.tag: [] for element in self._search_elements
        }
        # How far into the file each element's values have been found, so that the
        # bytes carried from one chunk to the next are not searched again.
        found_to_file_positions: dict[BaseTag, int] = {
            element.tag: 0 for element in self._search_elements
        }
        # Bytes carried from one chunk to the next, so that an element lying across the
        # boundary is whole in the next buffer. Long enough for the longest element,
        # and for a delimiter with the start of the element after it.
        max_carried_bytes = max(
            max(element.max_bytes for element in self._search_elements),
            len(self.SEQUENCE_DELIMITER) + self.TAG_AND_VR_BYTES,
        )
        # None while the sequence states no length, as there is then no telling how
        # much is left until the delimiter turns up.
        unread_bytes = sequence_length
        # The buffer starts at the first byte of the sequence content and moves through
        # it, so its position is where the offsets found in it are counted from.
        buffer_file_position = self._sequence_file_position + self.SEQUENCE_HEADER_BYTES
        end_of_sequence = (
            None if sequence_length is None else buffer_file_position + sequence_length
        )

        self._file.seek(buffer_file_position)
        buffer = b""
        while True:
            bytes_to_read = (
                self._chunk_size
                if unread_bytes is None
                else min(self._chunk_size, unread_bytes)
            )
            chunk = self._file.read(bytes_to_read) if bytes_to_read > 0 else b""
            if unread_bytes is not None:
                unread_bytes -= len(chunk)
            buffer = buffer + chunk

            # Values after the delimiter belong to whatever follows the sequence, so
            # the search stops there rather than at the end of the buffer.
            delimiter_buffer_position = (
                self._find_delimiter(buffer) if unread_bytes is None else None
            )
            search_to_buffer_position = (
                len(buffer)
                if delimiter_buffer_position is None
                else delimiter_buffer_position
            )
            search_buffer = SearchBuffer(
                buffer, buffer_file_position, search_to_buffer_position
            )
            for element in self._search_elements:
                # Each element carries on from where its own values were last found,
                # so that the bytes carried from the previous chunk are not searched
                # again.
                found = element.find_values(
                    search_buffer, found_to_file_positions[element.tag]
                )
                found_values[element.tag].extend(found.raw)
                found_to_file_positions[element.tag] = found.found_to_file_position

            if delimiter_buffer_position is not None:
                return (
                    found_values,
                    buffer_file_position
                    + delimiter_buffer_position
                    + len(self.SEQUENCE_DELIMITER),
                )
            if len(chunk) == 0:
                return found_values, end_of_sequence
            carried_bytes = min(max_carried_bytes, len(buffer))
            buffer_file_position += len(buffer) - carried_bytes
            buffer = buffer[len(buffer) - carried_bytes :]

    def _find_delimiter(self, buffer: bytes) -> int | None:
        """Return where the sequence delimiter starts in `buffer`, if it is there.

        These same bytes end every undefined-length sequence, and the per frame items
        hold nested ones, so the delimiter that matters is recognised by what follows
        it: the next top level element. That has a data element tag ordered after the
        sequence's own, followed by two letters naming the value representation. A
        candidate too close to the end of the buffer to judge is left for the next
        chunk, as it cannot match `DELIMITER_CANDIDATE` without the bytes after it.

        Parameters
        ----------
        buffer: bytes
            Buffer to search.

        Returns
        -------
        int | None
            Offset in the buffer of the delimiter, or None if it is not there.
        """
        delimiter_length = len(self.SEQUENCE_DELIMITER)
        for candidate in self.DELIMITER_CANDIDATE.finditer(buffer):
            if self._starts_top_level_element(candidate.group()[delimiter_length:]):
                return candidate.start()
        return None

    @staticmethod
    def _starts_top_level_element(header: bytes) -> bool:
        """Return True if these bytes start the element following the sequence.

        Parameters
        ----------
        header: bytes
            The `TAG_AND_VR_BYTES` bytes after a candidate delimiter.

        Returns
        -------
        bool
            True if they are a data element tag ordered after the sequence, followed by
            two letters naming a value representation.
        """
        group, element = struct.unpack_from("<HH", header)
        if (
            group == ItemTag.group
        ):  # an item or delimiter tag: still inside the sequence
            return False
        if ((group << 16) | element) <= int(PerFrameFunctionalGroupsSequenceTag):
            return False
        return header[4:5].isupper() and header[5:6].isupper()

    def _read_sequence_length(self) -> int | None:
        """Return the length the sequence states, or None if it states none.

        Read once per file rather than once per frame, so it reads the header the way
        the rest of the package does instead of unpacking the bytes itself.

        Returns
        -------
        int | None
            Length of the sequence, or None if it is delimited instead of stating one.

        Raises
        ------
        UnscannablePerFrameGroupsException
            If the element at the position is not the sequence.
        """
        self._file.seek(self._sequence_file_position)
        try:
            tag = self._file.read_tag()
            vr = self._file.read_tag_vr()
            length = self._file.read_tag_length(long=True)
        except EOFError as exception:
            raise UnscannablePerFrameGroupsException(
                f"Stream ended at {self._sequence_file_position}, where the Per Frame "
                "Functional Groups Sequence was expected."
            ) from exception
        if tag != PerFrameFunctionalGroupsSequenceTag or vr != b"SQ":
            raise UnscannablePerFrameGroupsException(
                f"Expected the Per Frame Functional Groups Sequence at "
                f"{self._sequence_file_position}, found {tag} {vr}."
            )
        return None if length == self.UNDEFINED_LENGTH else length
