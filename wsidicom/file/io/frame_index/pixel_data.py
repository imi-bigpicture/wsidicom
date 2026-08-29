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

"""Frame index for empty BOT, parsing the positions from the pixel data."""

import struct
from typing import ClassVar

import numpy as np
from pydicom.tag import ItemTag, SequenceDelimiterTag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.basic import EmptyBasicTableOffsetException
from wsidicom.file.io.frame_index.encapsulated_pixel_data import (
    EncapsulatedPixelDataFrameIndexParser,
)
from wsidicom.file.io.frame_index.frame_index import FrameIndex
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class PixelDataFrameIndexParser(EncapsulatedPixelDataFrameIndexParser):
    """Frame index parsed from reading the sequence of pixel data delimiters."""

    ITEM_TAG: ClassVar[bytes] = struct.pack("<HH", ItemTag.group, ItemTag.element)
    """The item tag introducing a frame."""

    BUFFER_BYTES: ClassVar[int] = 4 * 1024 * 1024
    """Bytes to read at a time when reading through the sequence of frames.

    Measured as the best of sizes from 8 KiB to 64 MiB, on slides whose frames differ
    in size by more than ten times, and it matters most when the file is not already
    in the page cache."""

    DELIMITER_TAG: ClassVar[bytes] = struct.pack(
        "<HH", SequenceDelimiterTag.group, SequenceDelimiterTag.element
    )
    """The sequence delimiter tag ending the pixel data."""

    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.EMPTY

    def _get_index(self) -> FrameIndex:
        """Get frame positions and length from the sequence of frames, which ends with
        a tag that is not the item tag.

        Each frame contains:
        item tag (4 bytes)
        item length (4 bytes)
        item data (item length)

        Returns
        -------
        FrameIndex
            Position and length of every frame.
        """
        lengths: list[int] = []
        with self._file.buffered(self.BUFFER_BYTES) as stream:
            stream.seek(self._pixels_start)
            while True:
                header = stream.read(self.HEADER_BYTES)
                if header[:4] != self.ITEM_TAG:
                    break
                length = int.from_bytes(header[4:], "little")
                if length == 0 or length % 2:
                    raise WsiDicomFileError(str(self._file), "Invalid frame length")
                lengths.append(length)
                stream.seek(length, 1)
            if header[:4] != self.DELIMITER_TAG:
                raise WsiDicomFileError(str(self._file), "No sequence delimiter tag")
        return self._create_index(lengths)

    def _create_index(self, lengths: list[int]) -> FrameIndex:
        """Return the index of frames of the given lengths, one after the other.

        The position of a frame is where the frame before it ended, so the positions
        are the lengths added up. The lengths are kept as the array that is made to add
        them up, which holds a quarter of what the numbers they came from hold.

        Parameters
        ----------
        lengths: list[int]
            Length of every frame, in the order they are in the pixel data.

        Returns
        -------
        FrameIndex
            Position and length of every frame.
        """
        frame_lengths = np.asarray(lengths, dtype=np.int64)
        header_bytes = self.HEADER_BYTES
        sizes = frame_lengths + header_bytes
        positions = self._pixels_start + header_bytes + np.cumsum(sizes) - sizes
        return FrameIndex(positions, frame_lengths)

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is not None:
            raise EmptyBasicTableOffsetException()
        return self._file.tell()
