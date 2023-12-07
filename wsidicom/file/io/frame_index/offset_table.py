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

"""Abstract class for FrameIndex that has an offset table (basic or extended)."""

from abc import abstractmethod
from struct import unpack
from typing import List, Tuple

from pydicom.tag import ItemTag
from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.encapsulated_pixel_data import EncapsulatedPixelData


class OffsetTable(EncapsulatedPixelData):
    @property
    @abstractmethod
    def bytes_per_item(self) -> int:
        """Return the number of bytes per item in the table."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return the mode used for unpacking the table."""
        raise NotImplementedError()

    def _parse_table(self, table: bytes, pixels_start: int) -> List[Tuple[int, int]]:
        """Parse table with offsets (BOT or EOT).

        Parameters
        ----------
        table: bytes
            BOT or EOT as bytes
        table_type: OffsetTableType
            Type of table, 'bot' or 'eot'.
        pixels_start: int
            Position of first frame item in pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            A list with frame positions and frame lengths.
        """
        if not self._file.is_little_endian:
            raise WsiDicomFileError(
                str(self._file), "Big endian not supported for BOT or EOT"
            )
        bytes_per_item = self.bytes_per_item
        mode = self.mode
        table_length = len(table)
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        positions: List[Tuple[int, int]] = []
        # Read through table to get offset and length for all but last item
        # All read offsets are for item tag of frame and relative to first
        # frame in pixel data.
        this_offset: int = unpack(mode, table[0:bytes_per_item])[0]
        if this_offset != 0:
            raise ValueError("First item in table should be at offset 0")
        for index in range(bytes_per_item, table_length, bytes_per_item):
            next_offset = unpack(mode, table[index : index + bytes_per_item])[0]
            offset = this_offset + TAG_BYTES + LENGTH_BYTES
            length = next_offset - offset
            if length <= 0 or length % 2:
                raise WsiDicomFileError(
                    str(self._file),
                    f"Invalid frame length {length} for frame {index // bytes_per_item}",
                )
            positions.append((pixels_start + offset, length))
            this_offset = next_offset

        # Go to last frame in pixel data and read the length of the frame
        self._file.seek(pixels_start + this_offset)
        if self._file.read_le_tag() != ItemTag:
            raise WsiDicomFileError(str(self._file), "Expected ItemTag in PixelData")
        length: int = self._file.read_leUL()
        if length <= 0 or length % 2:
            raise WsiDicomFileError(str(self._file), "Invalid frame length")
        offset = this_offset + TAG_BYTES + LENGTH_BYTES
        positions.append((pixels_start + offset, length))

        return positions
