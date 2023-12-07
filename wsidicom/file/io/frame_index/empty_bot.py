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

from typing import List, Tuple

from pydicom.tag import ItemTag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.bot import EmptyBotException
from wsidicom.file.io.frame_index.encapsulated_pixel_data import EncapsulatedPixelData
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class EmptyBot(EncapsulatedPixelData):
    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.EMPTY

    def _get_index(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from sequence of frames that ends
        with a tag not equal to ItemTag.

        Each frame contains:
        item tag (4 bytes)
        item length (4 bytes)
        item data (item length)
        The position of item data and the item length is stored.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        self._file.seek(self._pixels_start)
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        positions: List[Tuple[int, int]] = []
        frame_position = self._file.tell()
        # Read items until sequence delimiter
        while self._file.read_le_tag() == ItemTag:
            # Read item length
            length: int = self._file.read_leUL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(str(self._file), "Invalid frame length")
            positions.append((frame_position + TAG_BYTES + LENGTH_BYTES, length))
            # Jump to end of frame
            self._file.seek(length, 1)
            frame_position = self._file.tell()
        self._file.read_sequence_delimiter()
        return positions

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is not None:
            raise EmptyBotException()
        return self._file.tell()
