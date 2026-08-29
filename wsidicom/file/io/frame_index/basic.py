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

"""Frame index for BOT, parsing the positions from the BOT."""

import numpy as np
from numpy.typing import NDArray
from pydicom.tag import ItemTag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.frame_index import FrameIndex
from wsidicom.file.io.frame_index.offset_table import OffsetTableFrameIndexParser
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class EmptyBasicTableOffsetException(Exception):
    """Exception raised when BOT was empty."""


class BasicOffsetTableFrameIndexParser(OffsetTableFrameIndexParser):
    """Frame index parser for a basic offset table (BOT).

    A basic table states where every frame begins but not how long it is, so a length
    is the distance to the frame after it.
    """

    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.BASIC

    @property
    def dtype(self) -> str:
        return "<u4"

    def _get_index(self) -> FrameIndex:
        """Get frame positions and lengths from the basic offset table."""
        self._validate_pixel_data_start()
        table = self._read_table()
        offsets = self._parse_offsets(table)
        lengths = self._derive_lengths(offsets, self._pixels_start)
        return self._build_index(offsets, lengths, self._pixels_start)

    def _derive_lengths(
        self, offsets: NDArray[np.int64], pixels_start: int
    ) -> NDArray[np.int64]:
        """Return every frame length as the distance to the frame after it.

        A basic offset table states no lengths, so a frame reaches to the next one less
        the item header, and the last frame, which no next offset covers, has its length
        read from the pixel data.
        """
        lengths = np.empty(len(offsets), dtype=np.int64)
        lengths[:-1] = np.diff(offsets) - self.HEADER_BYTES
        lengths[-1] = self._read_last_frame_length(pixels_start, int(offsets[-1]))
        return lengths

    def _read_last_frame_length(self, pixels_start: int, offset: int) -> int:
        """Return the length of the last frame, which no next offset gives.

        Parameters
        ----------
        pixels_start: int
            Position of first frame item in pixel data.
        offset: int
            Offset of the last frame, relative to the first frame in the pixel data.

        Returns
        -------
        int
            Length of the last frame.
        """
        self._file.seek(pixels_start + offset)
        if self._file.read_tag() != ItemTag:
            raise WsiDicomFileError(str(self._file), "Expected ItemTag in PixelData")
        return self._file.read_UL()

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is None:
            raise EmptyBasicTableOffsetException()
        self._file.seek(bot_length, 1)
        return self._file.tell()

    def _read_table(self) -> bytes:
        """Read the basic offset table (BOT).

        Returns
        -------
        bytes
            BOT in bytes.
        """
        bot_length = self._read_bot_length()
        if bot_length is None:
            raise EmptyBasicTableOffsetException()
        return self._file.read(bot_length, need_exact_length=True)
