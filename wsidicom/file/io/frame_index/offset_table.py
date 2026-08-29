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

import numpy as np
from numpy.typing import NDArray

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.encapsulated_pixel_data import (
    EncapsulatedPixelDataFrameIndexParser,
)
from wsidicom.file.io.frame_index.frame_index import FrameIndex


class OffsetTableFrameIndexParser(EncapsulatedPixelDataFrameIndexParser):
    """Frame index parser for pixel data introduced by an offset table."""

    @property
    @abstractmethod
    def dtype(self) -> str:
        """Return the data type of an item in the table."""
        raise NotImplementedError()

    @property
    def bytes_per_item(self) -> int:
        """Return the number of bytes per item in the table."""
        return np.dtype(self.dtype).itemsize

    def _parse_offsets(self, table: bytes) -> NDArray[np.int64]:
        """Parse the offset of every frame out of a table (BOT or EOT).

        The whole table is turned into offsets at once rather than a frame at a time.

        Parameters
        ----------
        table: bytes
            BOT or EOT as bytes

        Returns
        -------
        NDArray[np.int64]
            Offset of every frame, relative to the first frame in the pixel data.
        """
        if not self._file.is_little_endian:
            raise WsiDicomFileError(
                str(self._file), "Big endian not supported for BOT or EOT"
            )
        bytes_per_item = self.bytes_per_item
        table_length = len(table)
        if table_length == 0 or table_length % bytes_per_item:
            raise WsiDicomFileError(
                str(self._file),
                f"Expected offset table of a non-zero multiple of {bytes_per_item} "
                f"bytes, got {table_length}.",
            )
        offsets = np.frombuffer(table, dtype=self.dtype).astype(np.int64)
        if offsets[0] != 0:
            raise WsiDicomFileError(
                str(self._file), "First item in offset table should be at offset 0"
            )
        return offsets

    def _build_index(
        self,
        offsets: NDArray[np.int64],
        lengths: NDArray[np.int64],
        pixels_start: int,
    ) -> FrameIndex:
        """Return where every frame is in the file, from its offset and its length.

        Parameters
        ----------
        offsets: NDArray[np.int64]
            Offset of every frame, relative to the first frame in the pixel data.
        lengths: NDArray[np.int64]
            Length of every frame.
        pixels_start: int
            Position of first frame item in pixel data.

        Returns
        -------
        FrameIndex
            Position and length of every frame.
        """
        invalid: NDArray[np.bool_] = (lengths <= 0) | (lengths % 2 != 0)
        if invalid.any():
            frame = int(invalid.argmax())
            raise WsiDicomFileError(
                str(self._file),
                f"Invalid frame length {lengths[frame]} for frame {frame}",
            )
        return FrameIndex(pixels_start + offsets + self.HEADER_BYTES, lengths)
