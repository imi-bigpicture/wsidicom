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

"""Frame index for EOT, parsing the positions from the EOT."""

import logging
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydicom.tag import Tag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.frame_index import FrameIndex
from wsidicom.file.io.frame_index.offset_table import OffsetTableFrameIndexParser
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType
from wsidicom.tags import ExtendedOffsetTableLengthsTag, ExtendedOffsetTableTag

logger = logging.getLogger(__name__)


class ExtendedOffsetFrameIndexParser(OffsetTableFrameIndexParser):
    """Frame index parser for an extended offset table (EOT).

    An extended table is accompanied by a second table stating the length of every
    frame, so unlike a basic table nothing has to be derived from the distance between
    two frames.
    """

    LENGTH_DTYPE: ClassVar[str] = "<u8"
    """The data type of an item in the lengths table, which the standard fixes at 64
    bits whatever the offsets are."""

    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.EXTENDED

    @property
    def dtype(self) -> str:
        return "<u8"

    def _get_index(self) -> FrameIndex:
        offset_table, length_table = self._read_tables()
        offsets = self._parse_offsets(offset_table)
        lengths = self._parse_lengths(length_table, len(offsets))
        return self._build_index(offsets, lengths, self._pixels_start)

    def _get_pixels_start(self) -> int:
        eot_tag = Tag(self._file.read_tag())
        if eot_tag != ExtendedOffsetTableTag:
            raise WsiDicomFileError(str(self._file), "Expected EOT tag")
        self._file.read_tag_vr()
        offset_table_length = self._read_offset_table_length()
        self._file.seek(offset_table_length, 1)
        self._skip_length_table()
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is not None:
            logger.warning(
                "BOT table was not empty in file with EOT table. "
                "The BOT table will be ignored."
            )
            self._file.seek(bot_length, 1)
        return self._file.tell()

    def _read_tables(self) -> tuple[bytes, bytes]:
        """Read extended table offset (EOT) and EOT lengths. Filepointer should be
        positionend to extended offset table.

        Returns
        -------
        tuple[bytes, bytes]
            EOT and EOT lengths in bytes.
        """
        eot_tag = Tag(self._file.read_tag())
        if eot_tag != ExtendedOffsetTableTag:
            raise WsiDicomFileError(
                str(self._file),
                f"Expected Extended offset table tag, found {eot_tag}",
            )
        self._file.read_tag_vr()
        offset_table_length = self._read_offset_table_length()
        offset_table = self._file.read(offset_table_length)
        return offset_table, self._read_length_table()

    def _read_offset_table_length(self) -> int:
        """Read the length in bytes of the extended offset table.

        Returns
        -------
        int
            Length in bytes of the extended offset table.
        """
        length = self._file.read_tag_length(True)
        self._validate_table_length(
            length, self.bytes_per_item, "Extended offset table"
        )
        return length

    def _parse_lengths(self, table: bytes, frame_count: int) -> NDArray[np.int64]:
        """Parse the length of every frame out of the extended offset table lengths.

        An extended offset table is accompanied by the length of every frame, so unlike
        a basic table nothing has to be derived from the distance between two frames.

        Parameters
        ----------
        table: bytes
            Extended offset table lengths as bytes.
        frame_count: int
            Number of frames the extended offset table holds an offset for.

        Returns
        -------
        NDArray[np.int64]
            Length of every frame.
        """
        lengths = np.frombuffer(table, dtype=self.LENGTH_DTYPE).astype(np.int64)
        if len(lengths) != frame_count:
            raise WsiDicomFileError(
                str(self._file),
                f"Extended offset table holds {frame_count} offsets but extended "
                f"offset table lengths holds {len(lengths)} lengths.",
            )
        return lengths

    def _read_length_table_length(self) -> int:
        """Read the tag of the extended offset table lengths and return its length.

        Leaves the file positioned at the first byte of the lengths.

        Returns
        -------
        int
            Length in bytes of the extended offset table lengths.
        """
        lengths_tag = self._file.read_tag()
        if lengths_tag != ExtendedOffsetTableLengthsTag:
            raise WsiDicomFileError(
                str(self._file),
                "Expected Extended offset table lengths tag after reading "
                f"Extended offset table, found {lengths_tag}",
            )
        self._file.read_tag_vr()
        return self._file.read_tag_length(True)

    def _validate_table_length(
        self, length: int, bytes_per_item: int, table: str
    ) -> None:
        """Raise unless a table is present and holds whole items.

        Parameters
        ----------
        length: int
            Length in bytes of the table.
        bytes_per_item: int
            Bytes of one item of the table.
        table: str
            Name of the table, for the error.
        """
        if length == 0 or length % bytes_per_item:
            raise WsiDicomFileError(
                str(self._file),
                f"{table} should be a non-zero multiple of {bytes_per_item} bytes, "
                f"got {length}.",
            )

    def _skip_length_table(self) -> None:
        """Skip over the extended offset table lengths."""
        self._file.seek(self._read_length_table_length(), 1)

    def _read_length_table(self) -> bytes:
        """Read the extended offset table lengths.

        Returns
        -------
        bytes
            Extended offset table lengths in bytes.
        """
        length = self._read_length_table_length()
        self._validate_table_length(
            length,
            np.dtype(self.LENGTH_DTYPE).itemsize,
            "Extended offset table lengths",
        )
        return self._file.read(length)
