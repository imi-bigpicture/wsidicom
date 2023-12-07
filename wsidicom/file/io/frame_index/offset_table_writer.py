#    Copyrigh 2023 SECTRA AB
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

"""Module for writing offset tables to file."""

from abc import ABCMeta, abstractmethod
from typing import Sequence

from pydicom.tag import ItemTag

from wsidicom.errors import WsiDicomBotOverflow
from wsidicom.tags import ExtendedOffsetTableLengthsTag, ExtendedOffsetTableTag
from wsidicom.file.io.wsidicom_io import WsiDicomIO


class OffsetTableWriter(metaclass=ABCMeta):
    """Abstract class for writing offset tables to file."""

    def __init__(self, file: WsiDicomIO) -> None:
        """Initiate offset table writer.

        Parameters
        ----------
        file: WsiDicomIO
            File to write to.

        """
        self._file = file

    @abstractmethod
    def reserve(self, number_of_frames: int):
        """Reserve space in file for offset table.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for.

        """
        raise NotImplementedError()

    @abstractmethod
    def write(
        self,
        pixel_data_start: int,
        frame_positions: Sequence[int],
        last_frame_end: int,
    ) -> None:
        """Write offset table to file.

        Parameters
        ----------
        pixel_data_start: int
            File position of pixel data start
        frame_positions: Sequence[int]
            List of file positions for frames, relative to file start
        last_frame_end: int
            Position of last frame end.

        """
        raise NotImplementedError()


class BotWriter(OffsetTableWriter):
    def reserve(self, number_of_frames: int):
        self._table_start = self._file.tell()
        BYTES_PER_ITEM = 4
        tag_lengths = BYTES_PER_ITEM * number_of_frames
        self._file.write_tag(ItemTag)
        self._file.write_leUL(tag_lengths)
        for _ in range(number_of_frames):
            self._file.write_leUL(0)

    def write(
        self,
        pixel_data_start: int,
        frame_positions: Sequence[int],
        last_frame_end: int,
    ) -> None:
        BYTES_PER_ITEM = 4
        # Check that last BOT entry is not over 2^32 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**32 - 1:
            raise WsiDicomBotOverflow(
                "Image data exceeds 2^32 - 1 bytes "
                "An extended offset table should be used"
            )

        self._file.seek(self._table_start)  # Go to first BOT entry
        self._file.check_tag_and_length(
            ItemTag, BYTES_PER_ITEM * len(frame_positions), False, True
        )

        for frame_position in frame_positions:  # Write BOT
            self._file.write_leUL(frame_position - pixel_data_start)


class EotWriter(OffsetTableWriter):
    def reserve(self, number_of_frames: int):
        self._table_start = self._file.tell()
        BYTES_PER_ITEM = 8
        eot_length = BYTES_PER_ITEM * number_of_frames
        self._file.write_tag_of_vr_and_length(ExtendedOffsetTableTag, "OV", eot_length)
        for _ in range(number_of_frames):
            self._file.write_unsigned_long_long(0)
        self._file.write_tag_of_vr_and_length(
            ExtendedOffsetTableLengthsTag, "OV", eot_length
        )
        for _ in range(number_of_frames):
            self._file.write_unsigned_long_long(0)

    def write(
        self,
        pixel_data_start: int,
        frame_positions: Sequence[int],
        last_frame_end: int,
    ) -> None:
        BYTES_PER_ITEM = 8
        # Check that last EOT entry is not over 2^64 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**64 - 1:
            raise ValueError(
                "Image data exceeds 2^64 - 1 bytes, likely something is wrong."
            )
        self._file.seek(self._table_start)  # Go to EOT table
        self._file.check_tag_and_length(
            ExtendedOffsetTableTag, BYTES_PER_ITEM * len(frame_positions), True, True
        )
        for frame_position in frame_positions:  # Write EOT
            relative_position = frame_position - pixel_data_start
            self._file.write_unsigned_long_long(relative_position)

        # EOT LENGTHS
        self._file.check_tag_and_length(
            ExtendedOffsetTableLengthsTag,
            BYTES_PER_ITEM * len(frame_positions),
            True,
            True,
        )
        frame_start = frame_positions[0]
        for frame_end in frame_positions[1:]:  # Write EOT lengths
            frame_length = frame_end - frame_start
            self._file.write_unsigned_long_long(frame_length)
            frame_start = frame_end

        # Last frame length, end does not include tag and length
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        last_frame_start = frame_start + TAG_BYTES + LENGTH_BYTES
        last_frame_length = last_frame_end - last_frame_start
        self._file.write_unsigned_long_long(last_frame_length)
