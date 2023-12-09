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

from pydicom.tag import Tag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.offset_table import OffsetTable
from wsidicom.tags import ExtendedOffsetTableLengthsTag, ExtendedOffsetTableTag
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class Eot(OffsetTable):
    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.EXTENDED

    @property
    def bytes_per_item(self) -> int:
        return 8

    @property
    def mode(self) -> str:
        return "<Q"

    def _get_index(self):
        table = self._read_table()
        return self._parse_table(table, self._pixels_start)

    def _get_pixels_start(self) -> int:
        pixel_data_or_eot_tag = Tag(self._file.read_tag())
        if pixel_data_or_eot_tag != ExtendedOffsetTableTag:
            raise WsiDicomFileError(str(self._file), "Expected EOT tag")
        self._file.read_tag_vr()
        eot_length = self._file.read_tag_length(True)
        self._file.seek(eot_length, 1)
        self._read_eot_lengths_tag()
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is not None:
            raise WsiDicomFileError(str(self._file), "Expected empty BOT")
        return self._file.tell()

    def _read_table(self) -> bytes:
        """Read extended table offset (EOT) and EOT lengths. Filepointer should be
        positionend to extended offset table.

        Returns
        ----------
        bytes
            EOT in bytes.
        """
        eot_tag = Tag(self._file.read_tag())
        if eot_tag != ExtendedOffsetTableTag:
            raise ValueError(f"Expected ExtendedOffsetTable tag, got {eot_tag}")
        self._file.read_tag_vr()
        eot_length = self._read_eot_length()
        # Read the EOT into bytes
        eot = self._file.read(eot_length)
        # Read EOT lengths tag
        self._read_eot_lengths_tag()
        return eot

    def _read_eot_length(self) -> int:
        """Read the length of the extended table offset (EOT).

        Returns
        ----------
        int
            EOT length.
        """
        EOT_BYTES = 8
        eot_length = self._file.read_tag_length(True)
        if eot_length == 0:
            raise WsiDicomFileError(
                str(self._file), "Expected Extended offset table present but empty"
            )
        elif eot_length % EOT_BYTES:
            raise WsiDicomFileError(
                str(self._file),
                "Extended offset table should be a multiple of " f"{EOT_BYTES} bytes",
            )
        return eot_length

    def _read_eot_lengths_tag(self):
        """Skip over the length of the extended table offset lengths tag."""
        eot_lenths_tag = self._file.read_tag()
        if eot_lenths_tag != ExtendedOffsetTableLengthsTag:
            raise WsiDicomFileError(
                str(self._file),
                "Expected Extended offset table lengths tag after reading "
                f"Extended offset table, found {eot_lenths_tag}",
            )
        self._file.read_tag_vr()
        length = self._file.read_tag_length(True)
        # Jump over EOT lengths for now
        self._file.seek(length, 1)
