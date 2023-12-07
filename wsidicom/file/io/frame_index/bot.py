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

from typing import List, Optional, Tuple
from wsidicom.file.io.frame_index.offset_table import OffsetTable
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class EmptyBotException(Exception):
    """Exception raised when BOT was empty."""

    pass


class Bot(OffsetTable):
    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.BASIC

    @property
    def bytes_per_item(self) -> int:
        return 4

    @property
    def mode(self) -> str:
        return "<L"

    def _get_index(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from bot."""
        self._validate_pixel_data_start()
        table = self._read_table()
        pixels_start = self._file.tell()
        assert table is not None
        return self._parse_table(table, pixels_start)

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start()
        bot_length = self._read_bot_length()
        if bot_length is None:
            raise EmptyBotException()
        return self._file.tell()

    def _read_table(self) -> Optional[bytes]:
        """Read basic table offset (BOT). Returns None if BOT is empty.

        Returns
        ----------
        Optional[bytes]
            BOT in bytes.
        """
        bot_length = self._read_bot_length()
        if bot_length is None:
            raise EmptyBotException()
        return self._file.read(bot_length, need_exact_length=True)
