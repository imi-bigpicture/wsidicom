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

from enum import Enum
from pathlib import Path
from typing import BinaryIO, Optional

from pydicom.filebase import DicomFileLike
from pydicom.tag import BaseTag


class OffsetTableType(Enum):
    NONE = "none"
    BASIC = "BOT"
    EXTENDED = "EOT"

    @classmethod
    def from_string(cls, offset_table: Optional[str]) -> "OffsetTableType":
        if offset_table is None:
            return OffsetTableType.NONE
        if offset_table.strip().lower() == "eot":
            return OffsetTableType.EXTENDED
        return OffsetTableType.BASIC


class WsiDicomFileBase:
    """Base class for reading or writing DICOM WSI file."""

    def __init__(
        self, file: BinaryIO, filepath: Optional[Path] = None, owned: bool = False
    ):
        """
        Create a WsiDicomFileBase.

        Parameters
        ----------
        file: BinaryIO
            Stream to open.
        filepath: Optional[Path] = None
            Optional filepath of stream.
        owned: bool = False
            If the stream should be closed by this instance.
        """
        self._file = DicomFileLike(file)
        self._filepath = filepath
        self._owned = owned
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.filepath})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        return f"File of stream: {self._file}"

    @property
    def filepath(self) -> Optional[Path]:
        """Return filepath."""
        return self._filepath

    def _read_tag_length(self, with_vr: bool = True) -> int:
        if (not self._file.is_implicit_VR) and with_vr:
            # Read VR
            self._file.read_UL()
        return self._file.read_UL()

    def _check_tag_and_length(
        self, tag: BaseTag, length: int, with_vr: bool = True
    ) -> None:
        """Check if tag at position is expected tag with expected length.

        Parameters
        ----------
        tag: BaseTag
            Expected tag.
        length: int
            Expected length.

        """
        read_tag = self._file.read_tag()
        if tag != read_tag:
            raise ValueError(f"Found tag {read_tag} expected {tag}.")
        read_length = self._read_tag_length(with_vr)
        if length != read_length:
            raise ValueError(f"Found length {read_length} expected {length}.")

    def close(self, force: Optional[bool] = False) -> None:
        """Close the file if owned by instance or forced."""
        if self._owned or force:
            self._file.close()
