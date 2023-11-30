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
    """Offset table type."""

    NONE = "none"
    EMPTY = "empty"
    BASIC = "BOT"
    EXTENDED = "EOT"

    @classmethod
    def from_string(cls, offset_table: str) -> "OffsetTableType":
        """Return OffsetTableType parsed from string."""
        if offset_table == "none":
            return OffsetTableType.NONE
        if offset_table.strip().lower() == "empty":
            return OffsetTableType.EMPTY
        if offset_table.strip().lower() == "eot":
            return OffsetTableType.EXTENDED
        if offset_table.strip().lower() == "bot":
            return OffsetTableType.BASIC
        raise ValueError(f"Unknown offset table type: {offset_table}")


class WsiDicomFileBase:
    """Base class for reading or writing DICOM WSI file."""

    def __init__(
        self, stream: BinaryIO, filepath: Optional[Path] = None, owned: bool = False
    ):
        """
        Create a WsiDicomFileBase.

        Parameters
        ----------
        stream: BinaryIO
            Stream to open.
        filepath: Optional[Path] = None
            Optional filepath of stream.
        owned: bool = False
            If the stream should be closed by this instance.
        """
        self._stream = stream
        self._file = DicomFileLike(self._stream)
        self._filepath = filepath
        self._owned = owned
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        return f"File of stream: {self._file.name}"

    @property
    def filepath(self) -> Optional[Path]:
        """Return filepath."""
        return self._filepath

    def _read_tag_length(self) -> int:
        """Read tag length."""
        return self._file.read_UL()

    def _read_tag_vr(self) -> None:
        """Read tag VR if implicit VRi."""
        if not self._file.is_implicit_VR:
            self._file.read(4, need_exact_length=True)

    def _check_tag_and_length(self, tag: BaseTag, length: int, with_vr: bool) -> None:
        """Check if tag at position is expected tag with expected length.

        Parameters
        ----------
        tag: BaseTag
            Expected tag.
        length: int
            Expected length.
        with_vr: bool
            If tag is expected to have VR.

        """
        read_tag = self._file.read_tag()
        if tag != read_tag:
            raise ValueError(f"Found tag {read_tag} expected {tag}.")
        if with_vr:
            self._read_tag_vr()
        read_length = self._read_tag_length()
        if length != read_length:
            raise ValueError(f"Found length {read_length} expected {length}.")

    def close(self, force: Optional[bool] = False) -> None:
        """Close the file if owned by instance or forced."""
        if self._owned or force:
            self._file.close()
