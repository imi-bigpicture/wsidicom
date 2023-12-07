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

"""Module for reading DICOM WSI files."""

import threading
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple

from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag
from pydicom.uid import UID

from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomFileError, WsiDicomNotSupportedError
from wsidicom.file.io.frame_index import (
    Bot,
    EmptyBot,
    EmptyBotException,
    Eot,
    FrameIndex,
    NativePixelData,
    OffsetTableType,
)
from wsidicom.tags import ExtendedOffsetTableTag
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.instance import ImageType, WsiDataset
from wsidicom.uid import FileUids


class WsiDicomReader:
    """Reader for DICOM WSI data in stream"""

    def __init__(self, stream: WsiDicomIO):
        """
        Parse DICOM stream. If valid WSI type read required parameters.

        Parameters
        ----------
        stream: WsiDicomIO
            File to open.
        """
        self._lock = threading.Lock()
        self._stream = stream
        try:
            file_meta = self._stream.read_file_meta_info()
        except InvalidDicomError:
            raise WsiDicomFileError(str(stream), "is not a DICOM file or stream.")
        self._transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)
        self._stream.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._stream.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR
        dataset = self._stream.read_dataset()
        self._pixel_data_position = self._stream.tell()

        self._image_type = WsiDataset.is_supported_wsi_dicom(dataset)
        if self._image_type is not None:
            self._dataset = WsiDataset(dataset)
        else:
            raise WsiDicomNotSupportedError(
                f"Non-supported file or stream {self._stream}."
            )
        syntax_supported = Codec.is_supported(
            self.transfer_syntax,
            self._dataset.samples_per_pixel,
            self._dataset.bits,
            self._dataset.photometric_interpretation,
        )
        if not syntax_supported:
            raise WsiDicomNotSupportedError(
                f"Non-supported transfer syntax {self.transfer_syntax}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def offset_table_type(self) -> OffsetTableType:
        """Return type of the offset table, or None if not present."""
        return self.frame_index.offset_table_type

    @cached_property
    def frame_index(self) -> FrameIndex:
        return self._get_frame_index()

    @property
    def dataset(self) -> WsiDataset:
        """Return pydicom dataset of stream."""
        return self._dataset

    @property
    def image_type(self) -> Optional[ImageType]:
        return self._image_type

    @property
    def uids(self) -> FileUids:
        """Return uids."""
        return self.dataset.uids

    @property
    def transfer_syntax(self) -> UID:
        """Return transfer syntax uid."""
        return self._transfer_syntax_uid

    @property
    def frame_offset(self) -> int:
        """Return frame offset (for concatenated stream, 0 otherwise)."""
        return self.dataset.frame_offset

    @property
    def frame_positions(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths."""
        return self.frame_index.index

    @property
    def frame_count(self) -> int:
        """Return number of frames."""
        return self.dataset.frame_count

    @property
    def filepath(self) -> Optional[Path]:
        """Return filename if stream is file."""
        return self._stream.filepath

    def read_frame(self, frame_index: int) -> bytes:
        """Return frame data from pixel data by frame index.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        bytes
            The frame as bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_positions[frame_index]
        with self._lock:
            self._stream.seek(frame_position, 0)
            return self._stream.read(frame_length)

    @classmethod
    def open(cls, file: Path) -> "WsiDicomReader":
        """Open file in path as WsiDicomFileReader.

        Parameters
        ----------
        file: Path
            Path to file.

        Returns
        ----------
        WsiDicomFileReader
            WsiDicomFileReader for file.
        """
        stream = WsiDicomIO.open(file, "rb")
        return cls(stream)

    def _get_frame_index(self) -> FrameIndex:
        """Create frame index for stream."""
        self._stream.seek(self._pixel_data_position)
        if not self.transfer_syntax.is_encapsulated:
            return NativePixelData(
                self._stream,
                self._pixel_data_position,
                self._dataset.frame_count,
                self._dataset.tile_size,
                self._dataset.samples_per_pixel,
                self._dataset.bits,
            )
        pixel_data_or_eot_tag = Tag(self._stream.read_tag())
        if pixel_data_or_eot_tag == ExtendedOffsetTableTag:
            return Eot(self._stream, self._pixel_data_position, self.frame_count)
        try:
            return Bot(self._stream, self._pixel_data_position, self.frame_count)
        except EmptyBotException:
            self._stream.seek(self._pixel_data_position)
            return EmptyBot(self._stream, self._pixel_data_position, self.frame_count)

    def close(self, force: Optional[bool] = False) -> None:
        """Close stream."""
        self._stream.close(force)
