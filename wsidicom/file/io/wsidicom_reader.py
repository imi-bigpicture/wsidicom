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
from typing import List, Optional, Tuple

from pydicom.tag import Tag
from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotSupportedError
from wsidicom.file.io.frame_index import (
    BasicOffsetTableFrameIndexParser,
    EmptyBasicTableOffsetException,
    ExtendedOffsetFrameIndexParser,
    FrameIndexParser,
    NativePixelDataFrameIndexParser,
    OffsetTableType,
    PixelDataFrameIndexParser,
)
from wsidicom.file.io.frame_index.tiff import (
    EmptyTiffFrameTagsException,
    TiffFrameIndexParser,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.instance import ImageType, WsiDataset
from wsidicom.tags import ExtendedOffsetTableTag
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
        self._lock = threading.RLock()
        self._stream = stream
        self._transfer_syntax_uid = UID(self._stream.file_meta_info.TransferSyntaxUID)
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
        self._frame_index_parser: Optional[FrameIndexParser] = None
        self._frame_index: Optional[List[Tuple[int, int]]] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def offset_table_type(self) -> OffsetTableType:
        """Return type of the offset table, or None if not present."""
        if self._frame_index_parser is None:
            with self._lock:
                if self._frame_index_parser is None:
                    self._frame_index_parser = self._get_frame_index_parser()

        return self._frame_index_parser.offset_table_type

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
    def frame_index(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths."""
        if self._frame_index is None:
            with self._lock:
                if self._frame_index_parser is None:
                    self._frame_index_parser = self._get_frame_index_parser()
                if self._frame_index is None:
                    self._frame_index = self._frame_index_parser.parse_frame_index()
        return self._frame_index

    @property
    def frame_count(self) -> int:
        """Return number of frames."""
        return self.dataset.frame_count

    @property
    def filepath(self) -> Optional[UPath]:
        """Return filename if stream is file."""
        return self._stream.filepath

    def read_frame(self, frame_index: int) -> bytes:
        """Return frame data from pixel data by frame index.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        -------
        bytes
            The frame as bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_index[frame_index]
        with self._lock:
            self._stream.seek(frame_position, 0)
            return self._stream.read(frame_length)

    def _get_frame_index_parser(self) -> FrameIndexParser:
        """Create frame index for stream."""
        self._stream.seek(self._pixel_data_position)
        if not self.transfer_syntax.is_encapsulated:
            return NativePixelDataFrameIndexParser(
                self._stream,
                self._pixel_data_position,
                self._dataset.frame_count,
                self._dataset.tile_size,
                self._dataset.samples_per_pixel,
                self._dataset.bits,
            )
        pixel_data_or_eot_tag = Tag(self._stream.read_tag())
        if pixel_data_or_eot_tag == ExtendedOffsetTableTag:
            return ExtendedOffsetFrameIndexParser(
                self._stream, self._pixel_data_position, self.frame_count
            )
        try:
            return BasicOffsetTableFrameIndexParser(
                self._stream, self._pixel_data_position, self.frame_count
            )
        except EmptyBasicTableOffsetException:
            pass

        try:
            return TiffFrameIndexParser(
                self._stream, self._pixel_data_position, self.frame_count
            )
        except EmptyTiffFrameTagsException:
            self._stream.seek(self._pixel_data_position)
            return PixelDataFrameIndexParser(
                self._stream, self._pixel_data_position, self.frame_count
            )

    def close(self, force: Optional[bool] = False) -> None:
        """Close stream."""
        self._stream.close(force)
