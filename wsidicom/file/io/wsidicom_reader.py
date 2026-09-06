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

import logging
from typing import ClassVar

from pydicom.dataelem import DataElement, RawDataElement
from pydicom.dataset import Dataset
from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotSupportedError, WsiDicomOutOfBoundsError
from wsidicom.file.io.deferred_dataset_reader import FileDeferredDatasetReader
from wsidicom.file.io.deferred_element import DeferredElement
from wsidicom.file.io.frame_index import (
    BasicOffsetTableFrameIndexParser,
    EmptyBasicTableOffsetException,
    ExtendedOffsetFrameIndexParser,
    FrameIndex,
    FrameIndexParser,
    NativePixelDataFrameIndexParser,
    OffsetTableType,
    PixelDataFrameIndexParser,
)
from wsidicom.file.io.frame_index.tiff import (
    EmptyTiffFrameTagsException,
    TiffFrameIndexParser,
)
from wsidicom.file.io.wsidicom_io import WsiDicomReadIO
from wsidicom.instance import WsiDataset
from wsidicom.metadata import ImageType
from wsidicom.tags import (
    ExtendedOffsetTableTag,
    OpticalPathSequenceTag,
    PerFrameFunctionalGroupsSequenceTag,
    SOPInstanceUIDTag,
)
from wsidicom.uid import FileUids

logger = logging.getLogger(__name__)


class WsiDicomReader:
    """Reader for DICOM WSI data in stream"""

    DEFERRED_VALUE_SIZE: ClassVar[int] = 64 * 1024
    """Values longer than this many bytes are read when they are used rather
    than when the dataset is read."""

    def __init__(self, stream: WsiDicomReadIO):
        """
        Parse DICOM stream. If valid WSI type read required parameters.

        Parameters
        ----------
        stream: WsiDicomReadIO
            File to open.
        """
        self._stream = stream
        self._transfer_syntax_uid = UID(self._stream.file_meta_info.TransferSyntaxUID)
        dataset, self._deferred_reader = self._read_dataset()
        self._dataset = WsiDataset(dataset, deferred_reader=self._deferred_reader)
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
        self._frame_index_parser: FrameIndexParser | None = None
        self._frame_index: FrameIndex | None = None

    def _read_dataset(self) -> tuple[Dataset, FileDeferredDatasetReader]:
        """Read the dataset, leaving the per frame functional groups as bytes.

        Read in parts, so that an instance that is not read is turned away before the
        rest of it is parsed, and so that the per frame functional groups sequence is
        searched for the tile positions rather than parsed.

        The read stops at the per frame functional groups sequence, and the stream is
        left there. What follows it, and where the pixel data starts, is read when a
        frame is.

        The elements are gathered as they are read and made into the dataset once,
        rather than a part at a time into a dataset that is added to, which costs an
        insert per element and a dataset per part that is thrown away again.

        Returns
        -------
        tuple[Dataset, FileDeferredDatasetReader]
            The dataset as far as it was read, and a reader for the rest of it.
        """
        elements: dict[BaseTag, DataElement | RawDataElement] = {}
        self._stream.read_elements_until(SOPInstanceUIDTag, elements)
        if not WsiDataset.is_supported_image_type(elements):
            raise WsiDicomNotSupportedError(
                f"{self._stream} is not a whole slide image, or is of an image type "
                f"that is not read."
            )

        self._stream.read_elements_from(
            self._stream.tell(), OpticalPathSequenceTag, elements
        )
        continue_from, deferred_elements = self._read_optical_paths(elements)

        stopped_at = self._stream.read_elements_from(
            continue_from, PerFrameFunctionalGroupsSequenceTag, elements
        )
        if not WsiDataset.is_supported(elements):
            raise WsiDicomNotSupportedError(
                f"{self._stream} is a whole slide image that cannot be read: an "
                f"attribute it needs is missing, or its pixel format is not "
                f"supported."
            )

        if stopped_at == PerFrameFunctionalGroupsSequenceTag:
            per_frame_position = self._stream.tell()
            pixel_data_position = None
        else:
            self._stream.read_elements_from(
                self._stream.tell(), ExtendedOffsetTableTag, elements
            )
            per_frame_position = None
            pixel_data_position = self._stream.tell()
        dataset = WsiDataset.make_dataset(elements, self._stream.transfer_syntax)
        return dataset, FileDeferredDatasetReader(
            self._stream,
            dataset,
            deferred_elements,
            per_frame_position,
            pixel_data_position,
        )

    def _read_optical_paths(
        self, elements: dict[BaseTag, DataElement | RawDataElement]
    ) -> tuple[int, list[DeferredElement]]:
        """Read the optical path sequence, deferring the values (e.g. ICC profile) that
        are large.

        Parameters
        ----------
        elements: dict[BaseTag, DataElement | RawDataElement]
            Elements read so far, which the sequence is added to.

        Returns
        -------
        tuple[int, list[DeferredElement]]
            Offset the rest of the dataset is to be read from, and the elements
            whose values were deferred.
        """
        position = self._stream.tell()
        if self._stream.read_tag() != OpticalPathSequenceTag:
            return position, []
        sequence, deferred, end_of_sequence = self._stream.read_sequence(
            position, self.DEFERRED_VALUE_SIZE
        )
        elements[OpticalPathSequenceTag] = DataElement(
            OpticalPathSequenceTag, "SQ", sequence
        )
        return end_of_sequence, deferred

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def offset_table_type(self) -> OffsetTableType:
        """Return type of the offset table, or None if not present."""
        return self._frame_index_parser_for_stream.offset_table_type

    @property
    def dataset(self) -> WsiDataset:
        """Return pydicom dataset of stream."""
        return self._dataset

    @property
    def image_type(self) -> ImageType:
        return self._dataset.image_type

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
    def frame_index(self) -> FrameIndex:
        """Return frame positions and lengths."""
        frame_index = self._frame_index
        if frame_index is not None:
            return frame_index
        with self._stream.exclusive():
            # Asked again with the stream held, so that two callers at once parse
            # the index once between them rather than once each.
            frame_index = self._frame_index
            if frame_index is None:
                frame_index = self._frame_index_parser_for_stream.parse_frame_index()
                self._frame_index = frame_index
            return frame_index

    @property
    def _frame_index_parser_for_stream(self) -> FrameIndexParser:
        """Parser for the frame index of the stream, made when first asked for.

        Making it is what finds where the pixel data starts, which for an instance
        whose read stopped at the per frame functional groups sequence means reading
        past that, so it is made once and kept.

        Returns
        -------
        FrameIndexParser
            Parser for the frame index.
        """
        parser = self._frame_index_parser
        if parser is not None:
            return parser
        with self._stream.exclusive():
            parser = self._frame_index_parser
            if parser is None:
                parser = self._get_frame_index_parser()
                self._frame_index_parser = parser
            return parser

    @property
    def frame_count(self) -> int:
        """Return number of frames."""
        return self.dataset.frame_count

    @property
    def filepath(self) -> UPath:
        """Return the filepath the stream is backed by."""
        return self._stream.filepath

    def read_frame(self, frame_index: int) -> bytes:
        """Return frame data from pixel data by frame index.

        Raises WsiDicomOutOfBoundsError if the frame is not in this file.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        -------
        bytes
            The frame as bytes
        """
        index_in_file = frame_index - self.frame_offset
        if not 0 <= index_in_file < len(self.frame_index):
            raise WsiDicomOutOfBoundsError(
                f"Frame index {frame_index}",
                f"frames {self.frame_offset} to "
                f"{self.frame_offset + len(self.frame_index) - 1} in file",
            )
        frame_position, frame_length = self.frame_index[index_in_file]
        with self._stream.exclusive():
            self._stream.seek(frame_position, 0)
            return self._stream.read(frame_length)

    def _get_frame_index_parser(self) -> FrameIndexParser:
        """Create frame index for stream."""
        pixel_data_position = self._deferred_reader.seek_to_pixel_data()
        self._stream.seek(pixel_data_position)
        if not self.transfer_syntax.is_encapsulated:
            return NativePixelDataFrameIndexParser(
                self._stream,
                pixel_data_position,
                self._dataset.frame_count,
                self._dataset.tile_size,
                self._dataset.samples_per_pixel,
                self._dataset.bits,
            )
        pixel_data_or_eot_tag = Tag(self._stream.read_tag())
        if pixel_data_or_eot_tag == ExtendedOffsetTableTag:
            return ExtendedOffsetFrameIndexParser(
                self._stream, pixel_data_position, self.frame_count
            )
        try:
            return BasicOffsetTableFrameIndexParser(
                self._stream, pixel_data_position, self.frame_count
            )
        except EmptyBasicTableOffsetException:
            pass

        try:
            return TiffFrameIndexParser(
                self._stream, pixel_data_position, self.frame_count
            )
        except EmptyTiffFrameTagsException:
            self._stream.seek(pixel_data_position)
            return PixelDataFrameIndexParser(
                self._stream, pixel_data_position, self.frame_count
            )

    def close(self) -> None:
        """Close stream."""
        self._stream.close()
