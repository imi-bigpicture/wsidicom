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
import threading

from pydicom.tag import Tag
from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotSupportedError, WsiDicomOutOfBoundsError
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
from wsidicom.file.io.per_frame_functional_groups_reader import (
    PerFrameFunctionalGroupsReader,
    UnscannablePerFrameGroupsException,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.instance import WsiDataset
from wsidicom.metadata import ImageType
from wsidicom.tags import ExtendedOffsetTableTag, PerFrameFunctionalGroupsSequenceTag
from wsidicom.uid import FileUids

logger = logging.getLogger(__name__)


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
        self._transfer_syntax_uid = UID(self._stream.file_meta_info.TransferSyntaxUID)
        self._dataset = self._read_dataset()
        self._pixel_data_position = self._stream.tell()

        self._image_type = WsiDataset.is_supported_wsi_dicom(self._dataset)
        if self._image_type is None:
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
        self._frame_index_parser: FrameIndexParser | None = None
        self._frame_index: FrameIndex | None = None

    def _read_dataset(self) -> WsiDataset:
        """Read the dataset, leaving the per frame functional groups as bytes.

        The read stops at the sequence, :class:`PerFrameFunctionalGroupsReader` searches
        its bytes for the tile positions, and the read picks up again after it. If the
        search cannot be trusted, or there is nothing to find because the image is
        tiled full, the sequence is read into pydicom datasets instead, so the outcome
        is the same either way. The stream is left where the pixel data starts,
        which is only known once the sequence has been passed, one way or the other.

        Returns
        -------
        WsiDataset
            Dataset, carrying the tile positions if they were found.
        """
        dataset, stopped_at = self._stream.read_dataset_until(
            stop_tag=PerFrameFunctionalGroupsSequenceTag
        )
        # The read stops at the first tag ordered at or after the sequence, which need
        # not be the sequence itself, and the rest of the dataset starts there.
        continue_from = self._stream.tell()
        frame_positions = None
        # A tiled full image states where its frames are by the order they are in, so
        # its per frame groups hold no tile positions to find, and a full tile index
        # would not ask for them if they did.
        tiled_full = dataset.get("DimensionOrganizationType", None) == "TILED_FULL"
        if stopped_at == PerFrameFunctionalGroupsSequenceTag and not tiled_full:
            reader = PerFrameFunctionalGroupsReader(
                self._stream,
                continue_from,
                int(dataset.get("NumberOfFrames", 0) or 0),
                self._transfer_syntax_uid,
                specific_character_set=dataset.get("SpecificCharacterSet", None),
            )
            try:
                frame_positions = reader.read_positions()
            except UnscannablePerFrameGroupsException as exception:
                # Carry on from the sequence rather than past it, so that it is read.
                logger.debug(
                    "Could not find the tile positions of %s in the bytes of the per "
                    "frame functional groups sequence (%s). Reading the sequence into "
                    "datasets instead, which is slower and holds more memory.",
                    self._stream,
                    exception,
                )
            else:
                continue_from = reader.end_of_sequence

        # The rest of the dataset: the sequence when its bytes were not enough, and
        # whatever is between it and the pixel data.
        rest = self._stream.read_dataset_from(continue_from, ExtendedOffsetTableTag)
        dataset.update(rest)
        return WsiDataset(dataset, frame_positions=frame_positions)

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
    def image_type(self) -> ImageType | None:
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
    def frame_index(self) -> FrameIndex:
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

    def close(self) -> None:
        """Close stream."""
        self._stream.close()
