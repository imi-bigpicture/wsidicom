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
from typing import ClassVar

from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotSupportedError, WsiDicomOutOfBoundsError
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
from wsidicom.file.io.per_frame_functional_groups_reader import (
    PerFrameFunctionalGroupsReader,
    UnscannablePerFrameGroupsException,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.instance import WsiDataset
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.metadata import ImageType
from wsidicom.tags import (
    DimensionOrganizationTypeTag,
    ExtendedOffsetTableTag,
    NumberOfFramesTag,
    OpticalPathSequenceTag,
    PerFrameFunctionalGroupsSequenceTag,
    SOPInstanceUIDTag,
    SpecificCharacterSetTag,
)
from wsidicom.uid import FileUids

logger = logging.getLogger(__name__)


class WsiDicomReader:
    """Reader for DICOM WSI data in stream"""

    DEFERRED_VALUE_SIZE: ClassVar[int] = 64 * 1024
    """Values longer than this many bytes are read when they are used rather
    than when the dataset is read."""

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
        self._deferred_elements: list[DeferredElement] = []
        self._transfer_syntax_uid = UID(self._stream.file_meta_info.TransferSyntaxUID)
        dataset = self._read_dataset()
        if dataset is None:
            raise WsiDicomNotSupportedError(
                f"Non-supported file or stream {self._stream}."
            )
        self._dataset = dataset
        self._pixel_data_position = self._stream.tell()
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

    def _read_dataset(self) -> WsiDataset | None:
        """Read the dataset, leaving the per frame functional groups as bytes.

        Read in parts, so that an instance that is not read is turned away before the
        rest of it is parsed, and so that the per frame functional groups sequence is
        searched for the tile positions rather than parsed.

        The stream is left where the pixel data starts, which is only known once the
        sequence has been passed, one way or the other.

        Returns
        -------
        WsiDataset | None
            Dataset, carrying the tile positions if they were found, or None if this
            is not an instance to read.
        """
        dataset = self._stream.read_dataset_until(stop_tag=SOPInstanceUIDTag)
        if not WsiDataset.is_supported_image_type(dataset):
            return None

        self._stream.read_dataset_into(
            self._stream.tell(), OpticalPathSequenceTag, dataset
        )
        continue_from = self._read_optical_paths(dataset)

        stopped_at = self._stream.read_dataset_into(
            continue_from, PerFrameFunctionalGroupsSequenceTag, dataset
        )
        if not WsiDataset.is_supported(dataset):
            return None

        frame_positions, continue_from = self._read_frame_positions(dataset, stopped_at)
        self._stream.read_dataset_into(continue_from, ExtendedOffsetTableTag, dataset)
        return WsiDataset(dataset, frame_positions, self)

    def _read_optical_paths(self, dataset: Dataset) -> int:
        """Read the optical path sequence, deferring the values (e.g. ICC profile) that
        are large.

        Parameters
        ----------
        dataset: Dataset
            Dataset read so far, which the sequence is added to.

        Returns
        -------
        int
            Offset the rest of the dataset is to be read from.
        """
        position = self._stream.tell()
        if self._stream.read_tag() != OpticalPathSequenceTag:
            return position
        sequence, deferred, end_of_sequence = self._stream.read_sequence(
            position, self.DEFERRED_VALUE_SIZE
        )
        dataset[OpticalPathSequenceTag] = DataElement(
            OpticalPathSequenceTag, "SQ", sequence
        )
        self._deferred_elements.extend(deferred)
        return end_of_sequence

    def read_deferred_elements(self) -> None:
        """Read every deferred element into the dataset it belongs in.

        Does nothing once there is nothing left to read, so a caller wanting a whole
        dataset can ask without first asking whether it is already whole.
        """
        with self._lock:
            while self._deferred_elements:
                element = self._deferred_elements.pop()
                self._stream.seek(element.offset)
                element.set(self._stream.read(element.length, need_exact_length=True))

    def _read_frame_positions(
        self, dataset: Dataset, stopped_at: BaseTag | None
    ) -> tuple[PerFrameGroupPositions | None, int]:
        """Search the bytes of the per frame functional groups for the tile positions.

        Called with the stream at the tag the read of the dataset stopped at, which is
        the first tag ordered at or after the sequence and need not be the sequence
        itself. A tiled full image states where its frames are by the order they are
        in, so its per frame groups hold no tile positions to find, and a full tile
        index would not ask for them if they did.

        Where there is nothing to find, or the search cannot be trusted, the read is
        to carry on from the sequence rather than past it, so that it is read into
        datasets instead. That is slower and holds more memory, but the outcome is
        the same either way.

        Parameters
        ----------
        dataset: Dataset
            Dataset read so far, for what the search needs to know about the frames.
        stopped_at: BaseTag | None
            Tag the read of the dataset stopped at, or None if the stream ended.

        Returns
        -------
        tuple[PerFrameGroupPositions | None, int]
            Tile positions if they were found, and the offset the rest of the dataset
            is to be read from.
        """
        continue_from = self._stream.tell()
        tiled_full = (
            WsiDataset.get_value(dataset, DimensionOrganizationTypeTag) == "TILED_FULL"
        )
        if stopped_at != PerFrameFunctionalGroupsSequenceTag or tiled_full:
            return None, continue_from
        reader = PerFrameFunctionalGroupsReader(
            self._stream,
            continue_from,
            int(WsiDataset.get_value(dataset, NumberOfFramesTag, 0) or 0),
            self._transfer_syntax_uid,
            specific_character_set=WsiDataset.get_value(
                dataset, SpecificCharacterSetTag
            ),
        )
        try:
            frame_positions = reader.read_positions()
        except UnscannablePerFrameGroupsException as exception:
            logger.debug(
                "Could not find the tile positions of %s in the bytes of the per "
                "frame functional groups sequence (%s). Reading the sequence into "
                "datasets instead, which is slower and holds more memory.",
                self._stream,
                exception,
            )
            return None, continue_from
        return frame_positions, reader.end_of_sequence

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
