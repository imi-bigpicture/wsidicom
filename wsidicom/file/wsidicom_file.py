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

import threading
import warnings
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from struct import unpack
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast

from pydicom.filebase import DicomFileLike
from pydicom.filereader import read_file_meta_info, read_partial
from pydicom.misc import is_dicom
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.wsidicom_file_base import OffsetTableType, WsiDicomFileBase
from wsidicom.geometry import Size
from wsidicom.instance import ImageType, WsiDataset
from wsidicom.uid import FileUids, SlideUids


class WsiDicomFile(WsiDicomFileBase):
    """Represents a DICOM file (potentially) containing WSI image and metadata."""

    def __init__(self, filepath: Path):
        """Open dicom file in filepath. If valid wsi type read required
        parameters. Parses frames in pixel data but does not read the frames.

        Parameters
        ----------
        filepath: Path
            Path to file to open
        """
        self._lock = threading.Lock()

        if not is_dicom(filepath):
            raise WsiDicomFileError(filepath, "is not a DICOM file")

        file_meta = read_file_meta_info(filepath)
        self._transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)

        super().__init__(filepath, mode="rb")
        self._fp.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR
        pixel_data_tags = {Tag("PixelData"), Tag("ExtendedOffsetTable")}

        def _stop_at(tag: BaseTag, VR: Optional[str], length: int) -> bool:
            return tag in pixel_data_tags

        dataset = read_partial(
            cast(BinaryIO, self._fp),
            _stop_at,
            defer_size=None,
            force=False,
            specific_tags=None,
        )
        self._pixel_data_position = self._fp.tell()

        self._image_type = WsiDataset.is_supported_wsi_dicom(
            dataset, self.transfer_syntax
        )
        if self._image_type is not None:
            self._dataset = WsiDataset(dataset)
            instance_uid = self.dataset.uids.instance
            concatenation_uid = self.dataset.uids.concatenation
            slide_uids = self.dataset.uids.slide
            self._uids = FileUids(instance_uid, concatenation_uid, slide_uids)
            self._frame_offset = self.dataset.frame_offset
            self._frame_count = self.dataset.frame_count
        else:
            warnings.warn(f"Non-supported file {filepath}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.filepath})"

    def __str__(self) -> str:
        return self.pretty_str()

    @cached_property
    def offset_table_type(self) -> OffsetTableType:
        """Return type of the offset table, or None if not present."""
        return self._get_offset_table_type()

    @property
    def dataset(self) -> WsiDataset:
        """Return pydicom dataset of file."""
        return self._dataset

    @property
    def image_type(self) -> Optional[ImageType]:
        return self._image_type

    @property
    def uids(self) -> FileUids:
        """Return uids"""
        return self._uids

    @property
    def transfer_syntax(self) -> UID:
        """Return transfer syntax uid"""
        return self._transfer_syntax_uid

    @property
    def frame_offset(self) -> int:
        """Return frame offset (for concatenated file, 0 otherwise)"""
        return self._frame_offset

    @cached_property
    def frame_positions(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths"""
        (frame_positions, offset_table_type) = self._parse_pixel_data()
        self._offset_table_type = offset_table_type
        return frame_positions

    @property
    def frame_count(self) -> int:
        """Return number of frames"""
        return self._frame_count

    def get_filepointer(self, frame_index: int) -> Tuple[DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame length for frame
        number.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        Tuple[WsiDicomFileLike, int, int]:
            File pointer, frame offset and frame length in number of bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_positions[frame_index]
        return self._fp, frame_position, frame_length

    def _get_offset_table_type(self) -> OffsetTableType:
        """
        Parse file for basic (BOT) or extended offset table (EOT). Return if file has
        EOT and do not check that the BOT is empty (it should be empty according to
        specifications, but we do not care if it not).
        """
        self._fp.seek(self._pixel_data_position)
        pixel_data_or_eot_tag = Tag(self._fp.read_tag())
        if pixel_data_or_eot_tag == Tag("ExtendedOffsetTable"):
            eot_length = self._read_tag_length()
            self._fp.seek(eot_length, 1)
            self._read_eot_lengths_tag()
            return OffsetTableType.EXTENDED
        self._validate_pixel_data_start(pixel_data_or_eot_tag)
        bot_length = self._read_bot_length()
        if bot_length is not None:
            return OffsetTableType.BASIC
        return OffsetTableType.NONE

    def _read_bot_length(self) -> Optional[int]:
        """Read the length of the basic table offset (BOT). Returns None if BOT
        is empty.

        Returns
        ----------
        Optional[int]
            BOT length.
        """
        BOT_BYTES = 4
        if self._fp.read_tag() != ItemTag:
            raise WsiDicomFileError(
                self.filepath, "Basic offset table did not start with an ItemTag"
            )
        bot_length = self._fp.read_UL()
        if bot_length == 0:
            return None
        elif bot_length % BOT_BYTES:
            raise WsiDicomFileError(
                self.filepath,
                f"Basic offset table should be a multiple of {BOT_BYTES} bytes",
            )
        return bot_length

    def _read_bot(self) -> Optional[bytes]:
        """Read basic table offset (BOT). Returns None if BOT is empty. Filepoiter
        should be positionen to pixel data.

        Returns
        ----------
        Optional[bytes]
            BOT in bytes.
        """
        bot_length = self._read_bot_length()
        if bot_length is None:
            return None
        bot = self._fp.read(bot_length)
        return bot

    def _read_eot_length(self) -> int:
        """Read the length of the extended table offset (EOT).

        Returns
        ----------
        int
            EOT length.
        """
        EOT_BYTES = 8
        eot_length = self._read_tag_length()
        if eot_length == 0:
            raise WsiDicomFileError(
                self.filepath, "Expected Extended offset table present but empty"
            )
        elif eot_length % EOT_BYTES:
            raise WsiDicomFileError(
                self.filepath,
                "Extended offset table should be a multiple of " f"{EOT_BYTES} bytes",
            )
        return eot_length

    def _read_eot_lengths_tag(self):
        """Skip over the length of the extended table offset lengths tag."""
        eot_lenths_tag = self._fp.read_tag()
        if eot_lenths_tag != Tag("ExtendedOffsetTableLengths"):
            raise WsiDicomFileError(
                self.filepath,
                "Expected Extended offset table lengths tag after reading "
                f"Extended offset table, found {eot_lenths_tag}",
            )
        length = self._read_tag_length()
        # Jump over EOT lengths for now
        self._fp.seek(length, 1)

    def _read_eot(self) -> bytes:
        """Read extended table offset (EOT) and EOT lengths. Filepointer should be
        positionend to extended offset table.

        Returns
        ----------
        bytes
            EOT in bytes.
        """
        eot_tag = Tag(self._fp.read_tag())
        if eot_tag != Tag("ExtendedOffsetTable"):
            raise ValueError(f"Expected ExtendedOffsetTable tag, got {eot_tag}")
        eot_length = self._read_eot_length()
        # Read the EOT into bytes
        eot = self._fp.read(eot_length)
        # Read EOT lengths tag
        self._read_eot_lengths_tag()
        return eot

    def _parse_table(
        self, table: bytes, table_type: OffsetTableType, pixels_start: int
    ) -> List[Tuple[int, int]]:
        """Parse table with offsets (BOT or EOT).

        Parameters
        ----------
        table: bytes
            BOT or EOT as bytes
        table_type: OffsetTableType
            Type of table, 'bot' or 'eot'.
        pixels_start: int
            Position of first frame item in pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            A list with frame positions and frame lengths.
        """
        if self._fp.is_little_endian:
            mode = "<"
        else:
            mode = ">"
        if table_type == OffsetTableType.BASIC:
            bytes_per_item = 4
            mode += "L"
        elif table_type == OffsetTableType.EXTENDED:
            bytes_per_item = 8
            mode = "Q"
        else:
            raise ValueError("table type should be 'bot' or 'eot'")
        table_length = len(table)
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        positions: List[Tuple[int, int]] = []
        # Read through table to get offset and length for all but last item
        # All read offsets are for item tag of frame and relative to first
        # frame in pixel data.
        this_offset: int = unpack(mode, table[0:bytes_per_item])[0]
        if this_offset != 0:
            raise ValueError("First item in table should be at offset 0")
        for index in range(bytes_per_item, table_length, bytes_per_item):
            next_offset = unpack(mode, table[index : index + bytes_per_item])[0]
            offset = this_offset + TAG_BYTES + LENGTH_BYTES
            length = next_offset - offset
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, "Invalid frame length")
            positions.append((pixels_start + offset, length))
            this_offset = next_offset

        # Go to last frame in pixel data and read the length of the frame
        self._fp.seek(pixels_start + this_offset)
        if self._fp.read_tag() != ItemTag:
            raise WsiDicomFileError(self.filepath, "Excepcted ItemTag in PixelData")
        length: int = self._fp.read_UL()
        if length == 0 or length % 2:
            raise WsiDicomFileError(self.filepath, "Invalid frame length")
        offset = this_offset + TAG_BYTES + LENGTH_BYTES
        positions.append((pixels_start + offset, length))

        return positions

    def _read_positions_from_pixeldata(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from sequence of frames that ends
        with a tag not equal to ItemTag. fp needs to be positioned after the
        BOT.
        Each frame contains:
        item tag (4 bytes)
        item length (4 bytes)
        item data (item length)
        The position of item data and the item length is stored.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        TAG_BYTES = 4
        LENGTH_BYTES = 4
        positions: List[Tuple[int, int]] = []
        frame_position = self._fp.tell()
        # Read items until sequence delimiter
        while self._fp.read_tag() == ItemTag:
            # Read item length
            length: int = self._fp.read_UL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, "Invalid frame length")
            positions.append((frame_position + TAG_BYTES + LENGTH_BYTES, length))
            # Jump to end of frame
            self._fp.seek(length, 1)
            frame_position = self._fp.tell()
        self._read_sequence_delimiter()
        return positions

    def _read_sequence_delimiter(self):
        """Check if last read tag was a sequence delimiter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._fp.seek(-TAG_BYTES, 1)
        if self._fp.read_tag() != SequenceDelimiterTag:
            raise WsiDicomFileError(self.filepath, "No sequence delimeter tag")

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
        fp, frame_position, frame_length = self.get_filepointer(frame_index)
        with self._lock:
            fp.seek(frame_position, 0)
            frame: bytes = fp.read(frame_length)
        return frame

    def _validate_pixel_data_start(self, tag: Union[BaseTag, Tuple[int, int]]):
        """Check that pixel data tag is present and that the tag length is
        set as undefined. Raises WsiDicomFileError otherwise.

        Parameters
        ----------
        tag: Union[BaseTag, Tuple[int, int]]
            Tag that should be pixel data tag.
        """
        if tag != Tag("PixelData"):
            WsiDicomFileError(self.filepath, "Expected PixelData tag")
        length = self._read_tag_length()
        if length != 0xFFFFFFFF:
            raise WsiDicomFileError(
                self.filepath, "Expected undefined length when reading Pixel data"
            )

    def _parse_pixel_data(self) -> Tuple[List[Tuple[int, int]], OffsetTableType]:
        """Parse file pixel data, reads frame positions.

        An EOT can be present before the pixel data, and must
        then not be empty. A BOT most always be the first item in the Pixel
        data, but can be empty (zero length). If EOT is used BOT should be empty.

        First seach to pixel data position, which is either EOT tag or PixelData tag.
        If EOT read the EOT. For all cases validate that the filepointer now is at the
        PixelData tag. If BOT read the BOT, otherwise skip the BOT. If EOT nor BOT has
        been read, parse frame positions from pixel data. Otherwise parse frame
        positions from EOT or BOT. Finaly check that the number of read frames equals
        the specified number of frames, otherwise frames are fragmented which we dont
        support.

        Returns
        ----------
        Tuple[List[Tuple[int, int]], OffsetTableType]
            List of frame positions and lengths, and table type.
        """
        table_type = self.offset_table_type
        table = None
        self._fp.seek(self._pixel_data_position)
        if table_type == OffsetTableType.EXTENDED:
            table = self._read_eot()

        self._validate_pixel_data_start(Tag(self._fp.read_tag()))
        if table_type == OffsetTableType.BASIC:
            table = self._read_bot()
        else:
            self._read_bot_length()

        if table is None:
            frame_positions = self._read_positions_from_pixeldata()
        else:
            frame_positions = self._parse_table(table, table_type, self._fp.tell())

        if self.frame_count != len(frame_positions):
            raise WsiDicomFileError(
                self.filepath,
                (
                    f"Frame count {self.frame_count} "
                    f"!= Fragments {len(frame_positions)}."
                    " Fragmented frames are not supported"
                ),
            )

        return frame_positions, table_type

    @staticmethod
    def filter_files(
        files: Iterable["WsiDicomFile"],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> List["WsiDicomFile"]:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: Iterable['WsiDicomFile']
            Wsi files to filter.
        series_uids: Uids
            Uids to check against.
        series_tile_size: Optional[Size] = None
            Tile size to check against.

        Returns
        ----------
        List['WsiDicomFile']
            List of matching wsi dicom files.
        """
        valid_files: List[WsiDicomFile] = []

        for file in files:
            if file.dataset.matches_series(series_uids, series_tile_size):
                valid_files.append(file)
            else:
                warnings.warn(
                    f"{file.filepath} with uids {file.uids.slide} "
                    f"did not match series with {series_uids} "
                    f"and tile size {series_tile_size}"
                )
                file.close()

        return valid_files

    @classmethod
    def group_files(
        cls, files: Iterable["WsiDicomFile"]
    ) -> Dict[str, List["WsiDicomFile"]]:
        """Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: Iterable[WsiDicomFile]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomFile]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomFile]] = defaultdict(list)
        for file in files:
            grouped_files[file.uids.identifier].append(file)
        return grouped_files
