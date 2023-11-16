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
from functools import cached_property
from pathlib import Path
from struct import unpack
from typing import BinaryIO, List, Optional, Tuple, Union, cast

from pydicom.errors import InvalidDicomError
from pydicom.filebase import DicomFileLike
from pydicom.filereader import _read_file_meta_info, read_partial, read_preamble
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID, UncompressedTransferSyntaxes
from wsidicom.codec import Codec

from wsidicom.errors import WsiDicomFileError, WsiDicomNotSupportedError
from wsidicom.file.wsidicom_file_base import OffsetTableType, WsiDicomFileBase
from wsidicom.instance import ImageType, WsiDataset
from wsidicom.uid import FileUids


class WsiDicomFile(WsiDicomFileBase):
    """Represents a DICOM file (potentially) containing WSI image and metadata."""

    def __init__(
        self, stream: BinaryIO, filepath: Optional[Path] = None, owned: bool = False
    ):
        """
        Parse DICOM file in stream. If valid WSI type read required
        parameters. Parses frames in pixel data but does not read the frames.

        Parameters
        ----------
        stream: BinaryIO
            Stream to open.
        filepath: Optional[Path] = None
            Optional filepath of stream.
        owned: bool = False
            If the stream should be closed by this instance.
        """
        self._lock = threading.Lock()
        try:
            stream.seek(0)
            read_preamble(stream, False)
        except InvalidDicomError:
            raise WsiDicomFileError(stream, "is not a DICOM file.")
        file_meta = _read_file_meta_info(stream)
        self._transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)
        super().__init__(stream, filepath, owned)
        self._file.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._file.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR
        extended_offset_table_tag = Tag("ExtendedOffsetTable")

        def _stop_at(tag: BaseTag, vr: Optional[str], length: int) -> bool:
            return tag >= extended_offset_table_tag

        self._file.seek(0)
        dataset = read_partial(
            cast(BinaryIO, self._file),
            _stop_at,
            defer_size=None,
            force=False,
            specific_tags=None,
        )
        self._pixel_data_position = self._file.tell()

        self._image_type = WsiDataset.is_supported_wsi_dicom(dataset)
        if self._image_type is not None:
            self._dataset = WsiDataset(dataset)
        else:
            raise WsiDicomNotSupportedError(f"Non-supported file {self._file}.")
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

    def __str__(self) -> str:
        return self.pretty_str()

    @classmethod
    def open(cls, file: Path) -> "WsiDicomFile":
        """
        Open file in path as WsiDicomFile.
        """
        stream = open(file, "rb")
        return cls(stream, file, True)

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
        return self.dataset.uids

    @property
    def transfer_syntax(self) -> UID:
        """Return transfer syntax uid"""
        return self._transfer_syntax_uid

    @property
    def frame_offset(self) -> int:
        """Return frame offset (for concatenated file, 0 otherwise)"""
        return self.dataset.frame_offset

    @cached_property
    def frame_positions(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths"""
        (frame_positions, offset_table_type) = self._parse_pixel_data()
        self._offset_table_type = offset_table_type
        return frame_positions

    @property
    def frame_count(self) -> int:
        """Return number of frames"""
        return self.dataset.frame_count

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
        return self._file, frame_position, frame_length

    def _get_offset_table_type(self) -> OffsetTableType:
        """
        Parse file for basic (BOT) or extended offset table (EOT). Return if file has
        EOT and do not check that the BOT is empty (it should be empty according to
        specifications, but we do not care if it not).
        """
        if self.transfer_syntax in UncompressedTransferSyntaxes:
            return OffsetTableType.NONE
        self._file.seek(self._pixel_data_position)
        pixel_data_or_eot_tag = Tag(self._file.read_tag())
        if pixel_data_or_eot_tag == Tag("ExtendedOffsetTable"):
            self._read_tag_vr()
            eot_length = self._read_tag_length()
            self._file.seek(eot_length, 1)
            self._read_eot_lengths_tag()
            return OffsetTableType.EXTENDED
        self._validate_pixel_data_start(pixel_data_or_eot_tag, False)
        bot_length = self._read_bot_length()
        if bot_length is not None:
            return OffsetTableType.BASIC
        return OffsetTableType.EMPTY

    def _read_bot_length(self) -> Optional[int]:
        """Read the length of the basic table offset (BOT). Returns None if BOT
        is empty.

        Returns
        ----------
        Optional[int]
            BOT length.
        """
        BOT_BYTES = 4
        if self._file.read_tag() != ItemTag:
            raise WsiDicomFileError(
                self._file, "Basic offset table did not start with an ItemTag"
            )
        bot_length = self._file.read_UL()
        if bot_length == 0:
            return None
        elif bot_length % BOT_BYTES:
            raise WsiDicomFileError(
                self._file,
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
        bot = self._file.read(bot_length)
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
                self._file, "Expected Extended offset table present but empty"
            )
        elif eot_length % EOT_BYTES:
            raise WsiDicomFileError(
                self._file,
                "Extended offset table should be a multiple of " f"{EOT_BYTES} bytes",
            )
        return eot_length

    def _read_eot_lengths_tag(self):
        """Skip over the length of the extended table offset lengths tag."""
        eot_lenths_tag = self._file.read_tag()
        if eot_lenths_tag != Tag("ExtendedOffsetTableLengths"):
            raise WsiDicomFileError(
                self._file,
                "Expected Extended offset table lengths tag after reading "
                f"Extended offset table, found {eot_lenths_tag}",
            )
        self._read_tag_vr()
        length = self._read_tag_length()
        # Jump over EOT lengths for now
        self._file.seek(length, 1)

    def _read_eot(self) -> bytes:
        """Read extended table offset (EOT) and EOT lengths. Filepointer should be
        positionend to extended offset table.

        Returns
        ----------
        bytes
            EOT in bytes.
        """
        eot_tag = Tag(self._file.read_tag())
        if eot_tag != Tag("ExtendedOffsetTable"):
            raise ValueError(f"Expected ExtendedOffsetTable tag, got {eot_tag}")
        self._read_tag_vr()
        eot_length = self._read_eot_length()
        # Read the EOT into bytes
        eot = self._file.read(eot_length)
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
        if not self._file.is_little_endian:
            raise WsiDicomFileError(
                self._file, "Big endian not supported for BOT or EOT"
            )

        if table_type == OffsetTableType.BASIC:
            bytes_per_item = 4
            mode = "<L"
        elif table_type == OffsetTableType.EXTENDED:
            bytes_per_item = 8
            mode = "<Q"
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
                raise WsiDicomFileError(self._file, "Invalid frame length")
            positions.append((pixels_start + offset, length))
            this_offset = next_offset

        # Go to last frame in pixel data and read the length of the frame
        self._file.seek(pixels_start + this_offset)
        if self._file.read_le_tag() != ItemTag:
            raise WsiDicomFileError(self._file, "Expected ItemTag in PixelData")
        length: int = self._file.read_leUL()
        if length == 0 or length % 2:
            raise WsiDicomFileError(self._file, "Invalid frame length")
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
        frame_position = self._file.tell()
        # Read items until sequence delimiter
        while self._file.read_le_tag() == ItemTag:
            # Read item length
            length: int = self._file.read_leUL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(self._file, "Invalid frame length")
            positions.append((frame_position + TAG_BYTES + LENGTH_BYTES, length))
            # Jump to end of frame
            self._file.seek(length, 1)
            frame_position = self._file.tell()
        self._read_sequence_delimiter()
        return positions

    def _create_frame_positions_for_uncapsulated_data(
        self, offset: int
    ) -> List[Tuple[int, int]]:
        """Create frame positions for uncapsulated data.

        Parameters
        ----------
        offset: int
            Offset to first frame in pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            A list with frame positions and frame lengths.
        """
        frame_size = (
            self.dataset.tile_size.area
            * self.dataset.samples_per_pixel
            * (self.dataset.bits // 8)
        )
        return [
            (offset + index * frame_size, frame_size)
            for index in range(self.frame_count)
        ]

    def _read_sequence_delimiter(self):
        """Check if last read tag was a sequence delimiter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._file.seek(-TAG_BYTES, 1)
        if self._file.read_le_tag() != SequenceDelimiterTag:
            raise WsiDicomFileError(self._file, "No sequence delimiter tag")

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

    def _validate_pixel_data_start(
        self, tag: Union[BaseTag, Tuple[int, int]], defined_length: bool
    ):
        """Check that pixel data tag is present and that the tag length is
        set as undefined. Raises WsiDicomFileError otherwise.

        Parameters
        ----------
        tag: Union[BaseTag, Tuple[int, int]]
            Tag that should be pixel data tag.
        defined_length: bool
            If length of pixel data should be defined.
        """
        if tag != Tag("PixelData"):
            WsiDicomFileError(self._file, "Expected PixelData tag")
        self._read_tag_vr()
        length = self._read_tag_length()
        if defined_length:
            expected_length = (
                self.dataset.tile_size.area
                * self.dataset.samples_per_pixel
                * (self.dataset.bits // 8)
                * self.frame_count
            )
            if not length == expected_length:
                raise WsiDicomFileError(
                    self._file,
                    f"Expected {expected_length} length when reading Pixel data, got {length}.",
                )
        else:
            if length != 0xFFFFFFFF:
                raise WsiDicomFileError(
                    self._file,
                    f"Expected undefined length when reading Pixel data, got {length}.",
                )

    def _parse_pixel_data(self) -> Tuple[List[Tuple[int, int]], OffsetTableType]:
        """Parse file pixel data, reads frame positions.

        An EOT can be present before the pixel data, and must
        then not be empty. A BOT most always be the first item in the Pixel
        data, but can be empty (zero length). If EOT is used BOT should be empty.

        First search to pixel data position, which is either EOT tag or PixelData tag.
        If EOT read the EOT. For all cases validate that the filepointer now is at the
        PixelData tag. If BOT read the BOT, otherwise skip the BOT. If EOT nor BOT has
        been read, parse frame positions from pixel data. Otherwise parse frame
        positions from EOT or BOT. Finally check that the number of read frames equals
        the specified number of frames, otherwise frames are fragmented which we dont
        support.

        Returns
        ----------
        Tuple[List[Tuple[int, int]], OffsetTableType]
            List of frame positions and lengths, and table type.
        """
        table_type = self.offset_table_type
        table = None
        self._file.seek(self._pixel_data_position)
        if table_type == OffsetTableType.EXTENDED:
            table = self._read_eot()

        self._validate_pixel_data_start(
            Tag(self._file.read_tag()), table_type is OffsetTableType.NONE
        )
        if table_type == OffsetTableType.BASIC:
            table = self._read_bot()
        elif table_type == OffsetTableType.EMPTY:
            self._read_bot_length()

        if table_type == OffsetTableType.NONE:
            frame_positions = self._create_frame_positions_for_uncapsulated_data(
                self._file.tell()
            )
        elif table is None:
            frame_positions = self._read_positions_from_pixeldata()
        else:
            frame_positions = self._parse_table(table, table_type, self._file.tell())

        if len(frame_positions) < self.frame_count:
            raise WsiDicomFileError(
                self._file,
                (
                    f"ImageData contained less frames {len(frame_positions)} than "
                    f"NumberOfFrames {self.frame_count}."
                ),
            )
        if len(frame_positions) > self.frame_count:
            raise WsiDicomFileError(
                self._file,
                (
                    f"ImageData contained more fragments {len(frame_positions)} than "
                    f"NumberOfFrames {self.frame_count} and fragmented frames are not "
                    "supported."
                ),
            )

        return frame_positions, table_type
