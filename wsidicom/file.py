import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom
from pydicom.uid import UID as Uid

from .errors import WsiDicomFileError
from .geometry import Size
from .uid import BaseUids, FileUids
from .dataset import WsiDataset


class WsiDicomFile:
    def __init__(self, filepath: Path):
        """Open dicom file in filepath. If valid wsi type read required
        parameters. Parses frames in pixel data but does not read the frames.

        Parameters
        ----------
        filepath: Path
            Path to file to open
        """
        self._filepath = filepath

        if not pydicom.misc.is_dicom(self.filepath):
            raise WsiDicomFileError(self.filepath, "is not a DICOM file")

        file_meta = pydicom.filereader.read_file_meta_info(self.filepath)
        self._transfer_syntax_uid = Uid(file_meta.TransferSyntaxUID)

        self._fp = pydicom.filebase.DicomFile(self.filepath, mode='rb')
        self._fp.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR

        dataset = pydicom.dcmread(self._fp, stop_before_pixels=True)

        if WsiDataset.is_wsi_dicom(dataset):
            self._pixel_data_position = self._fp.tell()
            self._dataset = WsiDataset(dataset)
            self._wsi_type = self.dataset.get_supported_wsi_dicom_type(
                self.transfer_syntax
            )
            instance_uid = self.dataset.instance_uid
            concatenation_uid = self.dataset.concatenation_uid
            base_uids = self.dataset.base_uids
            self._uids = FileUids(instance_uid, concatenation_uid, base_uids)
            self._frame_offset = self.dataset.frame_offset
            self._frame_count = self.dataset.frame_count
            self._frame_positions = self._parse_pixel_data()
        else:
            self._wsi_type = "None"

    def __repr__(self) -> str:
        return f"WsiDicomFile('{self.filepath}')"

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def dataset(self) -> WsiDataset:
        """Return pydicom dataset of file."""
        return self._dataset

    @property
    def filepath(self) -> Path:
        """Return filepath"""
        return self._filepath

    @property
    def wsi_type(self) -> str:
        return self._wsi_type

    @property
    def uids(self) -> FileUids:
        """Return uids"""
        return self._uids

    @property
    def transfer_syntax(self) -> Uid:
        """Return transfer syntax uid"""
        return self._transfer_syntax_uid

    @property
    def frame_offset(self) -> int:
        """Return frame offset (for concatenated file, 0 otherwise)"""
        return self._frame_offset

    @property
    def frame_positions(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths"""
        return self._frame_positions

    @property
    def frame_count(self) -> int:
        """Return number of frames"""
        return self._frame_count

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return f"File with path: {self.filepath}"

    def get_filepointer(
        self,
        frame_index: int
    ) -> Tuple[pydicom.filebase.DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame lenght for frame
        number.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        Tuple[pydicom.filebase.DicomFileLike, int, int]:
            File pointer, frame offset and frame lenght in number of bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_positions[frame_index]
        return self._fp, frame_position, frame_length

    def close(self) -> None:
        """Close the file."""
        self._fp.close()

    def _read_bot(self) -> None:
        """Read (skips over) basic table offset (BOT). The BOT can contain
        frame positions, and the BOT should always be present but does not need
        to contain any data (exept lenght), and sometimes the content is
        corrupt. As we in that case anyway need to check the validity of the
        BOT by checking the actual frame sequence, we just skip over it.
        """
        # Jump to BOT
        offset_to_value = pydicom.filereader.data_element_offset_to_value(
            self._fp.is_implicit_VR,
            'OB'
        )
        self._fp.seek(offset_to_value, 1)
        # has_BOT, offset_table = pydicom.encaps.get_frame_offsets(self._fp)
        # Read the BOT lenght and skip over the BOT
        if(self._fp.read_tag() != pydicom.tag.ItemTag):
            raise WsiDicomFileError(self.filepath, 'No item tag after BOT')
        length = self._fp.read_UL()
        self._fp.seek(length, 1)

    def _read_positions(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from sequence of frames that ends
        with a tag not equal to itemtag. fp needs to be positioned after the
        BOT.
        Each frame contains:
        item tag (4 bytes)
        item lenght (4 bytes)
        item data (item length)
        The position of item data and the item lenght is stored.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        TAG_BYTES = 4
        LENGHT_BYTES = 4
        positions: List[Tuple[int, int]] = []
        frame_position = self._fp.tell()
        # Read items until sequence delimiter
        while(self._fp.read_tag() == pydicom.tag.ItemTag):
            # Read item length
            length = self._fp.read_UL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, 'Invalid frame length')
            # Frame position
            position = frame_position
            positions.append((position+TAG_BYTES+LENGHT_BYTES, length))
            # Jump to end of item
            self._fp.seek(length, 1)
            frame_position = self._fp.tell()
        return positions

    def _read_sequence_delimeter(self):
        """Check if last read tag was a sequence delimter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._fp.seek(-TAG_BYTES, 1)
        if(self._fp.read_tag() != pydicom.tag.SequenceDelimiterTag):
            raise WsiDicomFileError(self.filepath, 'No sequence delimeter tag')

    def _read_frame_positions(self) -> List[Tuple[int, int]]:
        """Parse pixel data to get frame positions (relative to end of BOT)
        and frame lenght.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        self._read_bot()
        positions = self._read_positions()
        self._read_sequence_delimeter()
        self._fp.seek(self._pixel_data_position, 0)  # Wind back to start
        return positions

    def _read_frame(self, frame_index: int) -> bytes:
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
        fp.seek(frame_position, 0)
        frame: bytes = fp.read(frame_length)
        return frame

    def _parse_pixel_data(self) -> List[Tuple[int, int]]:
        """Parse file pixel data, reads frame positions.
        Note that fp needs to be positionend at pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            List of frame positions and lenghts
        """
        frame_positions = self._read_frame_positions()
        fragment_count = len(frame_positions)
        if(self.frame_count != len(frame_positions)):
            raise WsiDicomFileError(
                self.filepath,
                (
                    f"Frames {self.frame_count} != Fragments {fragment_count}."
                    " Fragmented frames are not supported"
                )
            )
        return frame_positions

    @staticmethod
    def _filter_files(
        files: List['WsiDicomFile'],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List['WsiDicomFile']:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: List['WsiDicomFile']
            Wsi files to filter.
        series_uids: Uids
            Uids to check against.
        series_tile_size: Size
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
                    f'{file.filepath} with uids {file.uids.base} '
                    f'did not match series with {series_uids} '
                    f'and tile size {series_tile_size}'
                )
                file.close()

        return valid_files

    @classmethod
    def _group_files(
        cls,
        files: List['WsiDicomFile']
    ) -> Dict[str, List['WsiDicomFile']]:
        """Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomFile]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomFile]] = {}
        for file in files:
            try:
                grouped_files[file.uids.identifier].append(file)
            except KeyError:
                grouped_files[file.uids.identifier] = [file]
        return grouped_files
