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

"""Module with base IO class for handling DICOM WSI files."""

import struct
from datetime import datetime
from pathlib import Path
from struct import pack
from typing import Any, BinaryIO, Callable, Dict, Literal, Optional, Union

from pydicom.dataelem import DataElement_from_raw, RawDataElement
from pydicom.dataset import Dataset, FileMetaDataset, validate_file_meta
from pydicom.filebase import DicomIO
from pydicom.filereader import (
    _read_file_meta_info,
    data_element_generator,
    read_partial,
    read_preamble,
)
from pydicom.filewriter import write_dataset, write_file_meta_info, writers
from pydicom.tag import BaseTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID
from pydicom.valuerep import VR

from wsidicom.errors import WsiDicomFileError
from wsidicom.tags import ExtendedOffsetTableTag


class WsiDicomIO(DicomIO):
    """Class for reading or writing DICOM WSI to stream."""

    def __init__(
        self,
        stream: BinaryIO,
        little_endian: bool = True,
        implicit_vr: bool = False,
        filepath: Optional[Path] = None,
        owned: bool = False,
    ):
        """
        Create a WsiDicomIO.

        Parameters
        ----------
        stream: BinaryIO
            Stream to use.
        little_endian: bool = True
            If to set the stream to little endian.
        implicit_vr: bool = False
            If to set the stream to implicit VR.
        filepath: Optional[Path] = None
            Optional filepath of stream.
        owned: bool = False
            If the stream should be closed by this instance.
        """
        stream.seek(0)
        self._stream = stream
        self._filepath = filepath
        self._owned = owned
        super().__init__(stream)
        self.is_little_endian = little_endian
        self.is_implicit_VR = implicit_vr
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def open(
        cls,
        filepath: Path,
        mode: Union[Literal["rb"], Literal["r+b"], Literal["w+b"]],
        little_endian: bool = True,
        implicit_vr: bool = False,
    ) -> "WsiDicomIO":
        """Open a file and return a WsiDicomIO instance.

        Parameters
        ----------
        filepath: Path
            Path to file.
        mode: Union[Literal["rb"], Literal["w+b"]]
            Mode to open file in.
        little_endian: bool = True
            If to set the stream to little endian.
        implicit_vr: bool = False
            If to set the stream to implicit VR.

        Returns
        ----------
        WsiDicomIO
            Instance of WsiDicomIO.
        """
        stream = open(filepath, mode)
        return cls(stream, little_endian, implicit_vr, filepath=filepath, owned=True)

    @property
    def owned(self) -> bool:
        """Return True if the stream is owned by this instance."""
        return self._owned

    @property
    def closed(self) -> bool:
        """Return True if the stream is closed."""
        return self._stream.closed

    @property
    def filepath(self) -> Optional[Path]:
        """Return filepath, if opened from file."""
        return self._filepath

    @property
    def write(self) -> Callable[[bytes], int]:
        return self._stream.write

    @property
    def seek(self):
        return self._stream.seek

    @property
    def tell(self):
        return self._stream.tell

    @property
    def parent_read(self):
        return self._stream.read

    def read_media_storage_sop_class_uid(self) -> UID:
        """Read Media Storage SOP Class UID from file meta info."""
        metadata = self.read_file_meta_info()
        self.seek(0)
        return metadata.MediaStorageSOPClassUID

    def read_file_meta_info(self) -> FileMetaDataset:
        """Read file meta info from stream."""
        self.seek(0)
        read_preamble(self._stream, False)
        return _read_file_meta_info(self._stream)

    def read_dataset(self, force: bool = False) -> Dataset:
        """Read dataset, excluding EOT and PixelData from stream."""
        extended_offset_table_tag = ExtendedOffsetTableTag

        def _stop_at(tag: BaseTag, vr: Optional[str], length: int) -> bool:
            return tag >= extended_offset_table_tag

        self.seek(0)
        return read_partial(
            self._stream,
            _stop_at,
            defer_size=None,
            force=force,
            specific_tags=None,
        )

    def read_tag_length(self, long: bool) -> int:
        """Read tag length."""
        if not long and not self.is_implicit_VR:
            return self.read_US()
        return self.read_UL()

    def read_tag_vr(self) -> Optional[bytes]:
        """Read tag VR if implicit VR."""
        if not self.is_implicit_VR:
            vr = self.read(4, need_exact_length=True)
            return vr[0:2]

    def check_tag_and_length(
        self, tag: BaseTag, length: int, with_vr: bool, long: bool
    ) -> None:
        """Check if tag at position is expected tag with expected length.

        Parameters
        ----------
        tag: BaseTag
            Expected tag.
        length: int
            Expected length.
        with_vr: bool
            If tag is expected to have VR.
        long: bool
            If length is expected to be long.

        """
        try:
            read_tag = self.read_tag()
            if tag != read_tag:
                raise WsiDicomFileError(
                    str(self), f"Found tag {read_tag} expected {tag}."
                )
            if with_vr:
                if self.is_implicit_VR:
                    raise WsiDicomFileError(str(self), "Expected VR, but implicit VR.")
                self.read_tag_vr()
            read_length = self.read_tag_length(long)
            if length != read_length:
                raise WsiDicomFileError(
                    str(self), f"Found length {read_length} expected {length}."
                )
        except struct.error:
            raise WsiDicomFileError(str(self), "Failed to unpack data.")

    def read_sequence_delimiter(self):
        """Check if last read tag was a sequence delimiter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self.seek(-TAG_BYTES, 1)
        if self.read_le_tag() != SequenceDelimiterTag:
            raise WsiDicomFileError(str(self), "No sequence delimiter tag")

    def write_unsigned_long_long(self, value: int):
        """Write unsigned long long integer (64 bits).

        Parameters
        ----------
        value: int
            Value to write.

        """
        if self.is_little_endian:
            format = "<Q"
        else:
            format = ">Q"
        self.write(pack(format, value))

    def write_tag_of_vr_and_length(
        self, tag: BaseTag, value_representation: str, length: Optional[int] = None
    ):
        """Write tag, tag VR and length.

        Parameters
        ----------
        tag: str
            Name of tag to write.
        value_representation: str.
            Value representation (VR) of tag to write.
        length: Optional[int] = None
            Length of data after tag. 'Unspecified' (0xFFFFFFFF) if None.

        """
        if self.is_little_endian:
            write_ul = self.write_leUL
            write_us = self.write_leUS
        else:
            write_ul = self.write_beUL
            write_us = self.write_beUS
        self.write_tag(Tag(tag))
        if not self.is_implicit_VR:
            self.write(bytes(value_representation, "iso8859"))
            write_us(0)
        if length is not None:
            write_ul(length)
        else:
            write_ul(0xFFFFFFFF)

    def write_preamble(self):
        """Write DICOM preamble."""
        self.seek(0)
        preamble = b"\x00" * 128
        self.write(preamble)
        self.write(b"DICM")

    def write_file_meta_info(
        self, instance_uid: UID, sop_class_uid: UID, transfer_syntax: UID
    ):
        """Write file meta info.

        Parameters
        ----------
        instance_uid: UID
            SOP Instance UID.
        sop_class_uid: UID
            SOP Class UID.

        """
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = transfer_syntax
        meta.MediaStorageSOPInstanceUID = instance_uid
        meta.MediaStorageSOPClassUID = sop_class_uid
        validate_file_meta(meta)
        write_file_meta_info(self, meta)

    def write_dataset(self, dataset: Dataset, content_datetime: datetime):
        """Write dataset to stream.

        Parameters
        ----------
        dataset: Dataset
            Dataset to write.

        """
        dataset.ContentDate = datetime.date(content_datetime).strftime("%Y%m%d")
        dataset.ContentTime = datetime.time(content_datetime).strftime("%H%M%S.%f")
        write_dataset(self, dataset)

    def close(self, force: Optional[bool] = False) -> None:
        """Close stream if owned by instance or forced."""
        if self._owned or force:
            self._stream.close()

    def update_dataset(self, dataset_start: int, update: Dict[BaseTag, Any]):
        """Update dataset in place.

        The element value representation must allow padding,
        and values to replace should be padded to the same length or longer than the
        replacement value.


        Parameters
        ----------
        dataset_start: int
            Position of dataset start.
        update: Dict[BaseTag, Any]
            Dictionary with tags to update and their new values.
        """

        rewind = self.tell()
        self.seek(dataset_start)
        for element in data_element_generator(
            self._stream,
            self.is_implicit_VR,
            self.is_little_endian,
            specific_tags=list(update.keys()),
        ):
            if element.tag not in update.keys():
                # generator can include `Specific Character Set` element
                continue
            if not isinstance(element, RawDataElement):
                raise ValueError("Can only update raw data elements.")
            element_value_position = element.value_tell
            length = element.length
            if isinstance(element, RawDataElement):
                element = DataElement_from_raw(element)
            element.value = update[BaseTag(element.tag)]
            self.seek(element_value_position)
            writer, param = writers[VR(element.VR)]
            if param is None:
                param = []
            writer(self, element, *param)
            if self.tell() > element_value_position + length:
                raise ValueError("Updated element is longer than original.")
        self.seek(rewind)
