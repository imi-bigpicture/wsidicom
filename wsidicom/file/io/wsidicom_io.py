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
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from datetime import datetime
from struct import pack
from typing import Any, BinaryIO, ClassVar, NamedTuple

from fsspec.implementations.local import LocalFileSystem
from pydicom import DataElement, Dataset, FileMetaDataset
from pydicom.config import RAISE
from pydicom.dataelem import RawDataElement, convert_raw_data_element
from pydicom.errors import InvalidDicomError
from pydicom.filebase import DicomIO
from pydicom.filereader import (
    _is_implicit_vr,
    _read_file_meta_info,
    data_element_generator,
    read_partial,
    read_preamble,
)
from pydicom.filereader import read_dataset as read_elements
from pydicom.filewriter import write_dataset, write_file_meta_info, writers
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID
from pydicom.valuerep import VR
from upath import UPath

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.deferred_element import DeferredElement
from wsidicom.tags import (
    InstanceCreationDateTag,
    InstanceCreationTimeTag,
    MediaStorageSOPClassUIDTag,
    MediaStorageSOPInstanceUIDTag,
    TransferSyntaxUIDTag,
)


class StreamStart(NamedTuple):
    """What reading the start of a stream gives.

    Reading the file meta information is what finds where the dataset after it
    starts, so the two are found together and kept together.
    """

    file_meta_info: FileMetaDataset
    """File meta information of the stream."""

    dataset_position: int
    """Offset the dataset starts at, past the preamble and the file meta information."""


class WsiDicomIO:
    """Class for reading or writing DICOM WSI to stream."""

    UNDEFINED_LENGTH: ClassVar[int] = 0xFFFFFFFF
    """Length stated by an item or a sequence that is delimited instead."""

    def __init__(self, stream: BinaryIO, filepath: UPath, transfer_syntax: UID):
        """Create a stream over a DICOM file or buffer.

        Parameters
        ----------
        stream: BinaryIO
            Stream to use.
        filepath: UPath
            Path the stream is over. Used for error messages and for opening a
            second stream over the same file, so this is required.
        transfer_syntax: UID
            Transfer syntax the stream is read and written with.
        """
        self._stream = stream
        self._stream.seek(0)
        self._filepath = filepath
        self._dicom_io = DicomIO(self._stream)
        self._lock = threading.RLock()
        self._transfer_syntax = transfer_syntax
        self._dicom_io.is_little_endian = transfer_syntax.is_little_endian
        self._dicom_io.is_implicit_VR = transfer_syntax.is_implicit_VR
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._filepath})"

    @contextmanager
    def exclusive(self) -> Generator[None, None, None]:
        """Hold the stream, for a read or a write that seeks before it.

        The position is the stream's own, so two of those at once would each move
        the other's. Re-entrant, so that one can be made of others without every
        one of them having to know which of them holds it.
        """
        with self._lock:
            yield

    @property
    def closed(self) -> bool:
        """Return True if the stream is closed."""
        return self._stream.closed

    @property
    def filepath(self) -> UPath:
        """Return the filepath the stream is backed by."""
        return self._filepath

    @property
    def transfer_syntax(self) -> UID:
        """Return the transfer syntax the stream is read and written with."""
        return self._transfer_syntax

    @contextmanager
    def buffered(self, buffer_bytes: int) -> Generator[BinaryIO, None, None]:
        """Yield a stream over the same file that reads it a block at a time.

        The stream this instance holds is yielded as it is when the file is not a
        local one, since nothing else can be opened for it, and a stream that fetches
        over a network has its own idea of how much to ask for at a time.

        Parameters
        ----------
        buffer_bytes: int
            Bytes to read at a time.

        Yields
        ------
        BinaryIO
            Stream over the same file, to read and seek within.
        """
        if (
            not isinstance(self._filepath.fs, LocalFileSystem)
            or not self._filepath.is_file()
        ):
            yield self._stream
            return
        size = self._filepath.stat().st_size
        buffer_bytes = max(min(buffer_bytes, size), 1)
        with open(str(self._filepath), "rb", buffering=buffer_bytes) as buffered:
            yield buffered

    @property
    def seek(self):
        return self._stream.seek

    @property
    def tell(self):
        return self._stream.tell

    @property
    def parent_read(self):
        return self._stream.read

    @property
    def is_little_endian(self):
        return self._dicom_io.is_little_endian

    @property
    def is_implicit_VR(self):
        return self._dicom_io.is_implicit_VR

    @property
    def stream(self):
        return self._stream

    def read(self, size: int, need_exact_length: bool = False) -> bytes:
        """Read bytes from stream."""
        data = self._stream.read(size)
        if need_exact_length and len(data) != size:
            raise EOFError()
        return data

    def read_tag(self) -> BaseTag:
        """Read tag from stream."""
        return Tag(self._dicom_io.read_tag())

    def read_tag_length(self, long: bool) -> int:
        """Read tag length."""
        if not long and not self._dicom_io.is_implicit_VR:
            return self._dicom_io.read_US()
        return self._dicom_io.read_UL()

    def read_tag_vr(self) -> bytes | None:
        """Read tag VR if implicit VR."""
        if not self._dicom_io.is_implicit_VR:
            vr = self.stream.read(4)
            return vr[0:2]
        return None

    def read_UL(self) -> int:
        """Read unsigned long integer (32 bits)."""
        return self._dicom_io.read_UL()

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
            read_tag = self._dicom_io.read_tag()
            if tag != read_tag:
                raise WsiDicomFileError(
                    str(self), f"Found tag {read_tag} expected {tag}."
                )
            if with_vr:
                if self._dicom_io.is_implicit_VR:
                    raise WsiDicomFileError(str(self), "Expected VR, but implicit VR.")
                self.read_tag_vr()
            read_length = self.read_tag_length(long)
            if length != read_length:
                raise WsiDicomFileError(
                    str(self), f"Found length {read_length} expected {length}."
                )
        except struct.error:
            raise WsiDicomFileError(str(self), "Failed to unpack data.") from None

    def close(self) -> None:
        """Close stream."""
        self._stream.close()


class WsiDicomReadIO(WsiDicomIO):
    """Stream a DICOM file is read from.

    The file meta information states what the transfer syntax is, so it is read when
    the stream is opened, which is also what finds where the dataset after it starts.
    """

    def __init__(self, stream: BinaryIO, filepath: UPath):
        """Create a stream to read a DICOM file over.

        Parameters
        ----------
        stream: BinaryIO
            Stream to use.
        filepath: UPath
            Path the stream is over. Used for error messages and for opening a
            second stream over the same file, so this is required.
        """
        self._stream_start = self._read_stream_start(stream, filepath)
        super().__init__(
            stream,
            filepath,
            UID(self._stream_start.file_meta_info.TransferSyntaxUID),
        )

    @property
    def stream_start(self) -> StreamStart:
        """What reading the start of the stream gave when it was opened."""
        return self._stream_start

    @staticmethod
    def _read_stream_start(stream: BinaryIO, filepath: UPath) -> StreamStart:
        """Read the preamble and the file meta information from the start of a stream.

        The stream is left where it was. Read before the stream is one of these, what
        it holds being what states how the rest of the stream is to be read.

        Parameters
        ----------
        stream: BinaryIO
            Stream to read.
        filepath: UPath
            Path the stream is over, for the error message.

        Returns
        -------
        StreamStart
            File meta information of the stream, and where the dataset starts.
        """
        stream.seek(0)
        try:
            read_preamble(stream, False)
            return StreamStart(_read_file_meta_info(stream), stream.tell())
        except InvalidDicomError:
            raise WsiDicomFileError(
                str(filepath), "is not a DICOM file or stream."
            ) from None
        finally:
            stream.seek(0)

    @property
    def is_dicom(self):
        rewind = self.tell()
        self.seek(0)
        self.stream.read(128)  # preamble
        is_dicom = self.stream.read(4) == b"DICM"
        self.seek(rewind)
        return is_dicom

    @property
    def media_storage_sop_class_uid(self) -> UID:
        """Read Media Storage SOP Class UID from file meta info."""
        return self.file_meta_info.MediaStorageSOPClassUID

    @property
    def file_meta_info(self) -> FileMetaDataset:
        """Read file meta info from stream."""
        return self.stream_start.file_meta_info

    def read_dataset(self, force: bool = False) -> Dataset:
        """Read the entire dataset from the stream.

        Parameters
        ----------
        force: bool = False
            Read the dataset even if the stream is not a valid DICOM stream.

        Returns
        -------
        Dataset
            The entire dataset read from the stream.
        """
        self.seek(0)
        return read_partial(
            self._stream,
            defer_size=None,
            force=force,
        )

    def read_elements_until(
        self, stop_tag: BaseTag, into: dict[BaseTag, DataElement | RawDataElement]
    ) -> BaseTag | None:
        """Read from the start of the stream into `into`, stopping at `stop_tag`.

        For a dataset read in parts, where the parts are made into one dataset by
        :func:`make_dataset` once the last of them has been read. The elements are
        kept as they were read, so that their values are made against the character
        set of the whole rather than against the one the part holding them had.

        Parameters
        ----------
        stop_tag: BaseTag
            First tag not to read.
        into: dict[BaseTag, DataElement | RawDataElement]
            Elements read so far, which the elements read here are added to.

        Returns
        -------
        BaseTag | None
            The tag the read stopped at, or None if the stream ended before a tag
            ordered at or after `stop_tag`.
        """
        return self.read_elements_from(
            self.stream_start.dataset_position, stop_tag, into
        )

    def read_elements_from(
        self,
        position: int,
        stop_tag: BaseTag,
        into: dict[BaseTag, DataElement | RawDataElement],
    ) -> BaseTag | None:
        """Read elements from `position` into `into`, stopping at `stop_tag`.

        For carrying on a read of a dataset read in parts. There is no header to read
        past, the elements being read straight from `position`.

        Parameters
        ----------
        position: int
            Offset of the first element to read.
        stop_tag: BaseTag
            First tag not to read.
        into: dict[BaseTag, DataElement | RawDataElement]
            Elements read so far, which the elements read here are added to.

        Returns
        -------
        BaseTag | None
            The tag the read stopped at, or None if the stream ended before a tag
            ordered at or after `stop_tag`.
        """
        stopped_at: BaseTag | None = None

        def _stop_at(tag: BaseTag, vr: str | None, length: int) -> bool:
            nonlocal stopped_at
            if tag < stop_tag:
                return False
            stopped_at = tag
            return True

        self.seek(position)
        # What the stream states it is need not be what it is, so the first bytes
        # are looked at the way pydicom looks at them when it reads a dataset.
        is_implicit_value_representation = _is_implicit_vr(
            self._stream,
            self._dicom_io.is_implicit_VR,
            self._dicom_io.is_little_endian,
            _stop_at,
            is_sequence=False,
        )
        self.seek(position)
        for element in data_element_generator(
            self._stream,
            is_implicit_value_representation,
            self._dicom_io.is_little_endian,
            stop_when=_stop_at,
        ):
            into[element.tag] = element
        return stopped_at

    def read_dataset_from(
        self, position: int, stop_tag: BaseTag, into: Dataset | None = None
    ) -> Dataset:
        """Read elements from `position` onwards into `into`, stopping at `stop_tag`.

        For continuing a read that stopped early. Unlike :func:`read_dataset` and
        :func:`read_dataset_until` this reads elements and nothing else: the preamble
        and file meta information are not read again, so there is nothing to carry a
        file name and its file meta information. There is no header to read past, so
        there is nothing for a force flag to force.

        The elements go into the dataset given, so that a dataset read in parts is
        one dataset and not several, or into one of their own when none is given.
        They are put in as they were read, so what they hold is made of them against
        the character set of the dataset they end up in rather than against one of
        their own, which a part read after the one stating it would not have.

        Parameters
        ----------
        position: int
            Offset of the first element to read.
        stop_tag: BaseTag
            First tag not to read.
        into: Dataset | None = None
            Dataset to read the elements into, or None for a dataset of their own.

        Returns
        -------
        Dataset
            The dataset the elements were read into.
        """

        def _stop_at(tag: BaseTag, vr: str | None, length: int) -> bool:
            return tag >= stop_tag

        self.seek(position)
        dataset = read_elements(
            self._stream,
            self._dicom_io.is_implicit_VR,
            self._dicom_io.is_little_endian,
            stop_when=_stop_at,
        )
        if into is None:
            return dataset
        for element in dataset.elements():
            into[element.tag] = element
        return into

    def read_sequence(
        self, position: int, defer_size: int
    ) -> tuple[DicomSequence, list[DeferredElement], int]:
        """Read the sequence at `position`, deferring values above `defer_size`.

        Parameters
        ----------
        position: int
            Offset of the tag of the sequence.
        defer_size: int
            Values longer than this many bytes are deferred.

        Returns
        -------
        tuple[DicomSequence, list[DeferredElement], int]
            The sequence, the deferred elements, and the offset just past the
            sequence.
        """
        self.seek(position)
        tag = self.read_tag()
        value_representation = self.read_tag_vr()
        if value_representation not in (None, b"SQ"):
            raise WsiDicomFileError(
                str(self),
                f"Expected a sequence at {position}, found {tag} with value "
                f"representation {value_representation!r}",
            )
        length = self.read_UL()
        if length == self.UNDEFINED_LENGTH:
            length = None
        items: list[Dataset] = []
        deferred: list[DeferredElement] = []
        end_of_sequence = None if length is None else self.tell() + length
        # End at stated length or when at the sequence delimiter
        while end_of_sequence is None or self.tell() < end_of_sequence:
            item_tag = self.read_tag()
            if item_tag == SequenceDelimiterTag:
                # The delimiter states a length of its own, always zero. Reading
                # past it is what puts the end where the sequence really ends.
                self.read_UL()
                end_of_sequence = self.tell()
                break
            if item_tag != ItemTag:
                raise WsiDicomFileError(
                    str(self),
                    f"Expected an item at {self.tell() - 4}, found {item_tag}",
                )
            item_length = self.read_UL()
            if item_length == self.UNDEFINED_LENGTH:
                item_length = None
            item = read_elements(
                self._stream,
                self._dicom_io.is_implicit_VR,
                self._dicom_io.is_little_endian,
                bytelength=item_length,
                defer_size=defer_size,
                at_top_level=False,
            )
            deferred.extend(self._take_deferred_elements(item))
            items.append(item)
        sequence = DicomSequence(items)
        sequence.is_undefined_length = length is None
        return sequence, deferred, self.tell()

    def _take_deferred_elements(self, dataset: Dataset) -> Iterable[DeferredElement]:
        """Take out the elements `dataset` holds whose values were deferred.

        Parameters
        ----------
        dataset: Dataset
            Dataset to take the elements out of.

        Returns
        -------
        Iterable[DeferredElement]
            One per value taken out.
        """
        elements = (
            dataset.get_item(tag, keep_deferred=True) for tag in list(dataset.keys())
        )
        deferred_elements = (
            element
            for element in elements
            if isinstance(element, RawDataElement)
            and element.value is None
            and element.length
        )
        for element in deferred_elements:
            del dataset[element.tag]
            yield DeferredElement(
                dataset,
                element.tag,
                element.VR,
                element.value_tell,
                element.length,
                self._dicom_io.is_implicit_VR,
                self._dicom_io.is_little_endian,
            )


class WsiDicomWriteIO(WsiDicomIO):
    """Stream a DICOM file is written to.

    Opened with the transfer syntax stated, there being nothing to read it from in a
    file that has not been written yet.
    """

    @property
    def write(self) -> Callable[[bytes], int]:
        return self._stream.write

    def write_unsigned_long_long(self, value: int):
        """Write unsigned long long integer (64 bits).

        Parameters
        ----------
        value: int
            Value to write.

        """
        format = "<Q" if self._dicom_io.is_little_endian else ">Q"
        self.write(pack(format, value))

    def write_tag(self, tag: BaseTag):
        """Write tag to stream.

        Parameters
        ----------
        tag: BaseTag
            Tag to write.

        """
        self._dicom_io.write_tag(tag)

    def write_UL(self, value: int):
        """Write unsigned long integer (32 bits).

        Parameters
        ----------
        value: int
            Value to write.

        """
        self._dicom_io.write_UL(value)

    def write_tag_of_vr_and_length(
        self, tag: BaseTag, value_representation: str, length: int | None = None
    ):
        """Write tag, tag VR and length.

        Parameters
        ----------
        tag: str
            Name of tag to write.
        value_representation: str.
            Value representation (VR) of tag to write.
        length: int | None = None
            Length of data after tag. 'Unspecified' (0xFFFFFFFF) if None.

        """
        self._dicom_io.write_tag(Tag(tag))
        if not self._dicom_io.is_implicit_VR:
            self.write(bytes(value_representation, "iso8859"))
            self._dicom_io.write_US(0)
        if length is not None:
            self._dicom_io.write_UL(length)
        else:
            self._dicom_io.write_UL(0xFFFFFFFF)

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
        meta = FileMetaDataset(
            {
                MediaStorageSOPClassUIDTag: DataElement(
                    MediaStorageSOPClassUIDTag,
                    "UI",
                    sop_class_uid,
                    validation_mode=RAISE,
                ),
                MediaStorageSOPInstanceUIDTag: DataElement(
                    MediaStorageSOPInstanceUIDTag,
                    "UI",
                    instance_uid,
                    validation_mode=RAISE,
                ),
                TransferSyntaxUIDTag: DataElement(
                    TransferSyntaxUIDTag, "UI", transfer_syntax, validation_mode=RAISE
                ),
            }
        )
        write_file_meta_info(self._dicom_io, meta)

    def write_dataset(self, dataset: Dataset, creation_datetime: datetime):
        """Write dataset to stream.

        Parameters
        ----------
        dataset: Dataset
            Dataset to write.
        creation_datetime: datetime
            Time this instance is created, written as InstanceCreationDate/Time.
            ContentDate/Time describe the image content and are left untouched.

        """
        creation_date = DataElement(
            InstanceCreationDateTag, "DA", creation_datetime.date()
        )
        creation_time = DataElement(
            InstanceCreationTimeTag, "TM", creation_datetime.time()
        )
        dataset.add(creation_date)
        dataset.add(creation_time)
        write_dataset(self._dicom_io, dataset)

    def update_dataset(self, dataset_start: int, update: dict[BaseTag, Any]):
        """Update dataset in place.

        The element value representation must allow padding,
        and values to replace should be padded to the same length or longer than the
        replacement value.


        Parameters
        ----------
        dataset_start: int
            Position of dataset start.
        update: dict[BaseTag, Any]
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
            if element.tag not in update:
                # generator can include `Specific Character Set` element
                continue
            if not isinstance(element, RawDataElement):
                raise ValueError("Can only update raw data elements.")
            element_value_position = element.value_tell
            length = element.length
            if isinstance(element, RawDataElement):
                element = convert_raw_data_element(element)
            element.value = update[element.tag]
            self.seek(element_value_position)
            writer, param = writers[VR(element.VR)]
            if param is None:
                param = []
            writer(self, element, *param)
            if self.tell() > element_value_position + length:
                raise ValueError("Updated element is longer than original.")
        self.seek(rewind)
