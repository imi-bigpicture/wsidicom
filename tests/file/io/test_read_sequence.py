#    Copyright 2026 SECTRA AB
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

"""Reading a sequence while deferring what is large in its items."""

from io import BytesIO

import pytest
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import write_dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import UID, ExplicitVRLittleEndian, ImplicitVRLittleEndian
from upath import UPath

from wsidicom.file.io.wsidicom_io import WsiDicomIO

OPTICAL_PATH_SEQUENCE_TAG = Tag(0x0048, 0x0105)
LARGE_VALUE = b"a profile" * 20000
DEFER_SIZE = 64 * 1024


def create_stream(
    sequence: Sequence,
    undefined_length: bool = True,
    transfer_syntax: UID = ExplicitVRLittleEndian,
) -> WsiDicomIO:
    """Write `sequence` as an optical path sequence at the start of a stream."""
    dataset = Dataset()
    dataset[OPTICAL_PATH_SEQUENCE_TAG] = DataElement(
        OPTICAL_PATH_SEQUENCE_TAG, "SQ", sequence
    )
    dataset[OPTICAL_PATH_SEQUENCE_TAG].is_undefined_length = undefined_length
    buffer = DicomBytesIO()
    buffer.is_little_endian = transfer_syntax.is_little_endian
    buffer.is_implicit_VR = transfer_syntax.is_implicit_VR
    write_dataset(buffer, dataset)
    return WsiDicomIO(
        BytesIO(buffer.getvalue()),
        filepath=UPath("optical_path.dcm"),
        transfer_syntax=transfer_syntax,
    )


def create_item(with_nested_sequence: bool = False) -> Dataset:
    """An optical path item holding a large value, and a large sequence if asked."""
    item = Dataset()
    item.OpticalPathIdentifier = "1"
    item.ICCProfile = LARGE_VALUE
    if with_nested_sequence:
        nested = Dataset()
        nested.ICCProfile = LARGE_VALUE
        item.IlluminationTypeCodeSequence = Sequence([nested])
    return item


@pytest.mark.unittest
class TestReadSequenceDeferringValues:
    @pytest.mark.parametrize("undefined_length", [True, False])
    def test_a_large_value_is_passed_over(self, undefined_length: bool):
        # Arrange
        stream = create_stream(Sequence([create_item()]), undefined_length)

        # Act
        sequence, deferred, _ = stream.read_sequence(0, DEFER_SIZE)

        # Assert
        assert [element.keyword for element in sequence[0]] == ["OpticalPathIdentifier"]
        assert [value.length for value in deferred] == [len(LARGE_VALUE)]

    @pytest.mark.parametrize("undefined_length", [True, False])
    def test_the_read_ends_just_past_the_sequence(self, undefined_length: bool):
        """The rest of the dataset is read from where this says the sequence ends,
        so saying it wrongly loses or repeats what follows."""
        # Arrange
        stream = create_stream(Sequence([create_item()]), undefined_length)
        length = len(stream.stream.getvalue())  # type: ignore[attr-defined]

        # Act
        _, _, end_of_sequence = stream.read_sequence(0, DEFER_SIZE)

        # Assert
        assert end_of_sequence == length

    @pytest.mark.parametrize(
        "transfer_syntax", [ExplicitVRLittleEndian, ImplicitVRLittleEndian]
    )
    def test_a_large_value_survives_an_implicit_value_representation(
        self, transfer_syntax: UID
    ):
        """A transfer syntax that states value representations implicitly writes none
        to read, so there is none to check the sequence by and none to read the value
        back with but the one the dictionary gives."""
        # Arrange
        stream = create_stream(
            Sequence([create_item()]), transfer_syntax=transfer_syntax
        )

        # Act
        sequence, deferred, _ = stream.read_sequence(0, DEFER_SIZE)
        for value in deferred:
            stream.seek(value.offset)
            value.set(stream.read(value.length, need_exact_length=True))

        # Assert
        assert sequence[0].ICCProfile == LARGE_VALUE

    def test_a_small_value_is_not_passed_over(self):
        # Arrange
        item = Dataset()
        item.OpticalPathIdentifier = "1"
        item.ICCProfile = b"small!"  # An even length, DICOM padding an odd one.
        stream = create_stream(Sequence([item]))

        # Act
        sequence, deferred, _ = stream.read_sequence(0, DEFER_SIZE)

        # Assert
        assert sequence[0].ICCProfile == b"small!"
        assert deferred == []

    def test_a_value_passed_over_is_put_back_as_it_was(self):
        # Arrange
        stream = create_stream(Sequence([create_item()]))
        sequence, deferred, _ = stream.read_sequence(0, DEFER_SIZE)

        # Act
        for value in deferred:
            stream.seek(value.offset)
            value.set(stream.read(value.length, need_exact_length=True))

        # Assert
        assert sequence[0].ICCProfile == LARGE_VALUE

    def test_a_large_nested_sequence_is_a_sequence_again_when_put_back(self):
        """What is deferred is chosen by length alone, so a sequence longer than
        the size read at is deferred whole and has to come back a sequence rather
        than the bytes it was deferred as."""
        # Arrange
        stream = create_stream(Sequence([create_item(with_nested_sequence=True)]))
        sequence, deferred, _ = stream.read_sequence(0, DEFER_SIZE)
        assert {value.value_representation for value in deferred} == {"SQ", "OB"}

        # Act
        for value in deferred:
            stream.seek(value.offset)
            value.set(stream.read(value.length, need_exact_length=True))

        # Assert
        nested = sequence[0].IlluminationTypeCodeSequence
        assert isinstance(nested, Sequence)
        assert nested[0].ICCProfile == LARGE_VALUE
        assert sequence[0].ICCProfile == LARGE_VALUE
