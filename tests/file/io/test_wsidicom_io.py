#    Copyright 2023 SECTRA AB
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

import datetime
import struct
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import pytest
from pydicom import DataElement, Dataset
from pydicom.dataset import FileMetaDataset
from pydicom.filebase import DicomFileLike
from pydicom.filereader import read_preamble, _read_file_meta_info
from pydicom.filewriter import write_file_meta_info
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag
from pydicom.uid import (
    JPEG2000,
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
    generate_uid,
)

from wsidicom.errors import WsiDicomFileError
from wsidicom.tags import (
    LossyImageCompressionRatioTag,
    PixelDataTag,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.uid import WSI_SOP_CLASS_UID


@pytest.fixture
def temp_path(tmpdir):
    yield Path(tmpdir).joinpath("test.dcm")


@pytest.fixture
def buffer():
    yield BytesIO()


@pytest.fixture
def buffer_with_preamble(buffer: BinaryIO):
    preamble = b"\x00" * 128
    buffer.write(preamble)
    buffer.write(b"DICM")
    yield buffer


@pytest.fixture
def buffer_with_file_meta(
    buffer_with_preamble: BinaryIO, file_meta_dataset: FileMetaDataset
):
    write_file_meta_info(DicomFileLike(buffer_with_preamble), file_meta_dataset)
    yield buffer_with_preamble


@pytest.fixture
def uid():
    yield generate_uid()


@pytest.fixture
def transfer_syntax():
    yield JPEGBaseline8Bit


@pytest.fixture
def sop_class_uid():
    yield WSI_SOP_CLASS_UID


@pytest.fixture
def file_meta_dataset(uid: UID, transfer_syntax: UID, sop_class_uid: UID):
    dataset = FileMetaDataset()
    dataset.TransferSyntaxUID = transfer_syntax
    dataset.MediaStorageSOPInstanceUID = uid
    dataset.MediaStorageSOPClassUID = sop_class_uid
    dataset.is_implicit_VR = transfer_syntax.is_implicit_VR
    dataset.is_little_endian = transfer_syntax.is_little_endian
    yield dataset


class TestWsiDicomIO:
    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    def test_open(self, temp_path: Path, transfer_syntax: UID):
        # Arrange

        # Act
        io = WsiDicomIO.open(
            temp_path,
            "w+b",
            transfer_syntax.is_little_endian,
            transfer_syntax.is_implicit_VR,
        )

        # Assert
        assert io.owned
        assert io.filepath == temp_path
        assert io.is_little_endian == transfer_syntax.is_little_endian
        assert io.is_implicit_VR == transfer_syntax.is_implicit_VR
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    @pytest.mark.parametrize("owned", [True, False])
    def test_init(self, buffer: BinaryIO, transfer_syntax: UID, owned: bool):
        # Arrange

        # Act
        io = WsiDicomIO(
            buffer,
            transfer_syntax.is_little_endian,
            transfer_syntax.is_implicit_VR,
            owned=owned,
        )

        # Assert
        assert io.owned == owned
        assert io.filepath is None
        assert io.is_little_endian == transfer_syntax.is_little_endian
        assert io.is_implicit_VR == transfer_syntax.is_implicit_VR
        io.close()

    @pytest.mark.parametrize("owned", [True, False])
    def test_close(self, buffer: BinaryIO, owned: bool):
        # Arrange
        io = WsiDicomIO(buffer, owned=owned)

        # Act
        io.close()

        # Assert
        assert io.closed == owned

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    def test_read_media_storage_sop_class_uid(
        self, buffer_with_file_meta: BinaryIO, file_meta_dataset: FileMetaDataset
    ):
        # Arrange
        io = WsiDicomIO(buffer_with_file_meta)

        # Act
        uid = io.read_media_storage_sop_class_uid()

        # Assert
        assert uid == file_meta_dataset.MediaStorageSOPClassUID
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    def test_read_file_meta_info(
        self, buffer_with_file_meta: BinaryIO, file_meta_dataset: FileMetaDataset
    ):
        # Arrange
        io = WsiDicomIO(buffer_with_file_meta)

        # Act
        meta_info = io.read_file_meta_info()

        # Assert
        assert meta_info == file_meta_dataset
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    def test_read_dataset(
        self,
        buffer_with_file_meta: BinaryIO,
        transfer_syntax: UID,
    ):
        # Arrange
        dataset = Dataset()
        dataset.is_implicit_VR = transfer_syntax.is_implicit_VR
        dataset.is_little_endian = transfer_syntax.is_little_endian
        dataset.PatientID = "Test123"
        dataset.save_as(buffer_with_file_meta, write_like_original=True)
        io = WsiDicomIO(buffer_with_file_meta)

        # Act
        read_dataset = io.read_dataset()

        # Assert
        assert read_dataset == dataset
        io.close()

    @pytest.mark.parametrize("little_endian", [True, False])
    @pytest.mark.parametrize(
        ["length", "long"],
        [
            (0, True),
            (142, True),
            (4294967295, True),
            (0, False),
            (142, False),
            (65535, False),
        ],
    )
    def test_read_tag_length(
        self, buffer: BinaryIO, little_endian: bool, long: bool, length: int
    ):
        # Arrange
        if little_endian:
            format = "<"
        else:
            format = ">"
        if long:
            format += "L"
        else:
            format += "H"
        buffer.write(struct.pack(format, length))
        io = WsiDicomIO(buffer, little_endian=little_endian)
        pre_position = io.tell()
        if long:
            expected_read_length = 4
        else:
            expected_read_length = 2

        # Act
        read_length = io.read_tag_length(long)

        # Assert
        post_position = io.tell()
        assert read_length == length
        assert post_position == pre_position + expected_read_length
        io.close()

    @pytest.mark.parametrize("is_implicit_VR", [True, False])
    def test_read_tag_vr(self, buffer: BinaryIO, is_implicit_VR: bool):
        # Arrange
        vr = bytes("OB", "iso8859")
        buffer.write(vr)
        buffer.write(bytes([0, 0]))
        io = WsiDicomIO(buffer, implicit_vr=is_implicit_VR)
        pre_position = io.tell()

        # Act
        read_vr = io.read_tag_vr()

        # Assert
        post_position = io.tell()
        if is_implicit_VR:
            assert read_vr is None
            assert pre_position == post_position
        else:
            assert read_vr is not None
            assert read_vr == vr
            assert post_position == pre_position + 4
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            # ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    @pytest.mark.parametrize(
        ["tag", "vr", "length"], [(PixelDataTag, "OB", 10), (ItemTag, None, 20)]
    )
    def test_check_tag_and_length(
        self,
        buffer: BinaryIO,
        tag: BaseTag,
        vr: Optional[str],
        length: int,
        transfer_syntax: UID,
    ):
        # Arrange
        if transfer_syntax.is_little_endian:
            format = "<"
        else:
            format = ">"
        buffer.write(struct.pack(format + "H", tag.group))
        buffer.write(struct.pack(format + "H", tag.element))
        if not transfer_syntax.is_implicit_VR and vr is not None:
            buffer.write(bytes("OB", "iso8859"))
            buffer.write(bytes([0, 0]))
        buffer.write(struct.pack(format + "L", length))
        io = WsiDicomIO(
            buffer, transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
        )

        # Act & Assert
        io.check_tag_and_length(tag, length, vr is not None, True)
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            # ImplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    @pytest.mark.parametrize(
        ["tag", "vr", "length", "expected_tag", "expected_vr", "expected_length"],
        [
            (PixelDataTag, "OB", 10, ItemTag, "OB", 10),
            (PixelDataTag, "OB", 10, PixelDataTag, None, 10),
            (PixelDataTag, "OB", 10, PixelDataTag, "OB", 100),
            (ItemTag, None, 20, PixelDataTag, None, 20),
            (ItemTag, None, 20, ItemTag, "OB", 20),
            (ItemTag, None, 20, ItemTag, None, 100),
        ],
    )
    def test_check_tag_and_length_raises(
        self,
        buffer: BinaryIO,
        tag: BaseTag,
        vr: Optional[str],
        length: int,
        expected_tag: BaseTag,
        expected_vr: Optional[str],
        expected_length: int,
        transfer_syntax: UID,
    ):
        # Arrange
        if transfer_syntax.is_little_endian:
            format = "<"
        else:
            format = ">"
        buffer.write(struct.pack(format + "H", tag.group))
        buffer.write(struct.pack(format + "H", tag.element))
        if not transfer_syntax.is_implicit_VR and vr is not None:
            buffer.write(bytes("OB", "iso8859"))
            buffer.write(bytes([0, 0]))
        buffer.write(struct.pack(format + "L", length))
        io = WsiDicomIO(
            buffer, transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
        )

        # Act & Assert
        with pytest.raises(WsiDicomFileError):
            io.check_tag_and_length(
                expected_tag, expected_length, expected_vr is not None, True
            )
        io.close()

    def test_read_sequence_delimiter(self, buffer: BinaryIO):
        # Arrange
        buffer.write(struct.pack("<H", SequenceDelimiterTag.group))
        buffer.write(struct.pack("<H", SequenceDelimiterTag.element))
        io = WsiDicomIO(buffer)

        # Act & Assert
        io.read_sequence_delimiter()

    def test_read_sequence_delimiter_raises(self, buffer: BinaryIO):
        # Arrange
        buffer.write(struct.pack("<H", ItemTag.group))
        buffer.write(struct.pack("<H", ItemTag.element))
        io = WsiDicomIO(buffer)

        # Act & Assert
        with pytest.raises(WsiDicomFileError):
            io.read_sequence_delimiter()
        io.close()

    @pytest.mark.parametrize("little_endian", [True, False])
    @pytest.mark.parametrize("value", [0, 1, 2**64 - 1])
    def test_write_unsigned_long_long(
        self, buffer: BinaryIO, little_endian: bool, value: int
    ):
        # Arrange
        if little_endian:
            format = "<Q"
        else:
            format = ">Q"
        # buffer.write(struct.pack(format, value))
        io = WsiDicomIO(buffer, little_endian=little_endian)

        # Act
        io.write_unsigned_long_long(value)

        # Assert
        buffer.seek(0)
        read_value = struct.unpack(format, buffer.read(8))[0]
        assert read_value == value
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
        ],
    )
    @pytest.mark.parametrize(["tag", "vr", "length"], [(PixelDataTag, "OB", 10)])
    def test_write_tag_of_vr_and_lengthk_tag_and_length(
        self,
        buffer: BinaryIO,
        tag: BaseTag,
        vr: str,
        length: int,
        transfer_syntax: UID,
    ):
        # Arrange
        if transfer_syntax.is_little_endian:
            format = "<"
        else:
            format = ">"

        io = WsiDicomIO(
            buffer, transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
        )

        # Act
        io.write_tag_of_vr_and_length(tag, vr, length)

        # Assert
        buffer.seek(0)
        group = struct.unpack(format + "H", buffer.read(2))[0]
        element = struct.unpack(format + "H", buffer.read(2))[0]
        assert group == tag.group
        assert element == tag.element
        if not transfer_syntax.is_implicit_VR:
            read_vr = buffer.read(2).decode("iso8859")
            buffer.read(2)
            assert read_vr == vr
        read_length = struct.unpack(format + "L", buffer.read(4))[0]
        assert read_length == length
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            JPEGBaseline8Bit,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
        ],
    )
    def test_write_preamble(self, buffer: BinaryIO, transfer_syntax: UID):
        # Arrange
        io = WsiDicomIO(
            buffer, transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
        )

        # Act
        io.write_preamble()

        # Assert
        buffer.seek(0)
        read_preamble(buffer, False)
        io.close()

    @pytest.mark.parametrize(
        "transfer_syntax",
        [
            DeflatedExplicitVRLittleEndian,
            ExplicitVRBigEndian,
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
            JPEG2000,
            JPEG2000Lossless,
            JPEGBaseline8Bit,
            JPEGExtended12Bit,
            JPEGLosslessP14,
            JPEGLosslessSV1,
            JPEGLSLossless,
            JPEGLSNearLossless,
            RLELossless,
        ],
    )
    def test_write_meta(self, buffer: BinaryIO, transfer_syntax: UID):
        # Arrange
        instance_uid = generate_uid()
        class_uid = WSI_SOP_CLASS_UID
        io = WsiDicomIO(
            buffer, transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
        )

        # Act
        io.write_file_meta_info(instance_uid, class_uid, transfer_syntax)

        # Assert
        buffer.seek(0)
        file_meta = _read_file_meta_info(buffer)
        assert file_meta.TransferSyntaxUID == transfer_syntax
        assert file_meta.MediaStorageSOPInstanceUID == instance_uid
        assert file_meta.MediaStorageSOPClassUID == class_uid
        io.close()

    @pytest.mark.parametrize(
        ["original_values", "update_values"],
        [
            ([" " * 16], "2.0"),
            (["1", " " * 16], ["1", "2"]),
            (["1", " " * 16], ["1", "12.34"]),
            (["43.21", " " * 16], ["43.21", "12.34"]),
        ],
    )
    def test_update_dataset(
        self,
        buffer: BinaryIO,
        original_values: List[str],
        update_values: Union[str, List[str]],
    ):
        # Arrange
        io = WsiDicomIO(buffer)
        dataset = Dataset()
        dataset.add(DataElement(LossyImageCompressionRatioTag, "DS", original_values))
        io.write_dataset(dataset, datetime.datetime.now())

        # Act
        io.update_dataset(0, {LossyImageCompressionRatioTag: update_values})

        # Assert
        io.seek(0)
        read_dataset = io.read_dataset(True)
        updated_values = read_dataset.LossyImageCompressionRatio
        assert updated_values == update_values
