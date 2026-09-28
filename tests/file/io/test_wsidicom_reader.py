#    Copyright 2022, 2023 SECTRA AB
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

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from pydicom import Dataset, dcmread
from pydicom.dataset import FileMetaDataset
from pydicom.tag import Tag
from upath import UPath

from tests.data_gen import (
    TESTFRAME,
    create_layer_file,
    create_main_dataset,
    create_meta_dataset,
)
from wsidicom.errors import WsiDicomError, WsiDicomOutOfBoundsError
from wsidicom.file.io import OffsetTableType, WsiDicomReader
from wsidicom.file.io.wsidicom_io import WsiDicomReadIO
from wsidicom.instance import TileType, WsiDataset
from wsidicom.metadata import ImageType
from wsidicom.tags import PerFrameFunctionalGroupsSequenceTag

TAG_AFTER_SEQUENCE = Tag(0x5401, 0x0010)
"""A tag ordered after the per frame functional groups sequence and before the pixel
data, so that reading the dataset has to carry on past the sequence to reach it."""

FILE_SETTINGS = {
    "sparse_no_bot": {
        "name": "sparse_no_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.EMPTY,
    },
    "sparse_with_bot": {
        "name": "sparse_with_bot.dcm",
        "tile_type": TileType.SPARSE,
        "bot_type": OffsetTableType.BASIC,
    },
    "full_no_bot_": {
        "name": "full_no_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.EMPTY,
    },
    "full_with_bot": {
        "name": "full_with_bot.dcm",
        "tile_type": TileType.FULL,
        "bot_type": OffsetTableType.BASIC,
    },
}


@pytest.fixture()
def meta_dataset():
    yield create_meta_dataset()


@pytest.fixture()
def padded_test_frame():
    yield TESTFRAME + b"\x00" * (len(TESTFRAME) % 2)


@pytest.fixture()
def dataset(name: str):
    file_setting = FILE_SETTINGS[name]
    dataset = create_main_dataset(file_setting["tile_type"], file_setting["bot_type"])
    yield dataset


@pytest.fixture()
def test_file(name: str, dataset: Dataset, meta_dataset: FileMetaDataset):
    file_setting = FILE_SETTINGS[name]
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath(file_setting["name"])
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))
        yield reader
        reader.close()


@pytest.fixture()
def file_with_element_after_sequence(
    name: str, dataset: Dataset, meta_dataset: FileMetaDataset
):
    """A file holding an element ordered between the per frame sequence and the pixel
    data, which the read of the dataset has to pick up rather than stop at."""
    dataset.add_new(TAG_AFTER_SEQUENCE, "LO", "After the sequence")
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath(FILE_SETTINGS[name]["name"])
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))
        yield reader
        reader.close()


@pytest.fixture()
def file_with_unreadable_sequence(meta_dataset: FileMetaDataset):
    """A file whose per frame sequence cannot be read without parsing it: it holds more
    items than the instance has frames, so the positions cannot be matched to frames."""
    dataset = create_main_dataset(TileType.SPARSE, OffsetTableType.BASIC)
    dataset.PerFrameFunctionalGroupsSequence.append(
        deepcopy(dataset.PerFrameFunctionalGroupsSequence[0])
    )
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath("unreadable_sequence.dcm")
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))
        yield reader
        reader.close()


@pytest.fixture()
def file_without_sequence_with_element_after_it(meta_dataset: FileMetaDataset):
    """The same, for a file with no per frame sequence at all, where the read stops at
    the element itself rather than at the sequence."""
    dataset = create_main_dataset(TileType.FULL, OffsetTableType.BASIC)
    del dataset[PerFrameFunctionalGroupsSequenceTag]
    dataset.add_new(TAG_AFTER_SEQUENCE, "LO", "After the sequence")
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath("no_sequence.dcm")
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))
        yield reader
        reader.close()


SEQUENCES_WRITTEN_WITHOUT_A_LENGTH = (
    "SpecimenDescriptionSequence",
    "AcquisitionContextSequence",
    "DimensionOrganizationSequence",
    "ContainerTypeCodeSequence",
    # Needed for opening, so read even with undefined length.
    "TotalPixelMatrixOriginSequence",
    "SharedFunctionalGroupsSequence",
    "OpticalPathSequence",
)


@pytest.fixture()
def file_with_sequences_without_a_length(meta_dataset: FileMetaDataset):
    """A file with undefined length sequences, which can be stepped over.

    Also with an ICC profile large enough to be deferred, so that both kinds of
    deferred element are in play.
    """
    dataset = create_main_dataset(TileType.FULL, OffsetTableType.BASIC)
    for keyword in SEQUENCES_WRITTEN_WITHOUT_A_LENGTH:
        dataset[Tag(keyword)].is_undefined_length = True
    dataset.OpticalPathSequence[0].ICCProfile = bytes(
        WsiDicomReader.DEFERRED_VALUE_SIZE + 1
    )
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir).joinpath("sequences_without_a_length.dcm")
        create_layer_file(path, dataset, meta_dataset)
        reader = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))
        yield reader, path
        reader.close()


@pytest.mark.unittest
class TestSteppingOverSequences:
    def test_the_sequences_opening_does_not_read_are_stepped_over(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """Only sequences not needed for opening are stepped over.

        Uses private state, as a stepped over sequence read back cannot be told
        apart from one read directly.
        """
        # Arrange
        reader, _ = file_with_sequences_without_a_length

        # Act
        stepped_over = {
            sequence.tag for sequence in reader._deferred_reader._root_deferred_elements
        }

        # Assert
        assert stepped_over == {
            Tag("SpecimenDescriptionSequence"),
            Tag("AcquisitionContextSequence"),
            Tag("DimensionOrganizationSequence"),
            Tag("ContainerTypeCodeSequence"),
        }

    def test_a_sequence_that_cannot_be_walked_is_read_instead(
        self, meta_dataset: FileMetaDataset
    ):
        """A sequence that cannot be stepped over is read by pydicom instead."""
        # Arrange
        # An item delimiter where an item belongs, which pydicom reads as an empty
        # item but the walker rejects.
        dataset = create_main_dataset(TileType.FULL, OffsetTableType.BASIC)
        dataset[Tag("SpecimenDescriptionSequence")].is_undefined_length = True
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir).joinpath("unwalkable.dcm")
            create_layer_file(path, dataset, meta_dataset)
            written = path.read_bytes()
            # The whole header, so that the tag bytes in some value are not taken
            # for it.
            sequence_header = b"\x40\x00\x60\x05" + b"SQ\x00\x00" + b"\xff\xff\xff\xff"
            items_start = written.index(sequence_header) + len(sequence_header)
            path.write_bytes(
                written[:items_start]
                + b"\xfe\xff\x0d\xe0\x00\x00\x00\x00"
                + written[items_start:]
            )
            reader = WsiDicomReader(
                WsiDicomReadIO(open(path, "rb"), filepath=UPath(path))
            )

            # Act
            with reader:
                read = reader.dataset.as_dataset()

            # Assert
            assert Tag("SpecimenDescriptionSequence") in read
            assert read.TotalPixelMatrixColumns == dataset.TotalPixelMatrixColumns

    def test_completed_dataset_writes_back_framed_as_it_was_read(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """A stepped over sequence is written back with a single delimiter.

        The sequences are not accessed before writing, so pydicom writes the raw
        bytes, where a delimiter included in the value would be written twice.
        """
        # Arrange
        reader, path = file_with_sequences_without_a_length
        original = dcmread(path, stop_before_pixels=True)
        read = reader.dataset.as_dataset()
        read.file_meta = original.file_meta
        buffer = BytesIO()

        # Act
        read.save_as(buffer, enforce_file_format=True)

        # Assert
        written = dcmread(BytesIO(buffer.getvalue()))
        # Check tags only, as accessing elements would fail on a stray delimiter.
        assert not any(tag.group == 0xFFFE for tag in written.keys())  # noqa: SIM118
        for keyword in SEQUENCES_WRITTEN_WITHOUT_A_LENGTH:
            assert written[Tag(keyword)].value == original[Tag(keyword)].value

    @pytest.mark.parametrize(
        "failing_read", [1, 2], ids=["deferred value", "stepped over sequence"]
    )
    def test_a_failed_read_leaves_the_element_to_be_read_again(
        self,
        file_with_sequences_without_a_length: tuple[WsiDicomReader, Path],
        monkeypatch: pytest.MonkeyPatch,
        failing_read: int,
    ):
        """An element whose read failed is read when the dataset is next completed.

        The deferred ICC profile is read first and the stepped over sequences after
        it, so failing the first read covers one kind and the second the other.
        """
        # Arrange
        reader, _ = file_with_sequences_without_a_length
        reader._deferred_reader.seek_to_pixel_data()
        stream = reader._stream
        read = stream.read
        reads = 0

        def fail_once(*args: Any, **kwargs: Any) -> bytes:
            nonlocal reads
            reads += 1
            if reads < failing_read:
                return read(*args, **kwargs)
            monkeypatch.setattr(stream, "read", read)
            raise OSError("Transient failure")

        monkeypatch.setattr(stream, "read", fail_once)
        with pytest.raises(OSError, match="Transient failure"):
            reader.dataset.as_dataset()

        # Act
        completed = reader.dataset.as_dataset()

        # Assert
        assert len(completed.OpticalPathSequence[0].ICCProfile) > 0
        for keyword in SEQUENCES_WRITTEN_WITHOUT_A_LENGTH:
            assert Tag(keyword) in completed

    def test_a_stepped_over_sequence_is_read_back_in_the_character_set(
        self, meta_dataset: FileMetaDataset
    ):
        """A value in a stepped over sequence is decoded with the character set."""
        # Arrange
        dataset = create_main_dataset(TileType.FULL, OffsetTableType.BASIC)
        dataset.SpecificCharacterSet = "ISO_IR 100"
        dataset.SpecimenDescriptionSequence[0].SpecimenIdentifier = "Spécimen"
        dataset[Tag("SpecimenDescriptionSequence")].is_undefined_length = True
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir).joinpath("latin1.dcm")
            create_layer_file(path, dataset, meta_dataset)
            reader = WsiDicomReader(
                WsiDicomReadIO(open(path, "rb"), filepath=UPath(path))
            )

            # Act
            with reader:
                read = reader.dataset.as_dataset()

            # Assert
            assert read.SpecimenDescriptionSequence[0].SpecimenIdentifier == "Spécimen"

    def test_completed_dataset_holds_what_was_stepped_over(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """The complete dataset contains the stepped over sequences."""
        # Arrange
        reader, path = file_with_sequences_without_a_length
        written = dcmread(path, stop_before_pixels=True)

        # Act
        read = reader.dataset.as_dataset()

        # Assert
        for keyword in SEQUENCES_WRITTEN_WITHOUT_A_LENGTH:
            assert read[Tag(keyword)].value == written[Tag(keyword)].value

    def test_stepped_over_sequences_keep_the_framing_they_were_written_with(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """Stepped over sequences are read back with undefined length."""
        # Arrange
        reader, _ = file_with_sequences_without_a_length

        # Act
        read = reader.dataset.as_dataset()

        # Assert
        assert read[Tag("SpecimenDescriptionSequence")].is_undefined_length
        assert read[Tag("AcquisitionContextSequence")].is_undefined_length

    def test_sequences_read_while_opening_are_not_stepped_over(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """Sequences needed for opening are read without completing the dataset."""
        # Arrange
        reader, _ = file_with_sequences_without_a_length

        # Act
        coordinate_system = reader.dataset.image_coordinate_system
        pixel_spacing = reader.dataset.pixel_spacing

        # Assert
        assert coordinate_system is not None
        assert pixel_spacing is not None

    def test_a_sequence_after_where_the_per_frame_sequence_belongs_is_stepped_over(
        self, meta_dataset: FileMetaDataset
    ):
        """In a file with no per frame sequence, the read on to the pixel data steps
        over sequences too."""
        # Arrange
        # A private sequence ordered after the per frame functional groups sequence,
        # as in a file from PixelMed.
        private_sequence_tag = Tag(0x7FDF, 0x1001)
        dataset = create_main_dataset(TileType.FULL, OffsetTableType.BASIC)
        del dataset[PerFrameFunctionalGroupsSequenceTag]
        item = Dataset()
        item.PatientID = "In a private sequence"
        dataset.add_new(Tag(0x7FDF, 0x0010), "LO", "PRIVATE CREATOR")
        dataset.add_new(private_sequence_tag, "SQ", [item])
        dataset[private_sequence_tag].is_undefined_length = True
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir).joinpath("private_sequence_after.dcm")
            create_layer_file(path, dataset, meta_dataset)
            reader = WsiDicomReader(
                WsiDicomReadIO(open(path, "rb"), filepath=UPath(path))
            )

            # Act
            with reader:
                stepped_over = {
                    sequence.tag
                    for sequence in reader._deferred_reader._root_deferred_elements
                }
                read = reader.dataset.as_dataset()

            # Assert
            assert private_sequence_tag in stepped_over
            assert read[private_sequence_tag].value[0].PatientID == (
                "In a private sequence"
            )

    def test_comparing_to_something_else_reads_nothing(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """Comparing to what cannot be equal does not complete the dataset."""
        # Arrange
        reader, _ = file_with_sequences_without_a_length
        reader.close()

        # Act & Assert
        assert reader.dataset != None  # noqa: E711
        assert reader.dataset != "a string"
        assert reader.dataset == reader.dataset

    def test_reading_a_frame_does_not_need_the_stepped_over_sequences(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """The frame index is read without the stepped over sequences."""
        # Arrange
        reader, _ = file_with_sequences_without_a_length

        # Act
        frame_index = reader.frame_index

        # Assert
        assert len(frame_index) > 0

    def test_datasets_of_the_same_file_are_equal_before_either_is_completed(
        self, file_with_sequences_without_a_length: tuple[WsiDicomReader, Path]
    ):
        """Equality does not depend on whether a dataset has been completed."""
        # Arrange
        reader, path = file_with_sequences_without_a_length
        other = WsiDicomReader(WsiDicomReadIO(open(path, "rb"), filepath=UPath(path)))

        # Act
        with other:
            are_equal = reader.dataset == other.dataset

        # Assert
        assert are_equal


@pytest.mark.unittest
class TestWWsiDicomReader:
    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_offset_table_type_property(
        self, test_file: WsiDicomReader, settings: dict[str, Any]
    ):
        # Arrange

        # Act
        offset_table_type = test_file.offset_table_type

        # Assert
        assert offset_table_type == settings["bot_type"]

    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_tile_type_property(
        self, test_file: WsiDicomReader, settings: dict[str, Any]
    ):
        # Arrange

        # Act
        tile_type = test_file.dataset.tile_type

        # Assert
        assert tile_type == settings["tile_type"]

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_dataset_property(self, test_file: WsiDicomReader):
        # Arrange
        path = test_file.filepath
        assert isinstance(path, Path)

        # Act
        expected = dcmread(path, stop_before_pixels=True)
        read = test_file.dataset.as_dataset()
        if PerFrameFunctionalGroupsSequenceTag not in read:
            # The sequence was read rather than parsed, so it is not in the dataset.
            del expected[PerFrameFunctionalGroupsSequenceTag]

        # Assert
        assert read == expected

    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_frame_positions_match_parsed_sequence(
        self, test_file: WsiDicomReader, settings: dict[str, Any]
    ):
        """Positions read out of the sequence have to say what parsing it says."""
        # Arrange
        path = test_file.filepath
        assert isinstance(path, Path)
        # The sequence holds the tile positions and nothing else that is wanted, so
        # it is never kept in the dataset, whatever the instance is tiled as.
        assert PerFrameFunctionalGroupsSequenceTag not in test_file.dataset.as_dataset()
        if settings["tile_type"] is TileType.FULL:
            # The per frame groups of a tiled full image hold no tile positions.
            return

        # Act
        positions = test_file.dataset.frame_positions
        parsed = WsiDataset(dcmread(path, stop_before_pixels=True))
        parsed_positions = parsed.frame_positions

        # Assert
        assert list(positions.columns) == list(parsed_positions.columns)
        assert list(positions.rows) == list(parsed_positions.rows)
        assert (positions.z_offsets is None) == (parsed_positions.z_offsets is None)
        if positions.z_offsets is not None and parsed_positions.z_offsets is not None:
            assert list(positions.z_offsets) == list(parsed_positions.z_offsets)
        assert (positions.optical_path_identifiers is None) == (
            parsed_positions.optical_path_identifiers is None
        )
        if (
            positions.optical_path_identifiers is not None
            and parsed_positions.optical_path_identifiers is not None
        ):
            assert list(positions.optical_path_identifiers) == list(
                parsed_positions.optical_path_identifiers
            )

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_element_after_sequence_is_read(
        self, file_with_element_after_sequence: WsiDicomReader
    ):
        """The dataset read stops at the sequence, so it has to carry on past it."""
        # Arrange
        reader = file_with_element_after_sequence

        # Act
        dataset = reader.dataset.as_dataset()

        # Assert
        assert dataset[TAG_AFTER_SEQUENCE].value == "After the sequence"

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_element_after_sequence_leaves_pixel_data_findable(
        self, file_with_element_after_sequence: WsiDicomReader, padded_test_frame: bytes
    ):
        """An element after the sequence must not be taken for the pixel data."""
        # Arrange
        reader = file_with_element_after_sequence

        # Act
        frame = reader.read_frame(0)

        # Assert
        assert frame == padded_test_frame

    @pytest.mark.parametrize(["name", "settings"], FILE_SETTINGS.items())
    def test_frame_positions_of_a_tiled_full_image(
        self, test_file: WsiDicomReader, settings: dict[str, Any]
    ):
        """The per frame groups of a tiled full image state no tile positions, so
        asking for them has to say so rather than answer from the shared groups or
        from an empty sequence."""
        # Arrange
        dataset = test_file.dataset
        if settings["tile_type"] is not TileType.FULL:
            return

        # Act & Assert
        with pytest.raises(WsiDicomError):
            _ = dataset.frame_positions

    def test_unreadable_sequence_is_parsed_instead(
        self, file_with_unreadable_sequence: WsiDicomReader
    ):
        """Where the bytes cannot be searched, parsing the sequence gives the same."""
        # Arrange
        reader = file_with_unreadable_sequence
        path = reader.filepath
        assert isinstance(path, Path)
        parsed = WsiDataset(dcmread(path, stop_before_pixels=True))

        # Act
        positions = reader.dataset.frame_positions

        # Assert
        assert list(positions.columns) == list(parsed.frame_positions.columns)
        assert list(positions.rows) == list(parsed.frame_positions.rows)

    def test_unreadable_sequence_is_not_kept_in_the_dataset(
        self, file_with_unreadable_sequence: WsiDicomReader
    ):
        """Parsed for the positions and let go of, as a searched one is."""
        # Arrange
        reader = file_with_unreadable_sequence

        # Act
        dataset = reader.dataset.as_dataset()

        # Assert
        assert PerFrameFunctionalGroupsSequenceTag not in dataset

    def test_unreadable_sequence_leaves_pixel_data_findable(
        self, file_with_unreadable_sequence: WsiDicomReader, padded_test_frame: bytes
    ):
        # Arrange
        reader = file_with_unreadable_sequence

        # Act
        frame = reader.read_frame(0)

        # Assert
        assert frame == padded_test_frame

    def test_element_after_missing_sequence_is_read(
        self, file_without_sequence_with_element_after_it: WsiDicomReader
    ):
        """Without a sequence the read stops at the element itself, not at the pixel
        data, so it still has to carry on."""
        # Arrange
        reader = file_without_sequence_with_element_after_it

        # Act
        dataset = reader.dataset.as_dataset()

        # Assert
        assert dataset[TAG_AFTER_SEQUENCE].value == "After the sequence"

    def test_element_after_missing_sequence_leaves_pixel_data_findable(
        self,
        file_without_sequence_with_element_after_it: WsiDicomReader,
        padded_test_frame: bytes,
    ):
        # Arrange
        reader = file_without_sequence_with_element_after_it

        # Act
        frame = reader.read_frame(0)

        # Assert
        assert frame == padded_test_frame

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_image_type_property(
        self,
        test_file: WsiDicomReader,
    ):
        # Arrange

        # Act
        image_type = test_file.image_type

        # Assert
        assert image_type == ImageType.VOLUME

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_uids_property(self, test_file: WsiDicomReader, dataset: Dataset):
        # Arrange

        # Act
        uids = test_file.uids

        # Assert
        assert uids.instance == dataset.SOPInstanceUID
        assert uids.concatenation == getattr(
            dataset, "SOPInstanceUIDOfConcatenationSource", None
        )
        assert uids.slide.frame_of_reference == dataset.FrameOfReferenceUID
        assert uids.slide.study_instance == dataset.StudyInstanceUID
        assert uids.slide.series_instance == dataset.SeriesInstanceUID

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_transfer_syntax_property(
        self, test_file: WsiDicomReader, meta_dataset: FileMetaDataset
    ):
        # Arrange

        # Act
        transfer_syntax = test_file.transfer_syntax

        # Assert
        assert transfer_syntax == meta_dataset.TransferSyntaxUID

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_frame_offset_property(self, test_file: WsiDicomReader):
        # Arrange

        # Act
        frame_offset = test_file.frame_offset

        # Assert
        assert frame_offset == 0

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_frame_count_property(self, test_file: WsiDicomReader):
        # Arrange

        # Act
        frame_count = test_file.frame_count

        # Assert
        assert frame_count == 1

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_read_frame_before_first_frame_in_file(self, test_file: WsiDicomReader):
        # Arrange
        before_first_frame = test_file.frame_offset - 1

        # Act & Assert
        with pytest.raises(WsiDicomOutOfBoundsError):
            test_file.read_frame(before_first_frame)

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_read_frame_after_last_frame_in_file(self, test_file: WsiDicomReader):
        # Arrange
        after_last_frame = test_file.frame_offset + len(test_file.frame_index)

        # Act & Assert
        with pytest.raises(WsiDicomOutOfBoundsError):
            test_file.read_frame(after_last_frame)

    @pytest.mark.parametrize("name", FILE_SETTINGS.keys())
    def test_read_frame(self, test_file: WsiDicomReader, padded_test_frame: bytes):
        # Arrange

        # Act
        frame = test_file.read_frame(0)

        # Assert
        assert frame == padded_test_frame
