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
from wsidicom.file.io.wsidicom_io import WsiDicomIO
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
        reader = WsiDicomReader(WsiDicomIO(open(path, "rb"), filepath=UPath(path)))
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
        reader = WsiDicomReader(WsiDicomIO(open(path, "rb"), filepath=UPath(path)))
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
        reader = WsiDicomReader(WsiDicomIO(open(path, "rb"), filepath=UPath(path)))
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
        reader = WsiDicomReader(WsiDicomIO(open(path, "rb"), filepath=UPath(path)))
        yield reader
        reader.close()


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
        if settings["tile_type"] is TileType.FULL:
            # The per frame groups of a tiled full image hold no tile positions, so
            # there is nothing to read and the sequence is parsed.
            assert PerFrameFunctionalGroupsSequenceTag in test_file.dataset.as_dataset()
            return
        assert PerFrameFunctionalGroupsSequenceTag not in test_file.dataset.as_dataset()

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
        """When the positions cannot be read, the sequence is parsed as it always was."""
        # Arrange
        reader = file_with_unreadable_sequence

        # Act
        dataset = reader.dataset.as_dataset()

        # Assert
        assert len(dataset.PerFrameFunctionalGroupsSequence) == 2

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
