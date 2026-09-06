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

"""A dataset gives out what the file states, whatever it deferred reading it."""

import pytest
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag

from wsidicom.instance.dataset import DeferredDatasetReader, WsiDataset

ICC_PROFILE_TAG = Tag(0x0028, 0x2000)
OPTICAL_PATH_SEQUENCE_TAG = Tag(0x0048, 0x0105)
PROFILE = b"a profile" * 1000


class FakeDeferredDatasetReader(DeferredDatasetReader):
    """Puts a value into the item it belongs in, counting the times it is asked."""

    def __init__(self, item: Dataset):
        self._item = item
        self.reads = 0

    def complete_dataset(self) -> None:
        if ICC_PROFILE_TAG in self._item:
            return
        self.reads += 1
        self._item[ICC_PROFILE_TAG] = DataElement(ICC_PROFILE_TAG, "OB", PROFILE)

    def read_frame_positions(self) -> None:
        return None


def create_dataset() -> tuple[WsiDataset, FakeDeferredDatasetReader]:
    """A dataset holding an optical path whose profile was deferred."""
    item = Dataset()
    item.OpticalPathIdentifier = "1"
    dataset = Dataset()
    dataset[OPTICAL_PATH_SEQUENCE_TAG] = DataElement(
        OPTICAL_PATH_SEQUENCE_TAG, "SQ", Sequence([item])
    )
    reader = FakeDeferredDatasetReader(item)
    return WsiDataset(dataset, deferred_reader=reader), reader


@pytest.mark.unittest
class TestDeferredElements:
    def test_reading_an_attribute_does_not_read_a_deferred_value(self):
        """The way to a tile goes through the attributes here, so they are what must
        not pay for a value that is only wanted by something describing the whole
        instance."""
        # Arrange
        dataset, reader = create_dataset()

        # Act
        identifiers = dataset.optical_path_identifiers

        # Assert
        assert identifiers == ["1"]
        assert reader.reads == 0

    def test_as_dataset_reads_deferred_elements(self):
        # Arrange
        dataset, reader = create_dataset()

        # Act
        given = dataset.as_dataset()

        # Assert
        assert given.OpticalPathSequence[0].ICCProfile == PROFILE
        assert reader.reads == 1

    def test_replacing_an_attribute_keeps_deferred_elements(self):
        """A copy stands on its own, with no way back to what was deferred, so it
        has to be made from the whole dataset."""
        # Arrange
        dataset, reader = create_dataset()

        # Act
        replaced = dataset.replace({"SeriesDescription": "changed"})

        # Assert
        assert replaced.as_dataset().OpticalPathSequence[0].ICCProfile == PROFILE
        assert reader.reads == 1

    def test_metadata_reads_deferred_elements(self):
        # Arrange
        dataset, reader = create_dataset()

        # Act
        metadata = dataset.pyramid_metadata

        # Assert
        assert metadata is not None
        assert reader.reads == 1

    def test_a_dataset_without_deferred_elements_is_given_out_whole(self):
        """Nothing is deferred when a dataset does not come from a file, so one
        made without a reader gives out what it holds."""
        # Arrange
        item = Dataset()
        item.OpticalPathIdentifier = "1"
        inner = Dataset()
        inner[OPTICAL_PATH_SEQUENCE_TAG] = DataElement(
            OPTICAL_PATH_SEQUENCE_TAG, "SQ", Sequence([item])
        )
        dataset = WsiDataset(inner)

        # Act
        given = dataset.as_dataset()

        # Assert
        assert given is inner
