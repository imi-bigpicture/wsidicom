#    Copyright 2024 SECTRA AB
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

import pytest
from pydicom import Dataset
from pydicom.sequence import Sequence as DicomSequence

from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.tile_index.tile_index import TileIndex


@pytest.fixture
def optical_path_sequence(identifiers: list[str]) -> DicomSequence:
    sequence = DicomSequence()
    for identifier in identifiers:
        item = Dataset()
        item.OpticalPathIdentifier = identifier
        sequence.append(item)
    return sequence


@pytest.mark.unittest
class TestTileIndex:
    """Optical paths must be read in Optical Path Sequence order (DICOM PS3.3
    C.7.6.17.3), as that defines the TILED_FULL frame order. They must not be
    sorted or set-ordered, which would misattribute frames."""

    @pytest.mark.parametrize(
        ["identifiers", "expected"],
        [
            (["1", "0"], ["1", "0"]),  # preserves sequence order, not sorted
            (["2", "0", "2"], ["2", "0"]),  # dedups, keeping first-seen
        ],
    )
    def test_read_optical_paths_from_dataset(
        self, optical_path_sequence: DicomSequence, expected: list[str]
    ):
        # Arrange
        dataset = WsiDataset()
        dataset.OpticalPathSequence = optical_path_sequence

        # Act
        paths = TileIndex._read_optical_paths_from_datasets([dataset])

        # Assert
        assert paths == expected

    @pytest.mark.parametrize("identifiers", [["1", "0"]])
    def test_read_optical_paths_dedups_across_datasets(
        self, optical_path_sequence: DicomSequence
    ):
        # Arrange — the same paths appear in two datasets.
        first = WsiDataset()
        first.OpticalPathSequence = optical_path_sequence
        second = WsiDataset()
        second.OpticalPathSequence = optical_path_sequence

        # Act
        paths = TileIndex._read_optical_paths_from_datasets([first, second])

        # Assert — deduped, first-seen order preserved.
        assert paths == ["1", "0"]

    def test_read_optical_paths_defaults_to_zero_when_absent(self):
        # Arrange
        dataset = WsiDataset()

        # Act
        paths = TileIndex._read_optical_paths_from_datasets([dataset])

        # Assert
        assert paths == ["0"]
