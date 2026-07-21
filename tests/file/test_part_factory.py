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

"""Unit tests for PartFactory dataset stamping. Parts are written to a temp dir;
each test inspects the stamped dataset the factory returns."""

import itertools
from pathlib import Path

import pytest
from pydicom.uid import JPEGBaseline8Bit
from upath import UPath

from wsidicom.file import OffsetTableType
from wsidicom.file.file_writer import PartFactory
from wsidicom.instance import WsiDataset
from wsidicom.metadata.uid_generator import CallableUidGenerator


@pytest.fixture
def base_dataset() -> WsiDataset:
    """A minimal WSI dataset complete enough for the writer to open an instance."""
    dataset = WsiDataset()
    dataset.NumberOfFrames = 1
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.SamplesPerPixel = 3
    dataset.BitsAllocated = 8
    return dataset


class TestPartFactory:
    def test_concatenated_parts_share_identity_and_partition_contiguously(
        self, base_dataset: WsiDataset, tmp_path: Path
    ) -> None:
        # Arrange
        part_total = 2
        first_count, second_count = 100, 50
        factory = PartFactory(
            base_dataset,
            CallableUidGenerator(),
            UPath(tmp_path),
            None,
            JPEGBaseline8Bit,
            OffsetTableType.BASIC,
            itertools.count(1),
            part_total,
        )

        # Act
        _, first_part_dataset = factory.open(0, first_count, concatenated=True)
        _, second_part_dataset = factory.open(
            first_count, second_count, concatenated=True
        )

        # Assert — one concatenation identity shared by every part
        assert (
            first_part_dataset.ConcatenationUID == second_part_dataset.ConcatenationUID
        )
        assert (
            first_part_dataset.SOPInstanceUIDOfConcatenationSource
            == second_part_dataset.SOPInstanceUIDOfConcatenationSource
        )
        # sequential, 1-based part numbers
        assert first_part_dataset.InConcatenationNumber == 1
        assert second_part_dataset.InConcatenationNumber == 2
        # per-part frame counts, and offsets that partition contiguously
        assert first_part_dataset.NumberOfFrames == first_count
        assert second_part_dataset.NumberOfFrames == second_count
        assert first_part_dataset.ConcatenationFrameOffsetNumber == 0
        assert second_part_dataset.ConcatenationFrameOffsetNumber == first_count
        # each part is its own SOP Instance
        assert first_part_dataset.SOPInstanceUID != second_part_dataset.SOPInstanceUID

    def test_total_number_emitted_when_known(
        self, base_dataset: WsiDataset, tmp_path: Path
    ) -> None:
        # Arrange
        part_total = 3
        factory = PartFactory(
            base_dataset,
            CallableUidGenerator(),
            UPath(tmp_path),
            None,
            JPEGBaseline8Bit,
            OffsetTableType.BASIC,
            itertools.count(1),
            part_total,
        )

        # Act
        _, part_dataset = factory.open(0, 100, concatenated=True)

        # Assert
        assert part_dataset.InConcatenationTotalNumber == part_total

    def test_total_number_omitted_when_unknown(
        self, base_dataset: WsiDataset, tmp_path: Path
    ) -> None:
        # Arrange — byte-size splitting doesn't know the part count up front.
        factory = PartFactory(
            base_dataset,
            CallableUidGenerator(),
            UPath(tmp_path),
            None,
            JPEGBaseline8Bit,
            OffsetTableType.BASIC,
            itertools.count(1),
            None,
        )

        # Act
        _, part_dataset = factory.open(0, 100, concatenated=True)

        # Assert
        assert "InConcatenationTotalNumber" not in part_dataset

    def test_non_concatenated_part_reuses_base_without_concat_attrs(
        self, base_dataset: WsiDataset, tmp_path: Path
    ) -> None:
        # Arrange
        factory = PartFactory(
            base_dataset,
            CallableUidGenerator(),
            UPath(tmp_path),
            None,
            JPEGBaseline8Bit,
            OffsetTableType.BASIC,
            itertools.count(1),
            None,
        )

        # Act — sole part of the level
        _, part_dataset = factory.open(0, 500, concatenated=False)

        # Assert — base reused in place (no copy), no concatenation attributes
        assert part_dataset is base_dataset
        assert "ConcatenationUID" not in part_dataset
        assert part_dataset.NumberOfFrames == 500
