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

"""Unit tests for InstanceFileWriter's concatenated-flag decision and offset
advance. Both collaborators (splitter and factory) are mocked, so the writer is
exercised independently of any concrete splitter."""

from pathlib import Path

import pytest
from decoy import Decoy, matchers
from upath import UPath

from wsidicom.file.file_writer import (
    InstanceFileWriter,
    PartFactory,
    PartSplitter,
)
from wsidicom.file.io import WsiDicomWriter
from wsidicom.instance import WsiDataset


@pytest.fixture
def level_frames() -> int:
    """Frame count of the single-part scenario."""
    return 10


@pytest.fixture
def split_frames() -> int:
    """Frame count of the two-part scenario."""
    return 4


@pytest.fixture
def split_part() -> int:
    """Part size of the two-part scenario (so `split_frames` is two parts)."""
    return 2


@pytest.fixture
def part_factory(decoy: Decoy) -> PartFactory:
    return decoy.mock(cls=PartFactory)


@pytest.fixture
def part_splitter(decoy: Decoy) -> PartSplitter:
    return decoy.mock(cls=PartSplitter)


@pytest.fixture
def whole_level_splitter(
    decoy: Decoy, part_splitter: PartSplitter, level_frames: int
) -> PartSplitter:
    """Keeps the whole level in a single part (never cuts)."""
    decoy.when(part_splitter.next_part_frame_count(matchers.Anything())).then_return(
        level_frames
    )
    decoy.when(part_splitter.should_start_new_part(matchers.Anything())).then_return(
        False
    )
    return part_splitter


@pytest.fixture
def two_part_splitter(
    decoy: Decoy, part_splitter: PartSplitter, split_part: int
) -> PartSplitter:
    """Cuts a two-part level, before its 3rd tile."""
    decoy.when(part_splitter.next_part_frame_count(matchers.Anything())).then_return(
        split_part
    )
    decoy.when(part_splitter.should_start_new_part(matchers.Anything())).then_return(
        False, True, False
    )
    return part_splitter


class TestInstanceFileWriter:
    def test_single_part_covering_the_level_is_not_concatenated(
        self,
        decoy: Decoy,
        part_factory: PartFactory,
        whole_level_splitter: PartSplitter,
        level_frames: int,
        tmp_path: Path,
        minimal_dataset: WsiDataset,
    ) -> None:
        # Arrange
        tile = b"t"
        tiles = [tile] * level_frames
        part_writer = decoy.mock(cls=WsiDicomWriter)
        decoy.when(part_writer.filepath).then_return(UPath(tmp_path) / "part.dcm")
        decoy.when(part_factory.open(0, level_frames, concatenated=False)).then_return(
            (part_writer, minimal_dataset)
        )
        writer = InstanceFileWriter(
            part_factory, whole_level_splitter, level_frames, None, UPath(tmp_path)
        )

        # Act
        writer.write_tiles(tiles)
        writer.finalize()

        # Assert
        decoy.verify(whole_level_splitter.reset(), times=0)
        decoy.verify(whole_level_splitter.account(tile), times=level_frames)
        decoy.verify(part_writer.write_tiles(tiles), times=1)
        decoy.verify(part_writer.finalize(minimal_dataset, None), times=1)

    def test_split_marks_each_part_concatenated_and_advances_offset(
        self,
        decoy: Decoy,
        part_factory: PartFactory,
        two_part_splitter: PartSplitter,
        split_frames: int,
        split_part: int,
        tmp_path: Path,
        minimal_dataset: WsiDataset,
    ) -> None:
        # Arrange — two concatenated parts, opened at contiguous offsets
        tile = b"t"
        tiles = [tile] * split_frames
        part_tiles = [tile] * split_part
        part_1_writer = decoy.mock(cls=WsiDicomWriter)
        decoy.when(part_1_writer.filepath).then_return(UPath(tmp_path) / "part_1.dcm")
        part_2_writer = decoy.mock(cls=WsiDicomWriter)
        decoy.when(part_2_writer.filepath).then_return(UPath(tmp_path) / "part_2.dcm")
        decoy.when(part_factory.open(0, split_part, concatenated=True)).then_return(
            (part_1_writer, minimal_dataset)
        )
        decoy.when(
            part_factory.open(split_part, split_part, concatenated=True)
        ).then_return((part_2_writer, minimal_dataset))
        writer = InstanceFileWriter(
            part_factory, two_part_splitter, split_frames, None, UPath(tmp_path)
        )

        # Act
        writer.write_tiles(tiles)
        writer.finalize()

        # Assert — one cut, every tile accounted, each part's batch streamed
        decoy.verify(two_part_splitter.reset(), times=1)
        decoy.verify(two_part_splitter.account(tile), times=split_frames)
        decoy.verify(part_1_writer.write_tiles(part_tiles), times=1)
        decoy.verify(part_2_writer.write_tiles(part_tiles), times=1)
