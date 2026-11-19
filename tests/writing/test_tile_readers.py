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

"""Tests for the tile readers that submit encoded tiles to be recoded."""

from dataclasses import dataclass, field

import numpy as np
import pytest
from decoy import Decoy

from wsidicom.codec import Recoder
from wsidicom.geometry import Point
from wsidicom.instance import ImageData
from wsidicom.thread import CancellationToken, FifoCancelableQueue
from wsidicom.writing.models import EncodingTaskResult, RecodeTask
from wsidicom.writing.tile_readers import (
    CascadingRecodeTileReader,
    RecodeTileReader,
)


@dataclass
class RecordingAccumulator:
    """Accumulator that records the decoded tiles it is given."""

    chain_depth: int = 1
    """Number of generated levels in the cascade starting from this one."""
    added: list[tuple[int, int, int, int, np.ndarray]] = field(default_factory=list)
    """The tiles added, as (x, y, z, path, tile)."""

    def add_tile(self, x: int, y: int, z: int, path: int, tile: np.ndarray) -> None:
        self.added.append((x, y, z, path, tile))


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.fixture
def token() -> CancellationToken:
    """Create a cancellation token for queue operations."""
    return CancellationToken()


@pytest.fixture
def positions() -> list[Point]:
    """Two tiles in the first row and one in the second."""
    return [Point(0, 0), Point(1, 0), Point(0, 1)]


def drain(
    queue: FifoCancelableQueue[RecodeTask], token: CancellationToken
) -> list[RecodeTask]:
    """Return the tasks put on a queue, in order."""
    tasks: list[RecodeTask] = []
    while queue.qsize() > 0:
        tasks.append(queue.get(token))
    return tasks


@pytest.mark.unittest
class TestRecodeTileReader:
    def test_submits_one_recode_task_per_row(
        self, decoy: Decoy, token: CancellationToken, positions: list[Point]
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(image_data.get_encoded_tiles(positions, 0.0, "0")).then_return(
            iter([b"tile_0_0", b"tile_1_0", b"tile_0_1"])
        )
        pool_queue: FifoCancelableQueue[RecodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        recoder = decoy.mock(cls=Recoder)
        reader = RecodeTileReader(2, pool_queue, recoder, token)

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        tasks = drain(pool_queue, token)
        assert [task.tiles for task in tasks] == [
            [b"tile_0_0", b"tile_1_0"],
            [b"tile_0_1"],
        ]
        assert [
            (task.coordinates.level, task.coordinates.x_index, task.coordinates.y_index)
            for task in tasks
        ] == [(2, 0, 0), (2, 0, 1)]
        assert all(task.output_queue is output_queue for task in tasks)
        assert all(task.recoder is recoder for task in tasks)

    def test_does_not_write_tiles_directly(
        self, decoy: Decoy, token: CancellationToken, positions: list[Point]
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(image_data.get_encoded_tiles(positions, 0.0, "0")).then_return(
            iter([b"tile_0_0", b"tile_1_0", b"tile_0_1"])
        )
        pool_queue: FifoCancelableQueue[RecodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        reader = RecodeTileReader(0, pool_queue, decoy.mock(cls=Recoder), token)

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        assert output_queue.qsize() == 0


@pytest.mark.unittest
class TestCascadingRecodeTileReader:
    def test_submits_encoded_and_feeds_decoded_to_accumulator(
        self, decoy: Decoy, token: CancellationToken, positions: list[Point]
    ) -> None:
        # Arrange
        decoded = [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3)]
        image_data = decoy.mock(cls=ImageData)
        decoy.when(
            image_data.get_encoded_and_decoded_tiles(positions, 0.0, "0")
        ).then_return(
            iter(
                zip(
                    [b"tile_0_0", b"tile_1_0", b"tile_0_1"],
                    decoded,
                    strict=True,
                )
            )
        )
        pool_queue: FifoCancelableQueue[RecodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        accumulator = RecordingAccumulator()
        recoder = decoy.mock(cls=Recoder)
        reader = CascadingRecodeTileReader(0, pool_queue, recoder, accumulator, token)

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        tasks = drain(pool_queue, token)
        assert [task.tiles for task in tasks] == [
            [b"tile_0_0", b"tile_1_0"],
            [b"tile_0_1"],
        ]
        assert all(task.recoder is recoder for task in tasks)
        assert output_queue.qsize() == 0
        assert [(x, y) for x, y, _, _, _ in accumulator.added] == [
            (0, 0),
            (1, 0),
            (0, 1),
        ]
        for (_, _, _, _, tile), expected in zip(
            accumulator.added, decoded, strict=True
        ):
            assert np.array_equal(tile, expected)

    def test_reports_accumulator_chain_depth(
        self, decoy: Decoy, token: CancellationToken
    ) -> None:
        # Arrange
        pool_queue: FifoCancelableQueue[RecodeTask] = FifoCancelableQueue()
        accumulator = RecordingAccumulator(chain_depth=3)

        # Act
        reader = CascadingRecodeTileReader(
            0, pool_queue, decoy.mock(cls=Recoder), accumulator, token
        )

        # Assert
        assert reader.accumulator_chain_depth == 3
